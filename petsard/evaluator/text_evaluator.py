"""
Text evaluator -- parse natural-language data, then run native PETsARD evaluators.

PETsARD's evaluators are column-addressed: `secret`, `target`, `aux_cols` and
`n_cols` all name columns, and every free-text column is treated as a categorical
label (`create_metadata()` maps object -> categorical). Data generated as prose
therefore either errors out or scores meaninglessly -- CategoryAdherence and
mpuccs both reach their "good" value only when synthetic rows are verbatim copies
of training rows, which is backwards for a generator.

This evaluator closes that gap with two inputs:

  text_parser   path to a .py file exposing  def text_parser(df) -> pd.DataFrame
  Evaluator     a nested block mirroring PETsARD's own Evaluator config, holding
                any number of native evaluators, each with its own arguments

It parses ori / syn / control identically, unifies their dtypes, then fans the
parsed frames out to every nested evaluator and merges the results.

Wire it up with the native custom_method hook -- no PETsARD core changes::

    Evaluator:
      text_eval:
        method: custom_method
        module_path: petsard/evaluator/text_evaluator.py
        class_name: TextEvaluator
        text_parser: petsard/synthesizer/LLM_synth/utils/bloodpressure_text_parser.py
        Evaluator:
          validity:
            method: sdmetrics-diagnosticreport
          inference_risk:
            method: anonymeter-inference
            secret: age

Two classes are exported because PETsARD reads `REQUIRED_INPUT_KEYS` off the
class before instantiating it, so the requirement cannot be derived from config:

  TextEvaluator             ori + syn          -- SDMetrics, mpuccs
  TextEvaluatorWithControl  ori + syn + control -- Anonymeter, MLUtility
                                                  (requires a Splitter)

Use the nested key `Evaluator`, not `eval_method`: the latter is reserved for the
outer custom_method string and collides during config merge.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd

__all__ = ["TextEvaluator", "TextEvaluatorWithControl"]

DEFAULT_PARSER_ENTRY: str = "text_parser"

# Keys consumed here. Everything else is ignored rather than forwarded, because
# per-evaluator arguments belong inside the nested Evaluator block.
_OWN_KEYS = frozenset(
    {
        "eval_method",
        "method",
        "module_path",
        "class_name",
        "text_parser",
        "text_parser_entry",
        "Evaluator",
        "output_dir",
    }
)


def _load_parser(path: str, entry: str) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Load the parser entry function from a plain .py file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ImportError: If it cannot be executed or has no such function.
        TypeError: If the attribute found is not callable.
    """
    resolved = os.path.abspath(path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(
            f"text_parser not found: {path} (resolved to {resolved}). "
            "Give a path relative to the directory you run PETsARD from."
        )

    spec = importlib.util.spec_from_file_location(
        f"_petsard_text_parser_{os.path.splitext(os.path.basename(resolved))[0]}",
        resolved,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot build an import spec for {resolved}")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ImportError(f"Failed to execute text_parser {resolved}: {exc}") from exc

    if not hasattr(module, entry):
        available = [
            name
            for name in dir(module)
            if callable(getattr(module, name)) and not name.startswith("_")
        ]
        raise ImportError(
            f"{resolved} has no function '{entry}'. Callables found: {available}"
        )

    func = getattr(module, entry)
    if not callable(func):
        raise TypeError(f"'{entry}' in {resolved} is not callable")
    return func


def _unify_dtypes(
    frames: dict[str, pd.DataFrame],
    reference: str = "ori",
    logger: logging.Logger | None = None,
) -> dict[str, pd.DataFrame]:
    """Promote each shared column to a dtype every frame can represent.

    Independent parsing easily produces disagreeing dtypes -- one frame parses
    every age cleanly (int64) while another has a miss (float64) -- and
    `EvaluatorInputConfig.verify_required_inputs` rejects that outright. Column
    order is aligned to the reference frame as well.
    """
    if len(frames) < 2:
        return dict(frames)

    if reference not in frames:
        reference = next(iter(frames))

    result = {name: frame.copy() for name, frame in frames.items()}
    reference_columns = list(result[reference].columns)

    for column in reference_columns:
        present = [name for name, f in result.items() if column in f.columns]
        if len(present) < 2:
            continue

        series_list = [result[name][column] for name in present]
        if len({str(s.dtype) for s in series_list}) == 1:
            continue

        all_numeric = all(
            pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
            for s in series_list
        )
        all_datetime = all(pd.api.types.is_datetime64_any_dtype(s) for s in series_list)

        if all_numeric:
            target = (
                "float64"
                if any(pd.api.types.is_float_dtype(s) for s in series_list)
                else "int64"
            )
        elif all_datetime:
            target = "datetime64[ns]"
        else:
            target = "object"

        if logger:
            logger.debug(f"Unifying column '{column}' across {present} -> {target}")
        for name in present:
            try:
                result[name][column] = result[name][column].astype(target)
            except Exception as exc:
                if logger:
                    logger.warning(
                        f"Could not unify '{column}' in '{name}' to {target}: {exc}"
                    )

    for name, frame in result.items():
        ordered = [c for c in reference_columns if c in frame.columns]
        extra = [c for c in frame.columns if c not in ordered]
        result[name] = frame[ordered + extra]

    return result


class _BaseTextEvaluator:
    """Parse text into typed columns, then delegate to native PETsARD evaluators."""

    REQUIRED_INPUT_KEYS: list[str] = ["ori", "syn"]
    # Union across SDMetrics (global/columnwise/pairwise), Anonymeter and
    # MLUtility (global/details). Never enforced by PETsARD; a granularity no
    # nested evaluator produced comes back as None.
    AVAILABLE_SCORES_GRANULARITY: list[str] = [
        "global",
        "columnwise",
        "pairwise",
        "details",
    ]

    def __init__(self, config: dict) -> None:
        """
        Args:
            config (dict): The YAML block.

                text_parser (str): Path to a .py file exposing
                    `def text_parser(df) -> pd.DataFrame`. Required.
                text_parser_entry (str): Entry function name. Default
                    "text_parser".
                Evaluator (dict): `{name: {method: ..., **evaluator_args}}`.
                    Every key other than `method` is forwarded verbatim to that
                    evaluator. Required, non-empty.
                output_dir (str, optional): Directory to write the parsed frames
                    to for inspection. Nothing is written when omitted.
        """
        self._logger: logging.Logger = logging.getLogger(
            f"PETsARD.{self.__class__.__name__}"
        )
        self.config = config

        parser_path = config.get("text_parser")
        if not parser_path:
            raise ValueError(
                "'text_parser' is required -- point it at a .py file exposing "
                f"'def {DEFAULT_PARSER_ENTRY}(df)'."
            )
        self.text_parser_path: str = parser_path
        self._parser = _load_parser(
            parser_path, config.get("text_parser_entry", DEFAULT_PARSER_ENTRY)
        )

        self._specs: dict[str, dict[str, Any]] = self._read_nested(
            config.get("Evaluator")
        )
        self.output_dir: str | None = config.get("output_dir")

        ignored = sorted(set(config) - _OWN_KEYS)
        if ignored:
            self._logger.warning(
                f"Ignoring top-level keys {ignored}. Per-evaluator arguments belong "
                "inside the nested 'Evaluator' block, next to that evaluator's "
                "'method'."
            )

        self._logger.info(
            f"Parser: {parser_path} | evaluators: "
            + ", ".join(f"{n}({s['method']})" for n, s in self._specs.items())
        )

    def _read_nested(self, nested: Any) -> dict[str, dict[str, Any]]:
        """Validate and normalise the nested Evaluator block."""
        if not nested:
            raise ValueError(
                "'Evaluator' is required -- a nested block of evaluators to run "
                "on the parsed data, e.g.\n"
                "    Evaluator:\n"
                "      validity:\n"
                "        method: sdmetrics-diagnosticreport"
            )
        if not isinstance(nested, Mapping):
            raise TypeError(
                f"'Evaluator' must be a mapping of {{name: {{method: ...}}}}; "
                f"got {type(nested).__name__}"
            )

        specs: dict[str, dict[str, Any]] = {}
        for name, spec in nested.items():
            if not isinstance(spec, Mapping) or "method" not in spec:
                raise ValueError(
                    f"Evaluator '{name}' must be a mapping containing 'method'; "
                    f"got {spec!r}"
                )
            params = dict(spec)
            method = params.pop("method")

            # Anonymeter wants a 2-tuple of lists; YAML produces a list of lists.
            aux = params.get("aux_cols")
            if (
                isinstance(aux, list)
                and len(aux) == 2
                and all(isinstance(part, list) for part in aux)
            ):
                params["aux_cols"] = (aux[0], aux[1])

            specs[name] = {"method": method, "params": params}
        return specs

    # ------------------------------------------------------------------ #

    def eval(self, data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame | None]:
        """Parse every frame, then run each nested evaluator on the result."""
        from petsard.evaluator.evaluator import Evaluator

        frames = {k: v for k, v in data.items() if isinstance(v, pd.DataFrame)}
        self._logger.info(
            "Parsing: " + ", ".join(f"{k}({len(v)} rows)" for k, v in frames.items())
        )

        parsed: dict[str, pd.DataFrame] = {}
        for name, frame in frames.items():
            result = self._parser(frame)
            if not isinstance(result, pd.DataFrame):
                raise TypeError(
                    f"text_parser must return a pd.DataFrame; got "
                    f"{type(result).__name__} for frame '{name}'"
                )
            if len(result) != len(frame):
                raise ValueError(
                    f"text_parser must return one row per input row: frame "
                    f"'{name}' went from {len(frame)} to {len(result)} rows."
                )
            parsed[name] = result

        parsed = _unify_dtypes(parsed, reference="ori", logger=self._logger)
        self._report_coverage(parsed)

        if self.output_dir:
            self._dump(parsed)

        collected: dict[str, dict[str, Any]] = {}
        errors: dict[str, str] = {}

        for name, spec in self._specs.items():
            try:
                evaluator = Evaluator(method=spec["method"], **spec["params"])
                evaluator.create()
                collected[name] = evaluator.eval(
                    {k: v.copy() for k, v in parsed.items()}
                )
                self._logger.info(f"  {name} ({spec['method']}): ok")
            except Exception as exc:
                # One misconfigured evaluator should not discard the results of
                # the others -- evaluation runs after an expensive synthesis.
                errors[name] = f"{type(exc).__name__}: {exc}"
                self._logger.error(
                    f"  {name} ({spec['method']}) failed: {errors[name]}"
                )

        if not collected:
            raise RuntimeError(
                "Every nested evaluator failed:\n"
                + "\n".join(f"  {n}: {e}" for n, e in errors.items())
            )

        return self._merge(collected, errors)

    # ------------------------------------------------------------------ #

    def _merge(
        self, collected: dict[str, dict[str, Any]], errors: dict[str, str]
    ) -> dict[str, pd.DataFrame | None]:
        """Combine per-evaluator results, tagging each row with its evaluator.

        Keyed by name rather than blindly concatenated: MLUtility returns one row
        per metric while the others return a single row, so an untagged concat
        produces a frame you cannot attribute.
        """
        scores: dict[str, pd.DataFrame | None] = {}

        for granularity in self.AVAILABLE_SCORES_GRANULARITY:
            parts: list[pd.DataFrame] = []
            for name, result in collected.items():
                frame = result.get(granularity)
                if not isinstance(frame, pd.DataFrame) or frame.empty:
                    # Anonymeter returns 'details' as a dict of queries/links,
                    # which has no tabular form here.
                    if frame is not None and not isinstance(frame, pd.DataFrame):
                        self._logger.debug(
                            f"{name}: dropping non-tabular '{granularity}' "
                            f"({type(frame).__name__})"
                        )
                    continue
                tagged = frame.copy()
                tagged.insert(0, "evaluator", name)
                parts.append(tagged)

            if granularity == "global" and errors:
                parts.append(
                    pd.DataFrame(
                        [{"evaluator": n, "error": e} for n, e in errors.items()]
                    ).set_index(pd.Index(["failed"] * len(errors)))
                )

            scores[granularity] = pd.concat(parts) if parts else None

        return scores

    def _report_coverage(self, parsed: dict[str, pd.DataFrame]) -> None:
        """Log per-column fill rates, so a parser that matched nothing is visible."""
        for name, frame in parsed.items():
            rows = len(frame)
            counts = {str(c): int(frame[c].notna().sum()) for c in frame.columns}
            self._logger.info(
                f"  {name}: {rows} rows -> "
                + ", ".join(f"{c}={n}/{rows}" for c, n in counts.items())
            )
            empty = [c for c, n in counts.items() if rows and n == 0]
            if empty:
                self._logger.warning(
                    f"  {name}: columns empty in every row: {empty}. The parser "
                    "matched nothing -- scores built on them are meaningless."
                )

    def _dump(self, parsed: dict[str, pd.DataFrame]) -> None:
        """Write parsed frames for inspection. Never fails the run."""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            for name, frame in parsed.items():
                path = os.path.join(self.output_dir, f"parsed_{name}.csv")
                frame.to_csv(path, index=False, encoding="utf-8-sig")
                self._logger.info(f"Wrote {path}")
        except Exception as exc:
            self._logger.warning(f"Could not write parsed output: {exc}")


class TextEvaluator(_BaseTextEvaluator):
    """Parse-then-evaluate for evaluators needing only ori + syn.

    Use with SDMetrics and mpuccs. No Splitter required.
    """

    REQUIRED_INPUT_KEYS: list[str] = ["ori", "syn"]


class TextEvaluatorWithControl(_BaseTextEvaluator):
    """Parse-then-evaluate for evaluators that also need a control set.

    `control` is original data held out from synthesis, produced by a Splitter.
    Anonymeter reports the residual
    ``(attack_rate - control_rate) / (1 - control_rate)`` -- without control it
    cannot separate a real leak from the model having learnt the population
    distribution. MLUtility trains on ori and on syn, then tests both on control.
    """

    REQUIRED_INPUT_KEYS: list[str] = ["ori", "syn", "control"]
