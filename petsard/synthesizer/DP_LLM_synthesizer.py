"""
LLM-based synthetic data generator with differential privacy — PETsARD custom synthesizer.

Loaded dynamically by PETsARD's CustomSynthesizer via YAML:

    Synthesizer:
      llm-dp:
        method: custom_method
        module_path: petsard/synthesizer/DP_LLM_synthesizer.py
        class_name: DP_LLM_synthesizer
        epsilon: 4
        delta: 1.0e-5
        api_type: lmstudio
        api_url: "http://127.0.0.1:1234/v1"
        model: openai/gpt-oss-20b
        ...

Real rows reach the output only through two Gaussian mechanisms: the DP data profile
(step 0) and the nearest-neighbour vote of every epoch. The LLM never sees a real row:
the first round is generated zero-shot from the profile, later rounds take their few-shot
examples from synthetic rows already selected by the DP vote. The output is
(epsilon, delta)-differentially private with respect to adding or removing one row.
Usage and parameters: doc_site page "Differentially Private Synthesizers" (dp-synthesizers.md).

All supporting code lives in DP_LLM_synth/ (same directory as this file).
"""

from __future__ import annotations

import csv
import importlib.util
import io
import json
import logging
import math
import os
import re
import sys
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Make DP_LLM_synth (next to this file) importable as a top-level package without
# adding petsard/synthesizer/ to sys.path, where its sdv.py and other modules
# would shadow installed packages of the same name (e.g. sdv). Same as LLM_synthesizer.py.
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # petsard/synthesizer/
if "DP_LLM_synth" not in sys.modules:
    _dp_llm_synth_dir = os.path.join(_THIS_DIR, "DP_LLM_synth")
    _spec = importlib.util.spec_from_file_location(
        "DP_LLM_synth", os.path.join(_dp_llm_synth_dir, "__init__.py"),
        submodule_search_locations=[_dp_llm_synth_dir],
    )
    _pkg = importlib.util.module_from_spec(_spec)
    sys.modules["DP_LLM_synth"] = _pkg
    _spec.loader.exec_module(_pkg)

# Environment variable read when a stage's api_key is empty, so keys stay out of YAML files.
API_KEY_ENV = "PETSARD_LLM_API_KEY"

_LOGGER = logging.getLogger("PETsARD.DP_LLM_synthesizer")
_BLANK_SYMBOL = "<MASK>"  # same as DP_LLM_synth.synthetic_data.agents.variation_agent.BLANK_SYMBOL
_FENCE_RE = re.compile(r"^```[A-Za-z]*[ \t]*\n(.*?)\n?```$", re.S)

# keys of the non-DP synthesizer that would break the privacy guarantee
_REJECTED_KEYS = {
    "main_column_name": "it copies the other columns from the source real row; "
                        "the DP version always synthesizes whole rows",
    "noise_multiplier": "the vote noise is derived from epsilon, delta and epochs",
}


def _create_api(api_type: str, api_key: str | None, model: str | None, api_url: str | None = None) -> Any:
    """Instantiate one of the supported LLM API wrappers."""
    import DP_LLM_synth.api as _api

    _SDK_INSTALL = {
        "openai":    "pip install openai",
        "anthropic": "pip install anthropic",
        "gemini":    "pip install google-genai",
        "lmstudio":  "pip install openai",
    }

    if not api_type:
        # no default: a missing api_type must not send prompts to a cloud service by accident
        raise ValueError(
            "api_type is required. Choose from: 'openai', 'anthropic', 'gemini', 'lmstudio' "
            "('lmstudio' for a local server)."
        )
    t = str(api_type).lower()
    if t not in _SDK_INSTALL:
        raise ValueError(
            f"Unsupported api_type '{api_type}'. "
            "Choose from: 'openai', 'anthropic', 'gemini', 'lmstudio'."
        )

    _cls_map = {
        "openai":    (_api.OpenAIAPI,    "gpt-4o"),
        "anthropic": (_api.AnthropicAPI, "claude-sonnet-4-20250514"),
        "gemini":    (_api.GeminiAPI,    "gemini-2.5-flash"),
        "lmstudio":  (_api.LMStudioAPI,  "default"),
    }
    cls, default_model = _cls_map[t]

    if cls is None:
        raise ImportError(
            f"The '{t}' provider SDK is not installed. "
            f"Install it with:  {_SDK_INSTALL[t]}"
        )

    env_key = None
    if not api_key:
        env_key = os.environ.get(API_KEY_ENV) or None
        api_key = env_key

    if t == "lmstudio":
        if env_key and not api_url:
            # the key came from the environment, so it is not the server URL: use the default server
            api_url = "http://localhost:1234/v1"
        return cls(api_key=api_key or "http://localhost:1234/v1", model=model or default_model,
                   api_url=api_url)
    if api_url:
        raise ValueError(f"api_url is only supported by api_type 'lmstudio', not '{t}'.")
    return cls(api_key=api_key, model=model or default_model)


# ---------------------------------------------------------------------------
# CSV row handling (the LLM reads and writes every record as one CSV row)
# ---------------------------------------------------------------------------

def serialize_row(values) -> str:
    """One record as the LLM sees it: every field quoted, inner quotes doubled."""
    return ",".join('"' + str(v).replace('"', '""') + '"' for v in values)


def parse_row(text: str, columns: list[str], allow_mask: bool = False) -> tuple[list[str] | None, str | None]:
    """Parse one LLM answer (or, with allow_mask, a masked record) into exactly len(columns)
    fields; (fields, None) or (None, error)."""
    t = (text or "").strip()
    if not t:
        return None, "empty output"
    m = _FENCE_RE.match(t)
    if m:
        t = m.group(1).strip()
    if not allow_mask and _BLANK_SYMBOL in t:
        return None, f"output still contains {_BLANK_SYMBOL}"
    try:
        rows = [r for r in csv.reader(io.StringIO(t)) if r]
    except csv.Error as exc:
        return None, f"not a valid CSV row ({exc})"
    if len(rows) > 1 and [c.strip() for c in rows[0]] == columns:
        rows = rows[1:]  # a repeated header line
    if len(rows) != 1:
        return None, f"expected one CSV row, got {len(rows)}"
    if len(rows[0]) != len(columns):
        return None, f"expected {len(columns)} fields, got {len(rows[0])}"
    return rows[0], None


def _is_number(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


class DP_LLM_synthesizer:
    """
    PETsARD-compatible custom synthesizer: Aug-PE style LLM synthesis with an
    (epsilon, delta) differential-privacy guarantee.

    Pipeline:
      0. DP data profile — line labels and symbol patterns, short column values, frequent
         words and phrases, and the length x line-count distribution, released by four
         Gaussian mechanisms; no format is assumed (DP_LLM_synth/synthetic_data/dpsda/profile.py).
      1. Initial generation — zero-shot from the profile and a per-row target; no real row
         is in any prompt.
      2. Per epoch — mask-and-fill variations whose few-shot examples are synthetic rows,
         a DP nearest-neighbour vote (Gaussian noise from OS entropy), keep the top
         1/combine_divide_L, and vary the kept rows back to the pool size.
      3. Expansion — if sample_num_rows exceeds the pool, vary the final rows further
         (post-processing: no privacy cost).
    The noise of every step is derived from epsilon/delta/epochs (dpsda/accountant.py);
    privacy_report.json in output_dir records what was spent.

    Full reference: llm_synth_config.md. Keys marked * are required.

    PETsARD fixed keys (handled by CustomSynthesizer adapter, not this class):
      method          str   Must be "custom_method"
      module_path     str   Path to this file, relative to working directory
      class_name      str   Must be "DP_LLM_synthesizer"
      sample_num_rows int   Final output size (default 16)

    Differential privacy (*)
      epsilon *               float  Total privacy budget; .inf = no noise (reference run, NOT private)
      delta *                 float  Total delta, e.g. 1.0e-5 (should be well below 1 / rows)
      profile_budget_fraction float  Share of the budget (in mu^2) for the profile (default 0.2)
      profile_delta_s         float  Part of delta for the profile's release threshold (default 1e-6)
      profile_max_items       int    Structure items (line labels, symbol runs) per row (default 32)
      profile_max_vocab       int    Vocabulary items (words, phrases) per row; more are randomly
                                     subsampled (default 64)
      profile_joint_fields    bool   Release the combination of all short values of a row and
                                     draw targets as whole combinations (default True)
      profile_value_wording   bool   Fifth query: the wording typical of each short column value
                                     (default True; with False the profile has four queries)
      label_vote              bool   Write the initial texts WITHOUT any short column value and
                                     assign the values by a DP vote of the private rows
                                     (dpsda/label_vote.py) (default False)
      label_vote_budget_fraction float  Share of the budget (in mu^2) for that vote (default 0.3)
      label_vote_smooth_k     int    Post-processing: average each text's noisy counts with its
                                     k nearest synthetic texts (default 10)
      value_wording_budget_fraction float  Give the value-wording counts a budget of their own
                                     (share of mu^2; default None = shared with the profile)
      words_per_row           int    Draw this many vocabulary items per row from the value-wording
                                     counts given the row's drawn values, and ask the model to use
                                     them (default 0 = off)
      protect_drawn_words     bool   With words_per_row: the words drawn for a record are never masked
                                     in its variations (default True)
      hide_values             bool   The model writes only the long columns; the short values drawn
                                     for the row are filled in afterwards and never shown to it
                                     (default False)
      Not accepted: main_column_name (copies real columns), noise_multiplier (derived).
      The unit of privacy is one row: give one row per person for person-level privacy.

    API — init generation (*)
      api_type *       str   "openai" | "anthropic" | "gemini" | "lmstudio"; no default
                             (a missing api_type raises ValueError)
      api_key          str   API key; for lmstudio without api_url: server URL
                             (e.g. "http://localhost:1234/v1"). Empty or omitted = read
                             from the PETSARD_LLM_API_KEY environment variable (every
                             stage whose api_key is empty uses it)
      api_url          str   lmstudio server URL; None = api_key is the URL (default None)
      model             str   Provider default

    API — variation stage (defaults to init values)
      variation_api_type, variation_api_key, variation_api_url, variation_model

    DPSDA pipeline
      epochs                    int    DP vote rounds after the initial generation (default 1)
      num_samples_schedule      int|list  Pool size per round (first value used; default
                                     sample_num_rows); rounded up to a multiple of L
      variation_degree_schedule float|list  Fraction of tokens masked per epoch (default 0.1)
      init_combine_divide_L     int    (default 1)
      combine_divide_L          int    L: the vote keeps 1/L of the pool; >= 2 (default 2)
      donnot_keep_last_iter     bool   Vary each kept row L times (True) or keep it plus L-1
                                     variations (False) (default False)
      lookahead_degree          int    Variations per sample for the vote (default 2; 0 = off)
      lookahead_self            bool   (default False)
      select_syn_mode           str    "rank" | "prob" (default "rank")
      nn_mode                   str    "L2" | "IP" | "cos_sim" (default "L2")
      num_nearest_neighbor      int    Votes per real row (default 1)
      count_threshold           float  (default 0.0)
      save_syn_mode             str    "selected" | "one_var" | "all" (default "selected")
      output_dir                str    Per-epoch artefacts root (default "./llm_synth_output")
      continue                  bool   Resume from output_dir; the profile and the noisy votes
                                     already drawn are reused, never redrawn (default False)
      stop_after_epoch          int    Stop after this epoch (0 = after the initial generation)
                                     and write output_dir/epoch{t}_result.csv; the budget stays
                                     planned for all `epochs`, so a later run without this key
                                     (continue: true) carries on from there (default None)

    Feature extraction (requires sentence_transformers)
      feature_extractor             str  (default "all-MiniLM-L6-v2")
      feature_extractor_batch_size  int  (default 1024)

    LLM generation behaviour
      fewshot_n_examples  int    Synthetic examples per variation prompt (default 10)
      temperature         float  (default 1.0)
      reasoning_effort    str    Reasoning models only: "low" | "medium" | "high" (default: server's)
      top_p               float  (default 1.0)
      max_token_limit     int    (default 100000)
      sleep_time          int    Seconds between API calls (default 1)
      max_retries         int    (default 3); an answer that is not a valid row is retried
                                 at once and left empty after the last attempt
      retry_wait_time     float  Seconds to wait before retrying a failed API call (default 15)
      exit_on_api_error   bool   Stop the process when an API call still fails after
                                 max_retries attempts (default False)
      num_procs           int    Parallel workers (default 8)
      additional_info     str    Optional extra text in <AdditionalInfo> tags (default "")
      require_released_profile bool  Stop unless output_dir already holds profile.json, so that a
                                 model change cannot release the data again (default false)

    Generation internals (rarely needed)
      seed                  int    Controls masks, example draws and per-row targets only;
                                   the DP noise never depends on it (default 42)
      num_private_samples   int    Use at most this many rows (-1 = all, default -1)

    Logging
      verbose               int    0=silent, 1=standard, 2=full prompts (default 0)
      log_folder            str    LLM_result_logger directory (default "./LOG_FILES")
    """

    def __init__(self, config: dict, metadata=None) -> None:
        self.config = config
        self.metadata = metadata
        self._check_dp_config(config)
        self._header: str | None = None
        self._columns: list[str] = []
        self._numeric_idx: list[int] = []
        self._data: pd.DataFrame | None = None
        self._training_samples: list[tuple[int, str]] = []
        self._generator = None
        self._init_columns: list[str] | None = None  # label vote: columns the LLM writes first

    @staticmethod
    def _check_dp_config(cfg: dict) -> None:
        for key, why in _REJECTED_KEYS.items():
            if cfg.get(key) is not None:
                raise ValueError(f"'{key}' is not supported by the DP LLM synthesizer: {why}.")
        for key in ("epsilon", "delta"):
            if cfg.get(key) is None:
                raise ValueError(f"'{key}' is required by the DP LLM synthesizer.")
        if int(cfg.get("epochs", 1)) >= 1 and int(cfg.get("combine_divide_L", 2)) < 2:
            raise ValueError("combine_divide_L must be >= 2: with 1 the vote only reorders "
                             "the pool and the real data never shapes the output.")
        if "compute_fid" in cfg:
            _LOGGER.warning("compute_fid is ignored: FID would use the real data outside the "
                            "DP mechanisms and has been removed.")
        # validates epsilon / delta / fractions early
        from DP_LLM_synth.synthetic_data.dpsda.accountant import plan_budget
        plan_budget(
            epsilon=float(cfg["epsilon"]), delta=float(cfg["delta"]),
            epochs=int(cfg.get("epochs", 1)),
            profile_fraction=float(cfg.get("profile_budget_fraction", 0.2)),
            delta_s=float(cfg.get("profile_delta_s", 1e-6)),
            label_fraction=DP_LLM_synthesizer._label_fraction(cfg),
            wording_fraction=float(cfg.get("value_wording_budget_fraction") or 0.0),
        )
        if cfg.get("label_vote") and cfg.get("hide_values"):
            raise ValueError("label_vote and hide_values cannot be combined")
        if (cfg.get("label_vote") or cfg.get("hide_values")) and int(cfg.get("init_combine_divide_L", 1)) != 1:
            raise ValueError("label_vote needs init_combine_divide_L = 1")
        if int(cfg.get("profile_max_items", 32)) < 1 or int(cfg.get("profile_max_vocab", 64)) < 1:
            raise ValueError("profile_max_items and profile_max_vocab must be >= 1")

    @staticmethod
    def _label_fraction(cfg: dict) -> float:
        return float(cfg.get("label_vote_budget_fraction", 0.3)) if cfg.get("label_vote") else 0.0

    # ------------------------------------------------------------------
    # PETsARD interface
    # ------------------------------------------------------------------

    def fit(self, data: pd.DataFrame) -> None:
        """Keep the private rows. They are used only by the DP profile and the DP vote."""
        self._data = data.reset_index(drop=True)
        self._columns = [str(c) for c in self._data.columns]
        self._header = "\t".join(self._columns)
        self._numeric_idx = [
            i for i, c in enumerate(self._data.columns)
            if pd.api.types.is_numeric_dtype(self._data[c])
        ]
        self._training_samples = [
            (i, serialize_row(row)) for i, row in enumerate(self._data.to_numpy())
        ]
        self._generator = self._build_generator()

    def sample(self) -> pd.DataFrame:
        """Run the DP pipeline and return a pd.DataFrame."""
        if self._generator is None:
            raise RuntimeError("Call fit() before sample().")

        from DP_LLM_synth.synthetic_data.dpsda.accountant import plan_budget
        from DP_LLM_synth.synthetic_data.dpsda.dp_counter import dp_nn_histogram
        from DP_LLM_synth.synthetic_data.dpsda.logging import (
            Checkpoint,
            log_prompt_generation,
            log_samples,
        )
        from DP_LLM_synth.synthetic_data.dpsda.profile import build_profile

        cfg = self.config
        epochs                  = int(cfg.get("epochs", 1))
        sample_num_rows         = int(cfg.get("sample_num_rows", 16))
        lookahead_degree        = int(cfg.get("lookahead_degree", 2))
        lookahead_self          = bool(cfg.get("lookahead_self", False))
        init_combine_divide_L   = int(cfg.get("init_combine_divide_L", 1))
        combine_divide_L        = int(cfg.get("combine_divide_L", 2))
        donnot_keep_last_iter   = bool(cfg.get("donnot_keep_last_iter", False))
        select_syn_mode         = cfg.get("select_syn_mode", "rank")
        save_syn_mode           = cfg.get("save_syn_mode", "selected")
        nn_mode                 = cfg.get("nn_mode", "L2")
        num_nearest_neighbor    = int(cfg.get("num_nearest_neighbor", 1))
        count_threshold         = float(cfg.get("count_threshold", 0.0))
        feature_extractor       = cfg.get("feature_extractor", "all-MiniLM-L6-v2")
        feature_extractor_batch_size = int(cfg.get("feature_extractor_batch_size", 1024))
        num_private_samples     = int(cfg.get("num_private_samples", -1))
        output_dir              = cfg.get("output_dir", "./llm_synth_output")
        resume                  = bool(cfg.get("continue", False))
        stop_after_epoch        = cfg.get("stop_after_epoch")
        if stop_after_epoch is not None:
            stop_after_epoch = int(stop_after_epoch)
            if not 0 <= stop_after_epoch <= epochs:
                raise ValueError("stop_after_epoch must be between 0 and epochs")
        profile_caps            = {"max_items": int(cfg.get("profile_max_items", 32)),
                                   "max_vocab": int(cfg.get("profile_max_vocab", 64)),
                                   "joint_fields": bool(cfg.get("profile_joint_fields", True)),
                                   "conditional_vocab": bool(cfg.get("profile_value_wording", True))}
        if select_syn_mode not in ("rank", "prob"):
            raise ValueError(f"Unsupported select_syn_mode: {select_syn_mode}")

        from DP_LLM_synth.synthetic_data.dpsda.profile import n_queries
        # a profile released earlier keeps the number of queries it was released with
        wording_fraction = float(cfg.get("value_wording_budget_fraction") or 0.0)
        released = self._released_profile(output_dir, resume)
        if released is not None:
            profile_queries = int(released.get("n_queries", 4))
        else:
            profile_queries = n_queries(profile_caps["conditional_vocab"], wording_fraction > 0)
        plan = plan_budget(
            epsilon=float(cfg["epsilon"]), delta=float(cfg["delta"]), epochs=epochs,
            profile_fraction=float(cfg.get("profile_budget_fraction", 0.2)),
            delta_s=float(cfg.get("profile_delta_s", 1e-6)),
            n_profile_queries=profile_queries,
            label_fraction=self._label_fraction(cfg),
            wording_fraction=wording_fraction,
        )
        label_vote = bool(cfg.get("label_vote", False))
        hide_values = bool(cfg.get("hide_values", False))
        words_per_row = int(cfg.get("words_per_row", 0))

        # ── pool size: candidates per vote round (output size comes from step 3) ──
        user_schedule = cfg.get("num_samples_schedule")
        if user_schedule is None:
            pool_size = sample_num_rows
        else:
            pool_size = int(user_schedule[0] if isinstance(user_schedule, list) else user_schedule)
        if epochs >= 1:
            pool_size = math.ceil(pool_size / combine_divide_L) * combine_divide_L
        pool_size = math.ceil(pool_size / init_combine_divide_L) * init_combine_divide_L

        variation_degree_schedule = cfg.get("variation_degree_schedule", 0.1)
        if not isinstance(variation_degree_schedule, list):
            variation_degree_schedule = [variation_degree_schedule] * (epochs + 1)
        elif len(variation_degree_schedule) == 1:
            variation_degree_schedule = variation_degree_schedule * (epochs + 1)
        if len(variation_degree_schedule) != epochs + 1:
            raise ValueError("variation_degree_schedule must have epochs + 1 entries")

        g = self._generator
        header = self._header

        # ── private rows (never passed to the generator) ────────────────────
        private = self._training_samples
        if 0 < num_private_samples < len(private):
            import random as _rnd
            private = _rnd.sample(private, num_private_samples)

        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, "privacy_report.json")
        # ── step 0: DP data profile (computed once, reused on resume) ────────
        if cfg.get("require_released_profile") and self._released_profile(output_dir, resume) is None:
            raise RuntimeError(
                f"require_released_profile is set, but no released profile.json is in {output_dir} "
                "(continue: true is also needed). Nothing was computed from the private rows.")
        profile = self._load_or_build_profile(
            output_dir, resume, private, plan, profile_caps, build_profile)
        report_args = dict(plan=plan, n_rows=len(private), profile=profile,
                           num_nearest_neighbor=num_nearest_neighbor, label_votes=0)
        self._write_report(report_path, vote_rounds=0, complete=False, **report_args)

        init_profile = {**profile, "words_per_row": words_per_row} if words_per_row else profile
        if label_vote or hide_values:
            # the LLM writes only the columns that have no released short values; those
            # (labels, codes, ratings) come from the label vote after the generation
            short_cols = [c for c in self._columns if c in (profile.get("fields") or {})]
            long_cols = [c for c in self._columns if c not in short_cols]
            if not short_cols or not long_cols:
                raise ValueError("label_vote needs released short columns and at least one other column")
            self._init_columns = long_cols
            # the initial prompt asks only for the columns the LLM writes, whether or not
            # words are drawn and protected
            g.llm_generator._init_additional_info = "\n".join(
                x for x in (cfg.get("additional_info", ""), self._csv_note(long_cols)) if x)
            if label_vote:
                init_profile = {**init_profile, "columns": long_cols, "fields": {}, "combos": [],
                                "value_wording": {}}
            else:  # values drawn per row but never shown; they fill the row afterwards
                init_profile = {**init_profile, "columns": long_cols, "hide_fields": True}
        if words_per_row and bool(cfg.get("protect_drawn_words", True)):
            # the drawn words carry the link between text and values: variations keep them
            import functools
            from DP_LLM_synth.synthetic_data.dpsda.profile import sample_target as _st
            _view = init_profile

            @functools.lru_cache(maxsize=None)
            def _protected(seq_id):
                words = _st(_view, g._key_rng("init", int(seq_id), 0)).get("words") or []
                return frozenset(t.lower() for w in words for t in str(w).split())
            g.protected_words = _protected

        def stop_here(t, samples, rounds):
            """End the run after epoch t: the budget stays planned for all `epochs`, and a
            later run without stop_after_epoch continues from here with continue: true."""
            self._write_report(report_path, vote_rounds=rounds, complete=False,
                               status=f"stopped after epoch {t}", **report_args)
            df = self._to_dataframe([s for s in samples if s[1].strip()])
            df.to_csv(os.path.join(output_dir, f"epoch{t}_result.csv"), index=False)
            # the synthetic record each row descends from: variations of one record are
            # near-duplicates and must share a cross-validation fold in evaluations
            ids = [k for k, text in samples
                   if text.strip() and parse_row(text, self._columns)[1] is None]
            if len(ids) == len(df):
                pd.DataFrame({"row": range(len(ids)), "group": ids}).to_csv(
                    os.path.join(output_dir, f"epoch{t}_groups.csv"), index=False)
            g.logger.simple_log(f"Stopped after epoch {t} (stop_after_epoch)")
            return df.head(sample_num_rows).reset_index(drop=True)

        if epochs >= 1 and (stop_after_epoch is None or stop_after_epoch >= 1):
            from DP_LLM_synth.synthetic_data.feature_extractor import extract_features
            private_features, extract_feature_model = extract_features(
                data=[self._feature_text(text) for _, text in private],
                batch_size=feature_extractor_batch_size,
                model_name=feature_extractor,
            )

        # ── step 1: zero-shot initial generation ────────────────────────────
        ckpt = Checkpoint(os.path.join(output_dir, "0"), resume)
        syn_samples_list = ckpt.samples("output")
        resume = syn_samples_list is not None and len(syn_samples_list) >= pool_size
        if resume:
            syn_samples_list = syn_samples_list[:pool_size]
        else:
            g.logger.simple_log("Generating initial samples from the DP profile")
            seed_syn_samples, all_prefix_prompts = g.generate_init(
                num_samples=pool_size // init_combine_divide_L,
                headers=header,
                profile=init_profile,
                checkpoint=ckpt,
            )
            _step_dir = os.path.join(output_dir, "0")
            log_prompt_generation(
                fname=os.path.join(_step_dir, "prompt_generation.jsonl"),
                prompts=all_prefix_prompts,
                generations=[seed_syn_samples],
            )
            if label_vote:
                seed_syn_samples = self._label_vote(
                    seed_syn_samples, profile, plan, _step_dir, cfg, private)
            elif hide_values:
                seed_syn_samples = self._fill_drawn_values(seed_syn_samples, init_profile)
            log_samples(samples=seed_syn_samples, headers=header, folder=_step_dir)

            if init_combine_divide_L > 1:
                syn_samples_list = list(seed_syn_samples)
                stacked, _, _, _ = g.generate_variation(
                    sequences=seed_syn_samples,
                    num_variations_per_sequence=init_combine_divide_L - 1,
                    variation_degree=variation_degree_schedule[0],
                    headers=header,
                    example_pool=seed_syn_samples,
                    checkpoint=ckpt,
                    stage="init_combine",
                )
                for x in stacked:
                    syn_samples_list.extend(x)
                log_samples(samples=syn_samples_list, headers=header,
                            folder=os.path.join(output_dir, "-1"))
            else:
                syn_samples_list = seed_syn_samples
            ckpt.save("output", syn_samples_list)
        if label_vote or hide_values:
            report_args["label_votes"] = int(label_vote)
            self._init_columns = None  # everything from here on is a whole row
            self._write_report(report_path, vote_rounds=0, complete=False, **report_args)

        g.logger.simple_log(f"Initial samples size: {len(syn_samples_list)}")
        if stop_after_epoch == 0:
            return stop_here(0, syn_samples_list, 0)

        # ── step 2: epochs 1..T — variation + DP vote ────────────────────────
        vote_rounds = 0
        for t in range(1, epochs + 1):
            _step_dir = os.path.join(output_dir, str(t))
            opened_with_resume = resume
            ckpt = Checkpoint(_step_dir, resume)
            counts_path = os.path.join(_step_dir, "noisy_counts.json")
            if not opened_with_resume and os.path.exists(counts_path):
                os.remove(counts_path)  # belongs to a checkpoint that was just discarded
            recorded = ckpt.samples("output")
            resume = recorded is not None
            if resume:
                syn_samples_list = recorded
                vote_rounds += 1  # this round's vote was spent by an earlier run
                if stop_after_epoch == t:
                    return stop_here(t, syn_samples_list, vote_rounds)
                continue
            g.logger.simple_log(f"Epoch t={t}")

            # few-shot examples: the previous DP output (t = 1: the zero-shot samples)
            example_pool = syn_samples_list
            if lookahead_degree == 0:
                packed_samples = [[s] for s in syn_samples_list]
            else:
                g.logger.simple_log("Running text variation")
                packed_samples, _, _, all_masked_prompts = g.generate_variation(
                    sequences=syn_samples_list,
                    num_variations_per_sequence=lookahead_degree,
                    variation_degree=variation_degree_schedule[t],
                    headers=header,
                    example_pool=example_pool,
                    checkpoint=ckpt,
                    stage="lookahead",
                )
                if lookahead_self:
                    packed_samples = [row + [s] for row, s in zip(packed_samples, syn_samples_list, strict=False)]
                log_prompt_generation(
                    fname=os.path.join(_step_dir, "prompt_generation.jsonl"),
                    prompts=all_masked_prompts,
                    generations=packed_samples,
                )

            g.logger.simple_log("Running feature extraction")
            packed_features = np.mean([
                extract_features(
                    data=[self._feature_text(text) for _, text in column],
                    batch_size=feature_extractor_batch_size,
                    model_name=feature_extractor,
                    model=extract_feature_model,
                )[0]
                for column in zip(*packed_samples, strict=False)
            ], axis=0)

            # DP nearest-neighbour vote: only the noisy counts are kept
            g.logger.simple_log("Computing histogram")
            if opened_with_resume and os.path.exists(counts_path):
                with open(counts_path, encoding="utf-8") as f:
                    sub_count = np.asarray(json.load(f)["counts"], dtype=float)
                g.logger.simple_log("Reusing the noisy vote counts saved by an earlier run")
            else:
                sub_count, _clean = dp_nn_histogram(
                    public_features=packed_features,
                    private_features=private_features,
                    noise_multiplier=plan["sigma_vote"],
                    num_nearest_neighbor=num_nearest_neighbor,
                    mode=nn_mode,
                    threshold=count_threshold,
                )
                del _clean
                with open(counts_path, "w", encoding="utf-8") as f:
                    json.dump({"counts": [float(c) for c in sub_count],
                               "sigma": plan["sigma_vote"]}, f)
            vote_rounds += 1

            # selection (post-processing of the noisy counts); failed rows are never kept
            selected_size = len(syn_samples_list) // combine_divide_L
            usable = np.array([bool(text.strip()) for _, text in syn_samples_list])
            rank_count = np.where(usable, sub_count, -1.0)
            if select_syn_mode == "prob":
                p = np.clip(rank_count, 0.0, None)
                if p.sum() == 0:
                    p = usable.astype(float)
                sub_new_indices = np.random.choice(len(p), size=selected_size, p=p / p.sum())
            else:
                sub_new_indices = sorted(range(len(rank_count)), key=lambda i: -rank_count[i])[:selected_size]
            selected_syn_samples = ckpt.samples("select") or ckpt.save(
                "select", [syn_samples_list[i] for i in sub_new_indices])

            new_variants_samples = []
            if donnot_keep_last_iter:
                _num_var = combine_divide_L
            else:
                _num_var = combine_divide_L - 1
                new_variants_samples.extend(selected_syn_samples)
            stacked, _, _, _ = g.generate_variation(
                sequences=selected_syn_samples,
                num_variations_per_sequence=_num_var,
                variation_degree=variation_degree_schedule[t],
                headers=header,
                example_pool=selected_syn_samples,
                checkpoint=ckpt,
                stage="combine",
            )
            for x in stacked:
                new_variants_samples.extend(x)

            if save_syn_mode == "selected":
                all_selected_samples = list(selected_syn_samples)
            elif save_syn_mode == "one_var":
                all_selected_samples = [x[0] for x in stacked]
            else:  # "all"
                all_selected_samples = list(new_variants_samples)

            syn_samples_list = new_variants_samples
            log_samples(samples=all_selected_samples, headers=header, folder=_step_dir)
            log_samples(samples=syn_samples_list, headers=header,
                        folder=os.path.join(output_dir, f"{t}_all"))
            ckpt.save("output", syn_samples_list)
            self._write_report(report_path, vote_rounds=vote_rounds, complete=False, **report_args)
            if stop_after_epoch == t:
                return stop_here(t, syn_samples_list, vote_rounds)

        # ── step 3: expand to sample_num_rows (post-processing) ─────────────
        final = [s for s in syn_samples_list if s[1].strip()]
        need = sample_num_rows - len(final)
        if need > 0 and final:
            _step_dir = os.path.join(output_dir, "expand")
            ckpt = Checkpoint(_step_dir, resume)
            reps = math.ceil(need / len(final))
            g.logger.simple_log(f"Expanding {len(final)} rows by {need} variations")
            stacked, _, _, _ = g.generate_variation(
                sequences=final,
                num_variations_per_sequence=reps,
                variation_degree=variation_degree_schedule[-1],
                headers=header,
                example_pool=final,
                checkpoint=ckpt,
                stage="expand",
            )
            extra = [stacked[i][j] for j in range(reps) for i in range(len(final))]
            final = final + [s for s in extra if s[1].strip()][:need]
            log_samples(samples=final, headers=header, folder=_step_dir)

        self._write_report(report_path, vote_rounds=vote_rounds, complete=True, **report_args)

        df = self._to_dataframe(final)
        if len(df) < sample_num_rows:
            g.logger.simple_log(f"Only {len(df)} valid rows for sample_num_rows={sample_num_rows}")
        return df.head(sample_num_rows).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _released_profile(output_dir, resume):
        path = os.path.join(output_dir, "profile.json")
        if resume and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        return None

    def _load_or_build_profile(self, output_dir, resume, private, plan, caps, build_profile):
        """The DP profile is released once per output_dir and reused afterwards."""
        path = os.path.join(output_dir, "profile.json")
        released = self._released_profile(output_dir, resume)
        if released is not None:
            return released
        ckpt0 = os.path.join(output_dir, "0", "checkpoint.csv")
        if resume and os.path.exists(ckpt0):
            with open(ckpt0, encoding="utf-8") as f:
                has_rows = sum(1 for _ in f) > 1
            if has_rows:
                raise RuntimeError(
                    f"{path} is missing but {ckpt0} holds generations: a new profile would "
                    "release the data a second time. Restore profile.json, or start over "
                    "with continue: false.")
        idx = [i for i, _ in private]
        rows = [dict(zip(self._columns, r)) for r in self._data.iloc[idx].to_numpy()]
        profile = build_profile(
            rows, [text for _, text in private], self._columns,
            sigma=plan["sigma_profile"], delta_s=plan["delta_s"],
            max_items=caps["max_items"], max_vocab=caps["max_vocab"],
            joint_fields=caps["joint_fields"], conditional_vocab=caps["conditional_vocab"],
            wording_sigma=plan.get("sigma_wording"),
        )
        if profile["n_queries"] != plan["n_profile_queries"]:
            raise RuntimeError("profile queries do not match the privacy plan")
        del rows
        with open(path, "w", encoding="utf-8") as f:
            json.dump(profile, f, ensure_ascii=False, indent=1)
        return profile

    def _label_vote(self, samples, profile, plan, step_dir, cfg, private):
        """Assign the short column values of the text-only initial samples by the DP label
        vote (dpsda/label_vote.py). The noisy counts are saved and reused on resume."""
        from DP_LLM_synth.synthetic_data.dpsda.label_vote import assign_values, noisy_label_counts
        from DP_LLM_synth.synthetic_data.dpsda.profile import _is_missing
        from DP_LLM_synth.synthetic_data.feature_extractor import extract_features

        long_cols = self._init_columns
        short_cols = [c for c in self._columns if c not in long_cols]
        fields = profile["fields"]
        domain = {c: [v["value"] for v in fields[c]] for c in short_cols}
        shares = {c: [v["share"] for v in fields[c]] for c in short_cols}
        parsed = [(k, parse_row(text, long_cols)[0] if text.strip() else None) for k, text in samples]
        live = [i for i, (_, f) in enumerate(parsed) if f is not None]
        if not live:
            raise RuntimeError("label vote: no usable initial sample")
        model_name = cfg.get("feature_extractor", "all-MiniLM-L6-v2")
        batch = int(cfg.get("feature_extractor_batch_size", 1024))
        syn_feat, model = extract_features(
            data=["\n".join(parsed[i][1]) for i in live], batch_size=batch, model_name=model_name)

        path = os.path.join(step_dir, "label_vote.json")
        ids = [parsed[i][0] for i in live]
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                saved = json.load(f)
            if saved.get("candidates") != ids:
                raise RuntimeError(f"{path} belongs to other candidates; refusing a second vote")
            noisy = {c: np.asarray(saved["counts"][c], dtype=float) for c in short_cols}
            g_log = "Reusing the noisy label-vote counts saved by an earlier run"
        else:
            rows = self._data.iloc[[i for i, _ in private]]
            real_text = ["\n".join(str(v) for v in r) for r in rows[long_cols].to_numpy()]
            real_vals = [{c: "" if _is_missing(v) else str(v) for c, v in zip(short_cols, r)}
                         for r in rows[short_cols].to_numpy()]
            real_feat, _ = extract_features(data=real_text, batch_size=batch,
                                            model_name=model_name, model=model)
            noisy = noisy_label_counts(syn_feat, real_feat, real_vals, domain, plan["sigma_label"])
            del real_text, real_vals, real_feat
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"candidates": ids, "sigma": plan["sigma_label"],
                           "counts": {c: m.tolist() for c, m in noisy.items()}}, f)
            g_log = "Label vote: noisy counts drawn and saved"
        self._generator.logger.simple_log(g_log)

        assigned = assign_values(noisy, domain, shares, syn_feat,
                                 int(cfg.get("label_vote_smooth_k", 10)))
        out = [(k, "") for k, _ in parsed]
        for pos, i in enumerate(live):
            values = dict(zip(long_cols, parsed[i][1]), **assigned[pos])
            out[i] = (parsed[i][0], serialize_row([values[c] for c in self._columns]))
        return out

    def _fill_drawn_values(self, samples, init_profile):
        """hide_values: the short values of each row are the ones drawn for its target
        (re-drawn from the same per-row seed, so they match the words the model was given)."""
        from DP_LLM_synth.synthetic_data.dpsda.profile import sample_target

        g = self._generator
        long_cols = self._init_columns
        fields = init_profile.get("fields") or {}
        fallback = {c: vs[0]["value"] for c, vs in fields.items() if vs}
        out = []
        for k, text in samples:
            parsed = parse_row(text, long_cols)[0] if text.strip() else None
            if parsed is None:
                out.append((k, ""))
                continue
            drawn = sample_target(init_profile, g._key_rng("init", k, 0))["fields"]
            values = dict(zip(long_cols, parsed))
            for c in self._columns:
                if c not in values:
                    values[c] = drawn.get(c, fallback.get(c, ""))
            out.append((k, serialize_row([values[c] for c in self._columns])))
        return out

    @staticmethod
    def _csv_note(columns) -> str:
        """Generic output instructions built from the schema only."""
        return ("Every record is ONE CSV row with these columns, in this order: "
                + ", ".join(columns)
                + ". Output exactly one CSV row: one field per column, separated by commas; "
                "enclose a field in double quotes if it contains a comma, a double quote or a line "
                "break, and write a double quote inside a field as two double quotes. "
                "No header line, no commentary.")

    @staticmethod
    def _write_report(path, plan, n_rows, profile, num_nearest_neighbor, vote_rounds, complete,
                      status=None, label_votes=0):
        from DP_LLM_synth.synthetic_data.dpsda.accountant import realized_epsilon

        mu, eps = realized_epsilon(plan["sigma_profile"], plan["n_profile_queries"],
                                   plan["sigma_vote"], vote_rounds, plan["delta_gauss"],
                                   sigma_label=plan.get("sigma_label"), label_votes=label_votes,
                                   sigma_wording=plan.get("sigma_wording"),
                                   wording_queries=int(plan.get("sigma_wording") is not None))

        def num(x):
            return "inf" if isinstance(x, float) and math.isinf(x) else x

        report = {
            "status": status or ("complete" if complete else "running"),
            "differentially_private": not math.isinf(plan["epsilon"]),
            "privacy_unit": "one row (neighbouring datasets differ by adding or removing one row)",
            "rows_used": n_rows,
            "epsilon_spent": num(eps),
            "delta": plan["delta"],
            "mu_gdp": num(mu),
            "epsilon_budget": num(plan["epsilon"]),
            "delta_split": {"gaussian": plan["delta_gauss"], "profile_threshold": plan["delta_s"]},
            "profile": {
                "queries": plan["n_profile_queries"],
                "budget_fraction": plan["profile_fraction"],
                "sigma": plan["sigma_profile"],
                "release_thresholds": profile.get("release_thresholds"),
                "items_per_row_caps": profile.get("item_caps"),
            },
            "value_wording": None if plan.get("sigma_wording") is None else {
                "budget_fraction": plan["wording_fraction"],
                "sigma": plan["sigma_wording"],
            },
            "label_vote": None if plan.get("sigma_label") is None else {
                "budget_fraction": plan["label_fraction"],
                "sigma": plan["sigma_label"],
                "done": bool(label_votes),
            },
            "votes": {
                "rounds_planned": plan["epochs"],
                "rounds_done": vote_rounds,
                "sigma": plan["sigma_vote"],
                "num_nearest_neighbor": num_nearest_neighbor,
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)

    def _feature_text(self, text: str) -> str:
        """Text given to the sentence-embedding model for the vote, for real and synthetic
        rows alike. The model reads only the first ~256 tokens, so numeric columns (known
        from the schema) go first as "column=value" and long text after; otherwise values
        placed after a long report would never take part in the vote."""
        fields, error = parse_row(text, self._columns)
        if error:
            return text
        numeric = set(self._numeric_idx)
        head = "; ".join(f"{self._columns[i]}={fields[i]}" for i in self._numeric_idx)
        rest = "\n".join(f for i, f in enumerate(fields) if i not in numeric)
        return f"{head}\n{rest}" if head else rest

    def _check_output(self, text: str, masked: str | None = None) -> tuple[str, str | None]:
        """Accept an LLM answer only if it is one CSV row with the right fields; return it in
        the canonical serialization so that examples and votes see a uniform format.

        For a variation (`masked` given), the short fields were never masked, so by design
        they keep their values: they are restored from the masked record, whatever the
        model wrote there."""
        # label vote: the initial generation writes only self._init_columns
        columns = self._init_columns if (masked is None and self._init_columns) else self._columns
        fields, error = parse_row(text, columns)
        if error:
            return text, error
        if masked is not None:
            source = parse_row(masked, self._columns, allow_mask=True)[0]
            if source is not None:
                fields = [s if len(s) <= self._short_max and _BLANK_SYMBOL not in s else f
                          for f, s in zip(fields, source)]
        numeric = {self._columns[i] for i in self._numeric_idx}
        for c, value in zip(columns, fields):
            value = value.strip()
            if c in numeric and value and not _is_number(value):
                return text, f"column {c} needs a number, got {value[:30]!r}"
        return serialize_row(fields), None

    def _build_generator(self):
        from DP_LLM_synth.synthetic_data.dpsda.profile import SHORT_VALUE_MAX
        from DP_LLM_synth.synthetic_data.main import SyntheticDataGenerator

        cfg = self.config
        self._short_max = SHORT_VALUE_MAX

        # api_type has no default in any stage (see _create_api)
        init_api = _create_api(
            cfg.get("api_type"),
            cfg.get("api_key"),
            cfg.get("model"),
            cfg.get("api_url"),
        )
        variation_api = _create_api(
            cfg.get("variation_api_type", cfg.get("api_type")),
            cfg.get("variation_api_key", cfg.get("api_key")),
            cfg.get("variation_model", cfg.get("model")),
            cfg.get("variation_api_url", cfg.get("api_url")),
        )
        post_processing_api = _create_api(
            cfg.get("post_processing_api_type", cfg.get("api_type")),
            cfg.get("post_processing_api_key", cfg.get("api_key")),
            cfg.get("post_processing_model", cfg.get("model")),
            cfg.get("post_processing_api_url", cfg.get("api_url")),
        )

        # generic note built from the schema only; the user's additional_info is optional
        csv_note = self._csv_note(self._columns)
        additional_info = "\n".join(x for x in (cfg.get("additional_info", ""), csv_note) if x)

        return SyntheticDataGenerator(
            init_generation_api=init_api,
            variation_api=variation_api,
            post_processing_api=post_processing_api,
            output_dir=cfg.get("output_dir", "./llm_synth_output"),
            seed=cfg.get("seed", 42),
            num_procs=cfg.get("num_procs", 8),
            temperature=cfg.get("temperature", 1.0),
            top_p=cfg.get("top_p", 1.0),
            do_sample=cfg.get("do_sample", True),
            no_cuda=cfg.get("no_cuda", False),
            dynamic_len=cfg.get("dynamic_len", False),
            min_target_word=cfg.get("min_target_word", 5),
            word_var_scale=cfg.get("word_var_scale", 10),
            max_token_word_scale=cfg.get("max_token_word_scale", 1.2),
            sleep_time=cfg.get("sleep_time", 1),
            max_token_limit=cfg.get("max_token_limit", 100000),
            max_retries=cfg.get("max_retries", 3),
            fewshot_n_examples=cfg.get("fewshot_n_examples", 10),
            additional_info=additional_info,
            variation_type=cfg.get("variation_type", ""),
            exit_on_api_error=cfg.get("exit_on_api_error", False),
            retry_wait_time=cfg.get("retry_wait_time", 15),
            verbose=cfg.get("verbose", 0),
            log_folder=cfg.get("log_folder", None),
            output_check=self._check_output,
            row_codec=(lambda text: parse_row(text, self._columns)[0], serialize_row,
                       SHORT_VALUE_MAX),
            reasoning_effort=cfg.get("reasoning_effort"),
        )

    def _to_dataframe(self, samples: list[tuple[int, str]]) -> pd.DataFrame:
        """Parse synthetic rows; every column comes from the LLM, none from a real row."""
        rows = []
        for _, text in samples:
            fields, error = parse_row(text, self._columns)
            if error is None:
                rows.append(fields)
        df = pd.DataFrame(rows, columns=self._data.columns)
        for c in self._data.columns:
            if pd.api.types.is_numeric_dtype(self._data[c]):
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df
