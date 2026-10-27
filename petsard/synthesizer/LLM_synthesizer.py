"""
LLM-based synthetic data generator — PETsARD custom synthesizer adapter.

Loaded dynamically by PETsARD's CustomSynthesizer via YAML:

    Synthesizer:
      llm-synth:
        method: custom_method
        module_path: petsard/synthesizer/LLM_synthesizer.py
        class_name: LLM_synthesizer
        api_type: lmstudio
        api_key: "http://localhost:1234/v1"
        model: openai/gpt-oss-20b
        ...

All supporting code lives in LLM_synth/ (same directory as this file).
"""

from __future__ import annotations

import csv
import importlib.util
import os
import sys
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Make LLM_synth (next to this file) importable as a top-level package without
# adding petsard/synthesizer/ to sys.path, where its sdv.py and other modules
# would shadow installed packages of the same name (e.g. sdv).
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # petsard/synthesizer/
if "LLM_synth" not in sys.modules:
    _llm_synth_dir = os.path.join(_THIS_DIR, "LLM_synth")
    _spec = importlib.util.spec_from_file_location(
        "LLM_synth", os.path.join(_llm_synth_dir, "__init__.py"),
        submodule_search_locations=[_llm_synth_dir],
    )
    _pkg = importlib.util.module_from_spec(_spec)
    sys.modules["LLM_synth"] = _pkg
    _spec.loader.exec_module(_pkg)

# Environment variable read when a stage's api_key is empty, so keys stay out of YAML files.
API_KEY_ENV = "PETSARD_LLM_API_KEY"


def _create_api(api_type: str, api_key: str | None, model: str | None, api_url: str | None = None) -> Any:
    """Instantiate one of the supported LLM API wrappers."""
    import LLM_synth.api as _api

    _SDK_INSTALL = {
        "openai":    "pip install openai",
        "anthropic": "pip install anthropic",
        "gemini":    "pip install google-genai",
        "lmstudio":  "pip install openai",
    }

    if not api_type:
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


class LLM_synthesizer:
    """
    PETsARD-compatible custom synthesizer backed by the full DPSDA pipeline.

    Pipeline (mirrors main.py main()):
      1. fit()    — stores training rows as few-shot pool.
      2. sample() — runs the complete DPSDA pipeline:
                    generate_init → (variation → dp_nn_histogram) × epochs
                    and returns a pd.DataFrame of synthetic records.

    Full reference: doc_site/content/docs/petsard-yaml/synthesizer-yaml/llm-synthesizer.md.
    Keys marked * are required.

    PETsARD fixed keys (handled by CustomSynthesizer adapter, not this class):
      method          str   Must be "custom_method"
      module_path     str   Path to this file, relative to working directory
      class_name      str   Must be "LLM_synthesizer"
      sample_num_rows int   Target final output size returned to PETsARD (default 16).
                            Internal schedule is auto-inflated to produce at least this
                            many rows; any remainder is trimmed.

    Data
      main_column_name  str    Synthesize only this column; other columns are copied from
                               the source row of each sample (default None = whole row)

    API — init generation (*)
      api_type *       str   "openai" | "anthropic" | "gemini" | "lmstudio"; no default
                             (a missing api_type raises ValueError)
      api_key          str   API key; for lmstudio without api_url: server URL
                             (e.g. "http://localhost:1234/v1"). Empty or omitted = read
                             from the PETSARD_LLM_API_KEY environment variable (every
                             stage whose api_key is empty uses it)
      api_url          str   lmstudio server URL; None = api_key is the URL (default None).
                             Plain http to a host other than this machine warns.
      model             str   Provider default

    API — variation stage (defaults to init values)
      variation_api_type    str
      variation_api_key     str
      variation_api_url     str
      variation_model       str

    API — post-processing / QA stage (defaults to init values)
      post_processing_api_type    str
      post_processing_api_key     str
      post_processing_api_url     str
      post_processing_model       str

    DPSDA pipeline — epoch control
      epochs                    int    Variation+selection epochs after init (default 1).
                                     Total schedule length = epochs + 1.
      num_samples_schedule      int|list  Per-epoch sample count; auto-computed from
                                     sample_num_rows when omitted (default auto)
      variation_degree_schedule float|list  Fraction of tokens masked per epoch (default 0.1)

    DPSDA pipeline — combine/divide
      init_combine_divide_L   int    (default 1)
      combine_divide_L        int    (default 1)
      donnot_keep_last_iter   bool   (default False)

    DPSDA pipeline — DP selection
      noise_multiplier        float  (default 8.0); active when compute_fid is True
      lookahead_degree        int    Variations per sample for lookahead (default 2; 0 = off)
      lookahead_self          bool   (default False)
      select_syn_mode         str    "rank" | "prob" (default "rank")
      nn_mode                 str    "L2" | "IP" | "cos_sim" (default "L2")
      num_nearest_neighbor    int    (default 1)
      count_threshold         float  (default 0.0)

    DPSDA pipeline — output saving
      save_syn_mode           str    "selected" | "one_var" | "all" (default "selected")
      output_dir              str    Per-epoch artefacts root (default "./llm_synth_output")
      continue                bool   Resume from output_dir/{t}/checkpoint.csv; empty (failed)
                                     generations are redone (default False)

    Feature extraction / FID (requires sentence_transformers)
      compute_fid             bool   (default True)
      feature_extractor       str    (default "all-MiniLM-L6-v2")
      feature_extractor_batch_size int (default 1024)

    LLM generation behaviour
      fewshot_n_examples  int    (default 10)
      temperature         float  (default 1.0)
      top_p               float  (default 1.0)
      max_token_limit     int    (default 100000)
      sleep_time          int    Seconds between API calls (default 1)
      max_retries         int    (default 3)
      retry_wait_time     float  Seconds to wait before each retry, fixed (default 15)
      exit_on_api_error   bool   Stop the process when an API call still fails (exception
                                 or empty response) after max_retries attempts (default False)
      num_procs           int    Parallel workers (default 8)
      do_sample           bool   (default True)
      additional_info     str    Extra text in <AdditionalInfo> tags (default "")
      variation_type      str    e.g. "healthcare_blank_fill" (default "")

    Generation internals (rarely needed)
      seed                  int    Each generation's few-shot examples and mask come from
                                   its own RNG keyed by (seed, stage, index, variant), so
                                   `continue` never replays earlier draws (default 42)
      no_cuda               bool   (default False)
      dynamic_len           bool   (default False)
      min_target_word       int    (default 5)
      word_var_scale        int    (default 10)
      max_token_word_scale  float  (default 1.2)
      num_private_samples   int    Max training rows (-1 = all, default -1)

    Logging
      verbose               int    0=silent, 1=standard, 2=full prompts (default 0)
      log_folder            str    LLM_result_logger directory (default "./LOG_FILES")
      log_prompts           bool   Write prompt/response text to the log_folder log and
                                   output_dir/{t}/prompt_generation.jsonl (default True).
                                   False = the log keeps only each text's length, on file
                                   and console, and prompt_generation.jsonl is not written.
    """

    def __init__(self, config: dict, metadata=None) -> None:
        self.config = config
        self.metadata = metadata
        self._main_column_name: str | None = config.get("main_column_name")
        self._header: str | None = None
        self._data: pd.DataFrame | None = None
        self._training_samples: list[tuple[int, str]] = []
        self._log_prompts: bool = bool(config.get("log_prompts", True))
        self._generator = self._build_generator()

    # ------------------------------------------------------------------
    # PETsARD interface
    # ------------------------------------------------------------------

    def fit(self, data: pd.DataFrame) -> None:
        """Store training data as the few-shot pool and private dataset."""
        self._data = data
        if self._main_column_name is None:
            self._header = "\t".join(data.columns.astype(str))
            self._training_samples = [
                (i, ",".join(
                    f'"{str(v).replace(chr(34), chr(34) * 2)}"'
                    for v in row
                ))
                for i, row in enumerate(data.to_numpy())
            ]
        else:
            self._header = self._main_column_name
            self._training_samples = [
                (i, str(v)) for i, v in enumerate(data[self._main_column_name])
            ]

    def sample(self) -> pd.DataFrame:
        """Run the full DPSDA pipeline and return a pd.DataFrame."""
        if self._header is None:
            raise RuntimeError("Call fit() before sample().")

        import math

        cfg = self.config

        # ── DPSDA hyper-params (read first because schedule sizing needs L) ─
        epochs                  = int(cfg.get("epochs", 1))
        sample_num_rows         = int(cfg.get("sample_num_rows", 16))
        noise_multiplier        = cfg.get("noise_multiplier", 8.0)
        lookahead_degree        = cfg.get("lookahead_degree", 2)
        lookahead_self          = cfg.get("lookahead_self", False)
        init_combine_divide_L   = int(cfg.get("init_combine_divide_L", 1))
        combine_divide_L        = int(cfg.get("combine_divide_L", 1))
        donnot_keep_last_iter   = cfg.get("donnot_keep_last_iter", False)
        select_syn_mode         = cfg.get("select_syn_mode", "rank")
        save_syn_mode           = cfg.get("save_syn_mode", "selected")
        nn_mode                 = cfg.get("nn_mode", "L2")
        num_nearest_neighbor    = cfg.get("num_nearest_neighbor", 1)
        count_threshold         = cfg.get("count_threshold", 0.0)
        compute_fid_flag        = cfg.get("compute_fid", True)
        feature_extractor       = cfg.get("feature_extractor", "all-MiniLM-L6-v2")
        feature_extractor_batch_size = cfg.get("feature_extractor_batch_size", 1024)
        num_private_samples     = cfg.get("num_private_samples", -1)
        output_dir              = cfg.get("output_dir", "./llm_synth_output")
        resume                  = bool(cfg.get("continue", False))

        # ── schedule sizing ────────────────────────────────────────────────
        # Work backwards from sample_num_rows so the pipeline produces
        # exactly that many rows without any post-hoc trimming.
        #
        # Epoch t≥1 produces: int(epoch0_size / combine_divide_L) * combine_divide_L
        #   → epoch0_size must be the smallest multiple of combine_divide_L
        #     that is >= sample_num_rows.
        #
        # Epoch 0 produces: int(schedule[0] / init_combine_divide_L) * init_combine_divide_L
        #   → schedule[0] must be the smallest multiple of init_combine_divide_L
        #     that is >= required_epoch0_size.
        #
        # If epochs == 0, no variation loop runs, so required_epoch0_size == sample_num_rows.
        if epochs >= 1:
            required_epoch0_size = math.ceil(sample_num_rows / combine_divide_L) * combine_divide_L
        else:
            required_epoch0_size = sample_num_rows
        required_schedule0 = math.ceil(required_epoch0_size / init_combine_divide_L) * init_combine_divide_L

        user_schedule = cfg.get("num_samples_schedule", None)
        if user_schedule is None:
            num_samples_schedule = [required_schedule0] * (epochs + 1)
        else:
            if not isinstance(user_schedule, list):
                num_samples_schedule = [user_schedule] * (epochs + 1)
            elif len(user_schedule) == 1:
                num_samples_schedule = user_schedule * (epochs + 1)
            else:
                num_samples_schedule = list(user_schedule)
            # Inflate schedule[0] upward to the required multiple if too small.
            if num_samples_schedule[0] < required_schedule0:
                num_samples_schedule[0] = required_schedule0

        variation_degree_schedule = cfg.get("variation_degree_schedule", 0.1)
        if not isinstance(variation_degree_schedule, list):
            variation_degree_schedule = [variation_degree_schedule] * (epochs + 1)
        elif len(variation_degree_schedule) == 1:
            variation_degree_schedule = variation_degree_schedule * (epochs + 1)

        if len(num_samples_schedule) != len(variation_degree_schedule):
            raise ValueError(
                "num_samples_schedule and variation_degree_schedule must be the same length"
            )

        g = self._generator

        # ── private dataset ────────────────────────────────────────────────
        dataset_header  = self._header
        dataset_data    = self._training_samples
        parse_csv       = self._main_column_name is None
        if num_private_samples > 0:
            import random as _rnd
            dataset_data = _rnd.sample(dataset_data, min(num_private_samples, len(dataset_data)))

        # ── optional feature extraction (needs sentence_transformers) ──────
        if compute_fid_flag:
            from LLM_synth.synthetic_data.feature_extractor import extract_features
            all_private_features, extract_feature_model = extract_features(
                data=[text for _, text in dataset_data],
                batch_size=feature_extractor_batch_size,
                model_name=feature_extractor,
            )

        from LLM_synth.synthetic_data.dpsda.dp_counter import dp_nn_histogram
        from LLM_synth.synthetic_data.dpsda.logging import (
            Checkpoint,
            compute_fid,
            log_prompt_generation,
            log_samples,
        )

        # ── epoch 0: initial generation ────────────────────────────────────
        ckpt = Checkpoint(os.path.join(output_dir, "0"), resume)
        syn_samples_list = ckpt.samples("output")
        # reuse the saved output only if it covers the requested size (e.g. after raising sample_num_rows)
        resume = syn_samples_list is not None and len(syn_samples_list) >= num_samples_schedule[0]
        if not resume:
            g.logger.simple_log("Generating initial samples")
            num_seed_samples = int(num_samples_schedule[0] / init_combine_divide_L)

            seed_syn_samples, all_prefix_prompts = g.generate_init(
                num_samples=num_seed_samples,
                headers=dataset_header,
                training_samples=dataset_data,
                checkpoint=ckpt,
            )

            _step_dir = os.path.join(output_dir, "0")
            os.makedirs(_step_dir, exist_ok=True)
            if self._log_prompts:
                log_prompt_generation(
                    fname=os.path.join(_step_dir, "prompt_generation.jsonl"),
                    prompts=all_prefix_prompts,
                    generations=[seed_syn_samples],
                )
            log_samples(samples=seed_syn_samples, headers=dataset_header, folder=_step_dir,
                        parse_csv=parse_csv)

            if compute_fid_flag:
                syn_feats, _ = extract_features(
                    data=[text for _, text in seed_syn_samples],
                    batch_size=feature_extractor_batch_size,
                    model_name=feature_extractor,
                    model=extract_feature_model,
                )
                compute_fid(syn_feats, all_private_features, feature_extractor,
                            folder=output_dir, step=0)

            # ── combine initial samples (init_combine_divide_L > 1) ───────
            if init_combine_divide_L > 1:
                syn_samples_list = []
                stacked, _, _, _ = g.generate_variation(
                    sequences=seed_syn_samples,
                    num_variations_per_sequence=init_combine_divide_L - 1,
                    variation_degree=variation_degree_schedule[0],
                    headers=dataset_header,
                    training_samples=dataset_data,
                    checkpoint=ckpt,
                    stage="init_combine",
                )
                syn_samples_list.extend(seed_syn_samples)
                for x in stacked:
                    syn_samples_list.extend(x)
                log_samples(samples=syn_samples_list, headers=dataset_header,
                            folder=os.path.join(output_dir, "-1"), parse_csv=parse_csv)
            else:
                syn_samples_list = seed_syn_samples
            ckpt.save("output", syn_samples_list)

        g.logger.simple_log(f"Initial samples size: {len(syn_samples_list)}")

        # ── epochs 1..N: variation + DP selection ─────────────────────────
        for t in range(1, len(num_samples_schedule)):
            ckpt = Checkpoint(os.path.join(output_dir, str(t)), resume)
            recorded = ckpt.samples("output")
            resume = recorded is not None
            if resume:
                syn_samples_list = recorded
                continue
            g.logger.simple_log(f"Epoch t={t}")

            if lookahead_degree == 0:
                packed_samples = [[s] for s in syn_samples_list]
                all_prefix_prompts, all_masked_prompts = [], []
            else:
                g.logger.simple_log("Running text variation")
                packed_samples, all_gen_words, all_prefix_prompts, all_masked_prompts = \
                    g.generate_variation(
                        sequences=syn_samples_list,
                        num_variations_per_sequence=lookahead_degree,
                        variation_degree=variation_degree_schedule[t],
                        headers=dataset_header,
                        training_samples=dataset_data,
                        checkpoint=ckpt,
                        stage="lookahead",
                    )
                if lookahead_self:
                    packed_samples = [row + [s] for row, s in zip(packed_samples, syn_samples_list, strict=False)]
                _step_dir = os.path.join(output_dir, str(t))
                os.makedirs(_step_dir, exist_ok=True)
                if self._log_prompts:
                    log_prompt_generation(
                        fname=os.path.join(_step_dir, "prompt_generation.jsonl"),
                        prompts=all_masked_prompts,
                        generations=packed_samples,
                    )

            # feature extraction for all lookahead variations
            g.logger.simple_log("Running feature extraction")
            if compute_fid_flag:
                packed_features = []
                for column in zip(*packed_samples, strict=False):
                    packed_features.append(
                        extract_features(
                            data=[text for _, text in column],
                            batch_size=feature_extractor_batch_size,
                            model_name=feature_extractor,
                            model=extract_feature_model,
                        )[0]
                    )
                packed_features = np.mean(packed_features, axis=0)
            else:
                # without sentence_transformers, fall back to random uniform features
                packed_features = np.random.rand(len(syn_samples_list), 128)

            # DP nearest-neighbour histogram
            g.logger.simple_log("Computing histogram")
            new_syn_samples = []
            all_selected_samples = []
            selected_size = int(len(syn_samples_list) / combine_divide_L)

            if selected_size == 0:
                new_variants_samples = syn_samples_list * combine_divide_L
                new_syn_samples = new_variants_samples
                selected_syn_samples = syn_samples_list
            else:
                if compute_fid_flag:
                    sub_count, sub_clean_count = dp_nn_histogram(
                        public_features=packed_features,
                        private_features=all_private_features,
                        noise_multiplier=noise_multiplier,
                        num_nearest_neighbor=num_nearest_neighbor,
                        mode=nn_mode,
                        threshold=count_threshold,
                    )
                else:
                    # uniform counts when features are unavailable
                    sub_count = np.ones(len(syn_samples_list))

                assert np.sum(sub_count) > 0

                candidate_indices = np.arange(len(syn_samples_list), dtype=int)
                if select_syn_mode == "prob":
                    sampling_prob = sub_count / np.sum(sub_count)
                    sub_new_indices = np.random.choice(
                        candidate_indices, size=selected_size, p=sampling_prob
                    )
                elif select_syn_mode == "rank":
                    sort_index = [i for i, _ in sorted(
                        enumerate(sub_count), key=lambda x: -x[1]
                    )]
                    sub_new_indices = sort_index[:selected_size]
                else:
                    raise ValueError(f"Unsupported select_syn_mode: {select_syn_mode}")

                selected_syn_samples = ckpt.samples("select") or ckpt.save(
                    "select", [syn_samples_list[i] for i in sub_new_indices])
                new_variants_samples = []

                if combine_divide_L == 1:
                    _num_var = 1
                elif combine_divide_L > 1:
                    if donnot_keep_last_iter:
                        _num_var = combine_divide_L
                    else:
                        _num_var = combine_divide_L - 1
                        new_variants_samples.extend(selected_syn_samples)
                else:
                    raise ValueError("combine_divide_L must be >= 1")

                stacked, _, _, _ = g.generate_variation(
                    sequences=selected_syn_samples,
                    num_variations_per_sequence=_num_var,
                    variation_degree=variation_degree_schedule[t],
                    headers=dataset_header,
                    training_samples=dataset_data,
                    checkpoint=ckpt,
                    stage="combine",
                )
                for x in stacked:
                    new_variants_samples.extend(x)
                new_syn_samples.extend(new_variants_samples)

            # select samples to save this epoch
            if save_syn_mode == "selected":
                all_selected_samples.extend(selected_syn_samples)
            elif save_syn_mode == "one_var":
                all_selected_samples.extend(x[0] for x in stacked)
            else:  # "all"
                all_selected_samples.extend(new_variants_samples)

            syn_samples_list = new_syn_samples
            _step_dir = os.path.join(output_dir, str(t))
            os.makedirs(_step_dir, exist_ok=True)
            log_samples(samples=all_selected_samples, headers=dataset_header,
                        folder=_step_dir, parse_csv=parse_csv)
            log_samples(samples=syn_samples_list, headers=dataset_header,
                        folder=os.path.join(output_dir, f"{t}_all"), parse_csv=parse_csv)

            if compute_fid_flag:
                syn_feats, _ = extract_features(
                    data=[text for _, text in all_selected_samples],
                    batch_size=feature_extractor_batch_size,
                    model_name=feature_extractor,
                    model=extract_feature_model,
                )
                compute_fid(syn_feats, all_private_features, feature_extractor,
                            folder=output_dir, step=t)
            ckpt.save("output", syn_samples_list)

        df = self._to_dataframe(syn_samples_list)
        if len(df) > sample_num_rows:
            df = df.head(sample_num_rows).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_generator(self):
        from LLM_synth.synthetic_data.main import SyntheticDataGenerator

        cfg = self.config

        if not cfg.get("api_type"):
            raise ValueError(
                "LLM_synthesizer: 'api_type' is required. Choose from: 'openai', 'anthropic', "
                "'gemini', 'lmstudio' ('lmstudio' for a local server)."
            )

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

        generator = SyntheticDataGenerator(
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
            additional_info=cfg.get("additional_info", ""),
            variation_type=cfg.get("variation_type", ""),
            exit_on_api_error=cfg.get("exit_on_api_error", False),
            retry_wait_time=cfg.get("retry_wait_time", 15),
            verbose=cfg.get("verbose", 0),
            log_folder=cfg.get("log_folder", None),
        )
        # log_prompts: false keeps prompt/response text out of the LOG_FILES log (lengths only)
        generator.logger.log_prompts = self._log_prompts
        return generator

    def _to_dataframe(self, samples: list[tuple[int, str]]) -> pd.DataFrame:
        """Convert (unique_id, text) samples to a DataFrame."""
        samples = [(i, str(text).strip()) for i, text in samples if str(text).strip()]

        if self._main_column_name is not None:
            df = self._data.iloc[[i for i, _ in samples]].reset_index(drop=True)
            df[self._main_column_name] = [text for _, text in samples]
            return df

        columns = self._data.columns
        n_cols = len(columns)
        rows: list[list[str]] = []
        for _, text in samples:
            try:
                row = next(csv.reader([text]))
            except csv.Error:
                row = [text]
            rows.append((row + [""] * n_cols)[:n_cols])

        return pd.DataFrame(rows, columns=columns)
