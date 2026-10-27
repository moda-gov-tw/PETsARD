# Part of the LLM synthesizer adapted from AI-secure/aug-pe
# (https://github.com/AI-secure/aug-pe, Apache License 2.0; see
# petsard/synthesizer/LLM_synth/LICENSE-APACHE-2.0.txt and NOTICE).
# This module implements the Private Evolution generation and variation
# steps; it was rewritten and extended by NTUST in 2026.

import random
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ..api import APITemplate
from ..utils import LLM_result_logger
from .agents.variation_agent import BLANK_SYMBOL
from .llm_generation import LLM_Generation


def set_seed(seed, n_gpu=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class SyntheticDataGenerator:
    def __init__(self,
        init_generation_api: APITemplate,
        variation_api: APITemplate,
        post_processing_api: APITemplate,
        output_dir: str,
        seed: int,
        num_procs: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
        no_cuda: bool,
        dynamic_len: bool,
        min_target_word: int,
        word_var_scale: int,
        max_token_word_scale: float,
        sleep_time: int,
        max_token_limit: int,
        max_retries: int,
        fewshot_n_examples: int,
        additional_info: str = "",
        variation_type: str = "",
        exit_on_api_error: bool = False,
        retry_wait_time: float = 15.0,
        verbose: int = 0,
        log_folder: str = None,
    ):
        self.init_generation_api = init_generation_api
        self.variation_api = variation_api
        self.post_processing_api = post_processing_api
        self.verbose = verbose
        self.logger = LLM_result_logger.Logger(
            model_name="MainSyntheticData",
            verbose=verbose,
            folder=log_folder,
        )

        # ── dataset generation config ─────────────────────────────────────────
        self.num_procs = num_procs
        self.output_dir = output_dir
        self.additional_info = additional_info


        # ── generation hyper-parameters ───────────────────────────────────────
        self.temperature = temperature
        self.p = top_p
        self.do_sample = do_sample
        self.no_cuda = no_cuda
        self.dynamic_len = dynamic_len
        self.min_target_word = min_target_word
        self.word_var_scale = word_var_scale
        self.max_token_word_scale = max_token_word_scale
        self.max_token_limit = max_token_limit
        self.sleep_time = sleep_time
        self.max_retries = max_retries
        self.fewshot_n_examples = fewshot_n_examples
        self.variation_type = variation_type

        # ── random seeds & device ─────────────────────────────────────────────
        self.seed = seed
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not self.no_cuda else "cpu"
        )
        self.n_gpu = 0 if self.no_cuda else torch.cuda.device_count()
        set_seed(seed=seed, n_gpu=self.n_gpu)

        # ── blank-fill tokeniser (only for blank_fill variants) ───────────────
        #if "blank_fill" in self.variation_type:
        #    from transformers import BertTokenizer  # type: ignore
        #    self.mask_tokenizer = BertTokenizer.from_pretrained(
        #        "bert-base-cased", mask_token=BLANK_SYMBOL
        #    )
        #    self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

        self.llm_generator = LLM_Generation(
            init_generation_api=self.init_generation_api,
            variation_api=self.variation_api,
            post_processing_api=self.post_processing_api,
            logger=self.logger,
            sleep_time=self.sleep_time,
            max_retries=self.max_retries,
            additional_info=self.additional_info,
            exit_on_api_error=exit_on_api_error,
            seed=self.seed,
            retry_wait_time=retry_wait_time,
        )

        self.request_params = {
            "max_tokens": self.max_token_limit,
            "temperature": self.temperature,
            "top_p": self.p,
        }

    # ── public dataset generation interface ──────────────────────────────────
    def generate_init(
        self,
        num_samples: int,
        headers: str,
        training_samples: list[tuple[int, str]],
        checkpoint: dict = None,
    ):
        syn_samples = []
        all_prefix_prompts = []

        sequences, prefix_prompts = self._generate_init_sequence(
            num_samples, headers, training_samples, checkpoint,
        )
        #self.logger.info("%s  →  %d samples", prompt_text, len(sequences))

        syn_samples += sequences
        all_prefix_prompts += prefix_prompts

        return syn_samples, all_prefix_prompts

    def _generate_init_sequence(
        self, seq_num, headers, training_samples, checkpoint=None
    ):
        all_data = []
        all_prefix_prompts = []
        valid = [s for s in training_samples if str(s[1]).strip()]
        checkpoint = {} if checkpoint is None else checkpoint
        lock = threading.Lock()

        def _run_one(k):
            key = ("init", k, 0)
            examples_text = ""
            if key not in checkpoint:
                sampled = self._key_rng(*key).sample(valid, min(self.fewshot_n_examples, len(valid)))
                examples_text = "<Header>" + headers + "</Header>\n" + ("\n\n----------------------------------------\n\n".join(text for _, text in sampled[:-1])) + "\n\n---------------<QuerySet>-------------------------\n\n" + sampled[-1][1] + "\n\n---------------</QuerySet>-------------------------\n\n" + "\n\nIn your output, DO NOT output <Header></Header> tags."
                result = self.llm_generator.generate_init(self.request_params, examples_text)
                checkpoint[key] = (sampled[-1][0], result)
                with lock:
                    pbar.update()
            return checkpoint[key], examples_text

        done = sum(("init", k, 0) in checkpoint for k in range(seq_num))
        with tqdm(total=seq_num, initial=done) as pbar, \
                ThreadPoolExecutor(max_workers=self.num_procs) as executor:
            for result, examples_text in executor.map(_run_one, range(seq_num)):
                all_data.append(result)
                all_prefix_prompts.append(examples_text)

        if len(all_data) > seq_num:
            all_data = random.sample(all_data, seq_num)

        return all_data, all_prefix_prompts

    def generate_variation(
        self,
        sequences,
        num_variations_per_sequence: int,
        variation_degree: float,
        headers: str,
        training_samples: list[tuple[int, str]],
        checkpoint: dict = None,
        stage: str = "variation",
    ):
        if not 0 <= variation_degree <= 1:
            raise ValueError("variation_degree must be between 0 and 1.")

        variations, all_gen_words, all_prefix_prompts, all_masked_prompts = self._generate_variation_sequence(
            sequences=sequences,
            variation_degree=variation_degree,
            lookahead=num_variations_per_sequence,
            headers=headers,
            training_samples=training_samples,
            checkpoint=checkpoint,
            stage=stage,
        )
        return variations, all_gen_words, all_prefix_prompts, all_masked_prompts

    def _key_rng(self, stage: str, i: int, j: int) -> random.Random:
        """Private RNG for one checkpoint key (stage, i, j), derived from the seed.

        Every generation draws its few-shot examples and mask from its own key, never
        from the global RNG: set_seed() resets the global RNG on every (re)start, and
        finished keys are skipped on `continue`, so global draws would replay the first
        run's draws on the remaining keys and pick the same QuerySets again."""
        return random.Random(f"{self.seed}:{stage}:{i}:{j}")

    def _create_masked_seq(self, input_sequence: str, mlm_probability: float = 0.5,
                           rng: random.Random = None) -> str:
        """Word-level masking that handles both whitespace-separated (English) and
        CJK (Chinese/Japanese/Korean) text where words have no spaces.

        Tokens that consist entirely of structural/formatting characters are never
        masked — they are trivially restorable by pattern-matching the reference
        example and masking them wastes BLANK_SYMBOL budget."""
        import re
        try:
            # Character-class body for structural/formatting markers.
            # Shared by both the split pattern and _STRUCTURAL so the two
            # definitions stay in sync.
            # Tokens composed entirely of these characters must never be masked
            # (see variation_agent ClassifyBlanks step):
            #   markdown / restructuredtext: = : - * ` # | > _ + ~ ! [ ] { } \
            #   grouping / operators:        ( ) < > / @ ^ & % $ ? . , ; " '
            #   unicode bullets:             • ‣ ◦ ⁃ ∙ ※ · … — – ° § ¶
            #   Chinese / fullwidth punct:   ， 。 、 ； ： ！ ？ （ ） 「 」 『 』
            #                                【 】 《 》 〈 〉 ～ ＂ ＇ ⋯ ‥ ・
            _STRUCT_CC = (
                r'=:\-*`#|>_+~!\[\]{}\\/\(\)<>@^&%$?.,;\"\''  # ASCII structural
                r'\u2022\u2023\u25e6\u2043\u2219'               # unicode bullets  • ‣ ◦ ⁃ ∙
                r'\u203b\u00b7\u2026\u2014\u2013'               # ※ · … — –
                r'\u00b0\u00a7\u00b6'                           # ° § ¶
                r'\uff0c\u3002\u3001\uff1b\uff1a\uff01\uff1f'   # ， 。 、 ； ： ！ ？
                r'\uff08\uff09\u300c\u300d\u300e\u300f'         # （ ） 「 」 『 』
                r'\u3010\u3011\u300a\u300b\u3008\u3009'         # 【 】 《 》 〈 〉
                r'\uff5e\uff02\uff07\u22ef\u2025\u30fb'         # ～ ＂ ＇ ⋯ ‥ ・
            )
            _CJK_CC = (
                r'\u4e00-\u9fff'    # CJK Unified Ideographs
                r'\u3400-\u4dbf'    # CJK Extension A
                r'\uf900-\ufaff'    # CJK Compatibility Ideographs
                r'\u3040-\u309f'    # Hiragana
                r'\u30a0-\u30ff'    # Katakana
                r'\uac00-\ud7af'    # Korean Hangul
            )
            _STRUCTURAL = re.compile(rf'^[{_STRUCT_CC}]+$')

            # Split into four classes of spans, preserving order:
            #   1. Individual CJK characters   (each is its own maskable unit)
            #   2. Runs of structural chars     (never masked; emitted as own tokens
            #      so mixed spans like =35， are NOT lumped into one token)
            #   3. Runs of word characters      (non-CJK, non-structural, non-whitespace)
            #   4. Whitespace runs              (preserved as-is, never masked)
            pattern = (
                rf'[{_CJK_CC}]'                                 # single CJK char
                rf'|[{_STRUCT_CC}]+'                            # structural run
                rf'|[^\s{_CJK_CC}{_STRUCT_CC}]+'               # word run
                rf'|\s+'                                        # whitespace
            )
            parts = re.findall(pattern, input_sequence)
            if not parts:
                return input_sequence

            # Mask only non-whitespace, non-structural spans
            maskable = [
                i for i, p in enumerate(parts)
                if not p.isspace() and not _STRUCTURAL.match(p)
            ]
            if not maskable:
                return input_sequence

            rng = rng or random
            flags = [rng.random() < mlm_probability for _ in maskable]
            masked_set = {idx for idx, flag in zip(maskable, flags, strict=False) if flag}
            result = [BLANK_SYMBOL if i in masked_set else p for i, p in enumerate(parts)]
            return "".join(result)
        except Exception:
            return input_sequence

    def _generate_variation_mapping(self, i, sequence, lookahead, valid_pool, headers, checkpoint, stage, on_generated):
        seq_id, seq_text = sequence
        results = []
        all_prefix_prompts = []
        all_masked_prompts = []
        for j in range(lookahead):
            key = (stage, i, j)
            examples_text = masked_seq = ""
            if key not in checkpoint:
                rng = self._key_rng(*key)
                masked_seq = self._create_masked_seq(seq_text, self.mlm_probability, rng)
                examples = rng.sample(valid_pool, min(self.fewshot_n_examples, len(valid_pool)))
                examples_text = "<Header>"+headers+"</Header>\n"+"\n----------------------------------------\n".join(text for _, text in examples[:-1]) + "\n\n---------------<QuerySet>-------------------------\n\n" + examples[-1][1] + "\n\n---------------</QuerySet>-------------------------\n\n"
                # a failed (empty) sequence has nothing to vary
                result = self.llm_generator.generate_variation(self.request_params, examples_text, masked_seq) if seq_text.strip() else ""
                checkpoint[key] = (seq_id, result)
                on_generated()
            results.append(checkpoint[key])
            all_prefix_prompts.append(examples_text)
            all_masked_prompts.append(masked_seq)
        return results, all_prefix_prompts, all_masked_prompts

    def _generate_variation_sequence(
        self, sequences, variation_degree, lookahead, headers, training_samples,
        checkpoint=None, stage="variation"
    ):
        num_seq = len(sequences)
        all_data = []
        all_gen_words = []
        all_prefix_prompts = []
        all_masked_prompts = []
        valid_pool = [s for s in training_samples if str(s[1]).strip()]
        checkpoint = {} if checkpoint is None else checkpoint

        self.mlm_probability = variation_degree

        lock = threading.Lock()

        def _on_generated():
            with lock:
                pbar.update()

        def _run_one(i, sequence):
            return self._generate_variation_mapping(
                i, sequence, lookahead, valid_pool, headers, checkpoint, stage, _on_generated
            )

        done = sum((stage, i, j) in checkpoint for i in range(num_seq) for j in range(lookahead))
        with tqdm(total=num_seq * lookahead, initial=done) as pbar, \
                ThreadPoolExecutor(max_workers=self.num_procs) as executor:
            for r_ in executor.map(_run_one, range(num_seq), sequences):
                all_data.append(r_[0])
                all_gen_words += r_[0]
                all_prefix_prompts += r_[1]
                all_masked_prompts += r_[2]

        self.logger.simple_log(
            f"_text_variation output lens: {[len(d) for d in zip(*all_data, strict=False)]}"
        )
        return all_data, all_gen_words, all_prefix_prompts, all_masked_prompts

    def load_data(self, data_file, num_samples=-1):
        dataframe = pd.read_csv(data_file)
        if num_samples > 0:
            dataframe = dataframe.sample(num_samples)
        header = '\t'.join(dataframe.columns.astype(str))
        data = [
            (i, ','.join(f'"{str(v).replace(chr(34), chr(34) * 2)}"' for v in row))
            for i, row in enumerate(dataframe.to_numpy())
        ]
        return header, data
