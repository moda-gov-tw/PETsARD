# SPDX-License-Identifier: Apache-2.0
# Adapted from AI-secure/aug-pe (https://github.com/AI-secure/aug-pe),
# dpsda/logging.py, licensed under the Apache License 2.0
# (see petsard/synthesizer/LLM_synth/LICENSE-APACHE-2.0.txt and NOTICE).
# Modified by NTUST in 2026: UTF-8 log output, sample ids in the prompt log,
# CSV sample logging and the Checkpoint class for resumable runs.

import csv
import json
import logging
import os
import threading

import numpy as np

from .metrics import calculate_fid, knn_precision_recall_features


def setup_logging(log_file):
    import io
    import sys

    log_formatter = logging.Formatter(
        fmt=('%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  '
             '%(message)s'),
        datefmt='%m/%d/%Y %H:%M:%S %p')
    root_logger = logging.getLogger()
    # root_logger.setLevel(logging.DEBUG)
    root_logger.setLevel(logging.INFO)

    _utf8_stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True
    )
    console_handler = logging.StreamHandler(_utf8_stderr)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    pil_logger = logging.getLogger('PIL')
    pil_logger.setLevel(logging.INFO)


def log_embeddings(embeddings, additional_info, folder, fname=''):
    if not os.path.exists(folder):
        os.makedirs(folder)
    savefname = os.path.join(folder, fname+'.embeddings.npz')
    print("save embeddings into", savefname)
    np.savez(
        savefname,
        embeddings=embeddings,
        additional_info=additional_info)


def load_embeddings(path):
    data = np.load(path)
    embeddings = data['embeddings']
    additional_info = data['additional_info']

    return embeddings, additional_info


def log_num_words(fname="num_word_lookahead.csv", all_gen_words=None, all_target_words=None):
    if all_gen_words is None:
        all_gen_words = []
    if all_target_words is None:
        all_target_words = []
    if len(all_gen_words) == 0 or len(all_target_words) == 0:
        return
    with open(fname, 'w', newline='', encoding="utf-8") as wf:
        csv_writer = csv.writer(wf)
        csv_writer.writerow(["target", "gen", "diff"])
        diff_list = []
        diff_abs_list = []
        for i in range(len(all_target_words)):
            try:
                diff_list.append(all_gen_words[i] - all_target_words[i])
                diff_abs_list.append(
                    abs(all_gen_words[i] - all_target_words[i]))
                csv_writer.writerow(
                    [all_target_words[i], all_gen_words[i], all_gen_words[i] - all_target_words[i]])
            except Exception:
                continue
        csv_writer.writerow(["mean_abs", "var_abs", "mean", "var"])
        csv_writer.writerow([np.mean(diff_abs_list), np.std(
            diff_abs_list), np.mean(diff_list), np.std(diff_list)])


def log_prompt_generation(fname="prompt_generation.jsonl", prompts=None, generations=None):
    if prompts is None:
        prompts = []
    if generations is None:
        generations = []
    new_variants_samples = []
    for x in generations:
        new_variants_samples.extend(x)

    if len(prompts) == 0 or len(new_variants_samples) == 0:
        return
    with open(fname, "w") as file:
        for i in range(len(prompts)):
            try:
                sample_id, generation = new_variants_samples[i]
                json_str = json.dumps(
                    {"prompt": prompts[i], "id": sample_id, "generation": generation})
                file.write(json_str + "\n")
            except Exception:
                continue


def log_count(count, clean_count, path):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    title = ['type', 'count']
    with open(path, 'w', newline='', encoding="utf-8") as wf:
        csv_writer = csv.writer(wf)
        csv_writer.writerow(title)
        csv_writer.writerow(["count", count.tolist()])
        csv_writer.writerow(["clean_count", clean_count.tolist()])


def compute_fid(synthetic_features, all_private_features, feature_extractor, folder='', step=0):

    logging.info(
        f'Computing FID and F1 for syn shape {synthetic_features.shape}')
    fid = calculate_fid(synthetic_features, all_private_features)
    state = knn_precision_recall_features(
        ref_features=all_private_features, eval_features=synthetic_features)
    logging.info(f'fid={fid} F1={state}')
    log_fid(folder, fid, state["f1"],
            state["precision"], state["recall"], step)

def log_fid(folder, fid, f1, precision, recall, t, save_fname='fid.csv'):
    with open(os.path.join(folder, save_fname), 'a') as f:
        f.write(f'{t} {fid} {f1} {precision} {recall}\n')


def log_fid_list(folder, fids, t, save_fname='fid.csv'):
    write_list = [t]
    write_list.extend(fids)
    with open(os.path.join(folder, save_fname), 'a') as f:
        writer = csv.writer(f)
        writer.writerow(write_list)


def _parse_sample_row(sample, num_columns):
    text = str(sample).strip() if sample is not None else ""
    if not text:
        return None

    try:
        row = next(csv.reader([text]))
    except csv.Error:
        row = [text]

    if num_columns <= 0:
        return row

    if len(row) == num_columns:
        return row
    if len(row) == 1:
        return [row[0]] + [""] * (num_columns - 1)
    if len(row) < num_columns:
        return row + [""] * (num_columns - len(row))
    return row[:num_columns]


def log_samples(samples, headers, folder, filename="samples.csv", parse_csv=True):
    if not os.path.exists(folder):
        os.makedirs(folder)

    title = headers.split("\t") if isinstance(headers, str) else list(headers)
    num_columns = len(title)

    all_data = []
    for sample_id, sample in samples:
        row = _parse_sample_row(sample, num_columns) if parse_csv else [sample]
        if row and any(str(cell).strip() for cell in row):
            all_data.append([sample_id] + row)

    csv_path = os.path.join(folder, filename)

    def _write_rows(csv_writer):
        csv_writer.writerow(["unique_id"] + title)
        for obj in all_data:
            if obj[1]:
                csv_writer.writerow(obj)

    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as wf:
            _write_rows(csv.writer(wf, quoting=csv.QUOTE_MINIMAL))
    except Exception as e:
        logging.warning(
            "CSV writing failed with error: %s. Retrying with QUOTE_ALL mode.", e
        )
        with open(csv_path, "w", newline="", encoding="utf-8") as wf:
            _write_rows(
                csv.writer(wf, quoting=csv.QUOTE_ALL, escapechar="\\")
            )
    return all_data


class Checkpoint(dict):
    """Per-epoch CSV of every finished generation, appended as it happens and reloaded by `continue`.

    Key: (stage, index, variant). Value: (unique_id, text). An empty text is a failed generation:
    it is never written, and `continue` removes it from an existing file so it gets regenerated.
    """

    COLUMNS = ["stage", "index", "variant", "unique_id", "text"]

    def __init__(self, folder, resume=False):
        super().__init__()
        os.makedirs(folder, exist_ok=True)
        self.path = os.path.join(folder, "checkpoint.csv")
        self._lock = threading.Lock()
        if resume and os.path.exists(self.path) and os.path.getsize(self.path) > 0:
            csv.field_size_limit(2**31 - 1)
            with open(self.path, newline="", encoding="utf-8") as f:
                rows = [r for r in list(csv.reader(f))[1:] if len(r) == len(self.COLUMNS)]
            # a select/output list holding an empty text is dropped whole to keep positions aligned
            broken = {r[0] for r in rows if not r[4].strip()} & {"select", "output"}
            kept = [r for r in rows if r[4].strip() and r[0] not in broken]
            if len(kept) < len(rows):
                self._append([self.COLUMNS, *kept], mode="w", path=self.path + ".tmp")
                os.replace(self.path + ".tmp", self.path)
            for stage, index, variant, unique_id, text in kept:
                super().__setitem__((stage, int(index), int(variant)), (int(unique_id), text))
        else:
            self._append([self.COLUMNS], mode="w")

    def __setitem__(self, key, value):
        with self._lock:
            if value[1].strip():
                self._append([[*key, *value]])
            super().__setitem__(key, value)

    def _append(self, rows, mode="a", path=None):
        with open(path or self.path, mode, newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)

    def samples(self, stage):
        found = sorted((k[1], v) for k, v in self.items() if k[0] == stage)
        return [s for _, s in found] or None

    def save(self, stage, samples):
        rows = [[stage, k, 0, *s] for k, s in enumerate(samples)]
        with self._lock:
            if all(text.strip() for _, text in samples):
                self._append(rows)
            for row in rows:
                super().__setitem__(tuple(row[:3]), tuple(row[3:]))
        return samples
