# Part of the differentially private LLM synthesizer added in 2026; not adapted from Aug-PE.
# Licensed under the MIT License of this repository (see LICENSE).

"""DP label vote: assign the short column values of synthetic texts from the private rows.

The initial texts are written without any column value, so the model cannot state the
values in the text. Their values come from the private data through one Gaussian mechanism:

  every private row finds its nearest synthetic text (embeddings of the long columns only)
  and adds 1 to (that text, its value of each short column); the domain (synthetic texts x
  released values) is public, so every cell gets noise. A row touches at most one cell per
  short column: L2 sensitivity sqrt(#short columns).

Post-processing (no privacy cost): the noisy counts of each text are averaged with those of
its nearest synthetic texts, then each column is assigned so that its value shares match the
released shares, the texts with the highest share of a value getting that value.
"""

from __future__ import annotations

import math

import numpy as np

from .dp_random import dp_rng


def noisy_label_counts(syn_features: np.ndarray, real_features: np.ndarray,
                       real_values: list[dict], domain: dict[str, list[str]],
                       sigma: float) -> dict[str, np.ndarray]:
    """{column: noisy counts, shape (n_synthetic, n_values)}; the only private step."""
    import faiss

    index = faiss.IndexFlatL2(syn_features.shape[1])
    index.add(syn_features.astype(np.float32))
    _, nn = index.search(real_features.astype(np.float32), 1)
    nn = nn[:, 0]
    n = syn_features.shape[0]
    counts = {c: np.zeros((n, len(vs))) for c, vs in domain.items()}
    pos = {c: {v: i for i, v in enumerate(vs)} for c, vs in domain.items()}
    for j, row in zip(nn, real_values):
        for c in domain:
            k = pos[c].get(row.get(c))
            if k is not None:
                counts[c][j, k] += 1
    std = sigma * math.sqrt(len(domain))
    noise = dp_rng()
    return {c: m + noise.normal(0.0, std, m.shape) for c, m in counts.items()}


def assign_values(noisy: dict[str, np.ndarray], domain: dict[str, list[str]],
                  shares: dict[str, list[float]], syn_features: np.ndarray,
                  smooth_k: int = 10) -> list[dict]:
    """Post-processing: one value per synthetic text and short column."""
    import faiss

    n = syn_features.shape[0]
    k = max(1, min(smooth_k, n))
    if k > 1:
        index = faiss.IndexFlatL2(syn_features.shape[1])
        index.add(syn_features.astype(np.float32))
        _, near = index.search(syn_features.astype(np.float32), k)
    out = [dict() for _ in range(n)]
    for c, values in domain.items():
        m = np.clip(noisy[c], 0.0, None)
        if k > 1:
            m = m[near].mean(axis=1)
        share = m / np.clip(m.sum(axis=1, keepdims=True), 1e-9, None)
        w = np.asarray(shares[c], dtype=float)
        target = np.floor(w / w.sum() * n).astype(int)
        order = np.argsort(w)  # rarest value first, the most common one takes the rest
        free = np.ones(n, dtype=bool)
        for vi in order[:-1]:
            cand = np.nonzero(free)[0]
            take = cand[np.argsort(-share[cand, vi])[:target[vi]]]
            for j in take:
                out[j][c] = values[vi]
            free[take] = False
        for j in np.nonzero(free)[0]:
            out[j][c] = values[order[-1]]
    return out
