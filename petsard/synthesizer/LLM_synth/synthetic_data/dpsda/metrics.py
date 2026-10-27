# SPDX-License-Identifier: MIT
#
# Feature-space metrics for the LLM synthesizer: FID and k-NN precision/recall.
#
# Original implementation written in 2026 from the published definitions:
#   - Frechet distance between Gaussians fitted to two feature sets
#     (Heusel et al., NeurIPS 2017).
#   - k-NN precision and recall (Kynkaanniemi et al., "Improved Precision and
#     Recall Metric for Assessing Generative Models", NeurIPS 2019).
# Function names and call signatures match dpsda/metrics.py of AI-secure/aug-pe,
# so dpsda/logging.py uses this module unchanged. See NOTICE in
# petsard/synthesizer/LLM_synth/ for the Aug-PE attribution.
"""FID and k-NN precision/recall between two sets of feature vectors."""

import logging
import time

import numpy as np
from scipy.linalg import sqrtm

# Largest number of pairwise distances computed at once (float64, about 128 MiB).
_MAX_BLOCK_ELEMENTS = 1 << 24
# Pairs whose squared distance is below this share of the squared norms are
# recomputed from their difference vector (see _euclidean_block).
_NEAR_TOL = 1e-9
_REPAIR_CHUNK = 1 << 16


def calculate_fid(act1, act2):
    """Frechet distance between Gaussians fitted to the rows of two arrays.

    FID = ||mu1 - mu2||^2 + Tr(S1) + Tr(S2) - 2 Tr((S1 S2)^(1/2)),
    where mu and S are the mean vector and covariance matrix of each set.

    Args:
        act1: array of shape (n_samples_1, n_features).
        act2: array of shape (n_samples_2, n_features).

    Returns:
        numpy.float64: the FID value.
    """
    mu1, mu2 = np.mean(act1, axis=0), np.mean(act2, axis=0)
    # atleast_2d: np.cov returns a scalar for one feature; keep a 1x1 matrix
    sigma1 = np.atleast_2d(np.cov(act1, rowvar=False))
    sigma2 = np.atleast_2d(np.cov(act2, rowvar=False))
    mean_term = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        # numerical noise can leave a tiny imaginary part; the FID is real
        covmean = covmean.real
    return mean_term + np.trace(sigma1 + sigma2 - 2.0 * covmean)


def _block_ranges(n, size):
    for start in range(0, n, size):
        yield start, min(start + size, n)


def _block_shape(n_rows, n_cols, row_batch_size, col_batch_size):
    cols = max(1, min(int(col_batch_size), n_cols))
    rows = max(1, min(int(row_batch_size), n_rows, _MAX_BLOCK_ELEMENTS // cols))
    return rows, cols


def _euclidean_block(x, x_sq, y, y_sq):
    """Euclidean distances between the rows of x and the rows of y."""
    # squared distances via ||x||^2 + ||y||^2 - 2 x.y (matrix product)
    dist = x @ y.T
    dist *= -2.0
    dist += x_sq[:, None]
    dist += y_sq[None, :]
    # the expansion loses precision for (near-)identical rows; recompute those
    # pairs from the differences so that identical rows are exactly 0 apart
    rows, cols = np.nonzero(dist <= _NEAR_TOL * (x_sq.max() + y_sq.max()))
    for start in range(0, rows.size, _REPAIR_CHUNK):
        r = rows[start:start + _REPAIR_CHUNK]
        c = cols[start:start + _REPAIR_CHUNK]
        diff = x[r] - y[c]
        dist[r, c] = np.einsum("ij,ij->i", diff, diff)
    np.maximum(dist, 0.0, out=dist)
    return np.sqrt(dist, out=dist)


def _kth_neighbor_radii(features, ks, row_batch_size, col_batch_size):
    """Distance from each sample to its k-th nearest other sample, per k in ks.

    Returns an array of shape (n_samples, len(ks)).
    """
    n = features.shape[0]
    k_max = max(ks)
    if k_max >= n:
        raise ValueError(
            f"k={k_max} needs more than {k_max} samples, got {n}")
    sq = np.einsum("ij,ij->i", features, features)
    keep = k_max + 1  # the sample itself is the closest entry (distance 0)
    rows, cols = _block_shape(n, n, row_batch_size, col_batch_size)
    radii = np.empty((n, len(ks)))
    for r0, r1 in _block_ranges(n, rows):
        nearest = np.empty((r1 - r0, 0))
        for c0, c1 in _block_ranges(n, cols):
            dist = _euclidean_block(features[r0:r1], sq[r0:r1],
                                    features[c0:c1], sq[c0:c1])
            nearest = np.concatenate([nearest, dist], axis=1)
            if nearest.shape[1] > keep:
                nearest = np.partition(nearest, keep - 1, axis=1)[:, :keep]
        nearest.sort(axis=1)
        radii[r0:r1] = nearest[:, ks]
    return radii


def _covered(queries, manifold, radii, row_batch_size, col_batch_size):
    """Whether each query lies in at least one ball B(manifold[j], radii[j, i]).

    Returns a boolean array of shape (n_queries, radii.shape[1]).
    """
    q_sq = np.einsum("ij,ij->i", queries, queries)
    m_sq = np.einsum("ij,ij->i", manifold, manifold)
    n_q, n_m = queries.shape[0], manifold.shape[0]
    rows, cols = _block_shape(n_q, n_m, row_batch_size, col_batch_size)
    covered = np.zeros((n_q, radii.shape[1]), dtype=bool)
    for r0, r1 in _block_ranges(n_q, rows):
        for c0, c1 in _block_ranges(n_m, cols):
            dist = _euclidean_block(queries[r0:r1], q_sq[r0:r1],
                                    manifold[c0:c1], m_sq[c0:c1])
            for i in range(radii.shape[1]):
                covered[r0:r1, i] |= np.any(dist <= radii[c0:c1, i], axis=1)
    return covered


def knn_precision_recall_features(ref_features, eval_features, nhood_sizes=(3,),
                                  row_batch_size=10000, col_batch_size=50000,
                                  num_gpus=1, debug=True):
    """k-NN precision and recall of eval_features against ref_features.

    Each feature set approximates its manifold by the union of balls centred
    on its samples, the radius of each ball being the distance from that
    sample to its k-th nearest neighbour in the same set. Precision is the
    share of evaluated samples inside the reference manifold; recall is the
    share of reference samples inside the evaluated manifold.

    Args:
        ref_features: array (n_ref, n_features), e.g. private-data embeddings.
        eval_features: array (n_eval, n_features), e.g. synthetic embeddings.
        nhood_sizes: neighbourhood sizes k (default k=3).
        row_batch_size, col_batch_size: upper bounds on the distance block
            size; they change memory use only, not the result.
        num_gpus: kept for interface compatibility; the computation runs on
            the CPU with NumPy.
        debug: when True (the default, as used by dpsda.logging.compute_fid),
            the k-NN evaluation is skipped and precision, recall and f1 are
            reported as 0. Pass False to compute them.

    Returns:
        dict with keys "precision", "recall" and "f1": floats for a single k,
        lists of floats (one per k) when several sizes are given.
    """
    if debug:
        return {"precision": 0, "recall": 0, "f1": 0}

    ref = np.asarray(ref_features, dtype=np.float64)
    evl = np.asarray(eval_features, dtype=np.float64)
    ks = [int(k) for k in nhood_sizes]
    start = time.time()

    ref_radii = _kth_neighbor_radii(ref, ks, row_batch_size, col_batch_size)
    eval_radii = _kth_neighbor_radii(evl, ks, row_batch_size, col_batch_size)
    precision = _covered(evl, ref, ref_radii,
                         row_batch_size, col_batch_size).mean(axis=0)
    recall = _covered(ref, evl, eval_radii,
                      row_batch_size, col_batch_size).mean(axis=0)
    total = precision + recall
    f1 = np.divide(2 * precision * recall, total,
                   out=np.zeros_like(total), where=total > 0)

    logging.info(
        f"k-NN precision/recall (k={ks}): {evl.shape[0]} evaluated vs "
        f"{ref.shape[0]} reference samples in {time.time() - start:.3f}s")
    if len(ks) == 1:
        return {"precision": float(precision[0]), "recall": float(recall[0]),
                "f1": float(f1[0])}
    return {"precision": precision.tolist(), "recall": recall.tolist(),
            "f1": f1.tolist()}
