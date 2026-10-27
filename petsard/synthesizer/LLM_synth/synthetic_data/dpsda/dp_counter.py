# SPDX-License-Identifier: Apache-2.0
# Adapted from AI-secure/aug-pe (https://github.com/AI-secure/aug-pe),
# dpsda/dp_counter.py, licensed under the Apache License 2.0
# (see petsard/synthesizer/LLM_synth/LICENSE-APACHE-2.0.txt and NOTICE).
# Modified by NTUST in 2026: CPU faiss index for Windows compatibility,
# float32 inputs to faiss, comments on the direction of the vote.

from collections import Counter

import faiss
import numpy as np
import torch


def dp_nn_histogram(public_features, private_features, noise_multiplier,
                    num_packing=1, num_nearest_neighbor=1, mode='L2',
                    threshold=0.0):
    assert public_features.shape[0] % num_packing == 0

    num_true_public_features = public_features.shape[0] // num_packing
    if public_features.shape[0] == 0:  # TODO debug, why this case exists
        return np.zeros(shape=num_true_public_features), np.zeros(shape=num_true_public_features)

    # Use CPU instead of GPU for Windows compatibility
    # faiss_res = faiss.StandardGpuResources()
    if mode == 'L2':
        index = faiss.IndexFlatL2(public_features.shape[1])
    # inner product; need normalization (https://github.com/spotify/annoy)
    elif mode == 'IP':
        index = faiss.IndexFlatIP(public_features.shape[1])
    elif mode == 'cos_sim':
        # normalize the embeddings first
        faiss.normalize_L2(public_features)
        faiss.normalize_L2(private_features)
        index = faiss.IndexFlatIP(public_features.shape[1])
    else:
        raise Exception(f'Unknown mode {mode}')
    # Use CPU-only for Windows compatibility
    # if torch.cuda.is_available():
    #     index = faiss.index_cpu_to_gpu(faiss_res, 0, index)

    # logging.info(f'public_features shape : {public_features.shape}')
    # logging.info(f'private_features shape : {private_features.shape}')

    # Index the synthetic (public) candidates; each private sample then votes
    # for its nearest candidate(s). Do NOT swap these: indexing private_features
    # and searching with public_features collapses the histogram to (at most)
    # one vote per synthetic candidate instead of one vote per private sample,
    # making the resulting counts trivially small and easily zeroed out by
    # the DP noise below.
    index.add(public_features.astype(np.float32))
    # logging.info(f'Number of samples in index: {index.ntotal}')

    distance, ids = index.search(private_features.astype(np.float32), k=num_nearest_neighbor)
    # logging.info('Finished search')

    counter = Counter(list(ids.flatten()))
    # shape of the synthetic samples
    count = np.zeros(shape=num_true_public_features)
    for k in counter:
        count[k % num_true_public_features] += counter[k]
    # logging.info(f'Clean count: {count}')
    # logging.info(f'Clean count sum: {np.sum(count)}')
    # logging.info(f'Clean count num>0: {np.sum(count > 0)}')
    # logging.info(f'Largest clean counters: {sorted(count)[::-1][:50]}')
    count = np.asarray(count)
    clean_count = count.copy()
    count += (np.random.normal(size=len(count)) * np.sqrt(num_nearest_neighbor)
              * noise_multiplier)
    # logging.info(f'Noisy count sum: {np.sum(count)}')
    # logging.info(f'Noisy count num>0: {np.sum(count > 0)}')
    # logging.info(f'Largest noisy counters: {sorted(count)[::-1][:50]}')
    count = np.clip(count, a_min=threshold, a_max=None)
    count = count - threshold
    # logging.info(f'Clipped noisy count sum: {np.sum(count)}')
    # logging.info(f'Clipped noisy count num>0: {np.sum(count > 0)}')
    # logging.info(f'Clipped largest noisy counters: {sorted(count)[::-1][:50]}')
    torch.cuda.empty_cache()
    return count, clean_count
