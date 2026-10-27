# SPDX-License-Identifier: Apache-2.0
# Adapted from AI-secure/aug-pe (https://github.com/AI-secure/aug-pe),
# dpsda/feature_extractor.py, licensed under the Apache License 2.0
# (see petsard/synthesizer/LLM_synth/LICENSE-APACHE-2.0.txt and NOTICE).
# Modified by NTUST in 2026: reuse of a loaded model, checks for empty input.

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# https://github.com/UKPLab/sentence-transformer


def extract_features(
        data,
        batch_size=1000,
        model_name="all-mpnet-base-v2",
        model=None):
    # Handle empty data
    if len(data) == 0:
        raise ValueError("Cannot extract features from empty data array")

    # If available, the model is automatically executed on the GPU. You can specify the device for the model like this:

    if model is None:
        model = SentenceTransformer(model_name)  # device='cuda',
        model.eval()

    with torch.no_grad():
        sentence_embeddings = []
        for i in tqdm(range(len(data) // batch_size+1)):
            batch_data = data[i * batch_size:(i + 1) * batch_size]
            if len(batch_data) == 0:
                continue
            embeddings = model.encode(batch_data)
            if len(embeddings) > 0:
                sentence_embeddings.append(embeddings)

    # Check if we have any embeddings
    if len(sentence_embeddings) == 0:
        raise ValueError("No embeddings were generated. All batches were empty or failed.")

    sentence_embeddings = np.concatenate(sentence_embeddings)
    return sentence_embeddings, model
