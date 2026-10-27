---
title: "AdvGAN Synthesizer"
type: docs
weight: 656
prev: docs/petsard-yaml/synthesizer-yaml/sdv-methods
next: docs/petsard-yaml/synthesizer-yaml/llm-synthesizer
math: true
---

An improved AdvGAN synthesizer implemented with PyTorch. Its discriminator is trained with differentially private stochastic gradient descent (DP-SGD) through [Opacus](https://opacus.ai/), and its generator adds a learned perturbation to each record under a boundary penalty that keeps synthetic values within the range of the training data. The synthesizer works on any table that Loader produces, including semi-structured JSON records.

{{< callout type="info" >}}
The AdvGAN synthesizer uses the `opacus` package. Install it as described in [Extended Features Setup](../../../installation/extended-features).
{{< /callout >}}

## Usage Examples

```yaml
Loader:
  load_json:
    filepath: benchmark://customer-churn
Preprocessor:
  default:
    method: 'default'      # Handles missing values, encodes categories, scales numbers
Synthesizer:
  advgan:
    method: 'AdvGAN'
    sample_num_rows: 1200  # Number of rows to generate (set explicitly)
    sigma: 1.0             # DP-SGD noise multiplier for the discriminator, default: 1.0
    outlier: 10            # Weight of the boundary penalty, default: 10
    epochs: 5              # Training epochs, default: 200
Postprocessor:
  default:
    method: 'default'      # Restores original units and category labels
Reporter:
  save_synthetic:
    method: save_data
    source: Postprocessor
    type: 'json'
```

The example uses `epochs: 5` so that the pipeline completes quickly; increase `epochs` toward the default of 200 for full training.

## Parameters

- **method** (`string`, required) - Fixed value: `AdvGAN` (case-insensitive)
- **sample_num_rows** (`integer`, required) - Number of synthetic rows to generate. Set it explicitly for AdvGAN; the number of training rows is a common choice
- **sigma** (`float` or `integer`, optional, default `1.0`) - Noise multiplier of DP-SGD on the discriminator (the `noise_multiplier` of Opacus). A larger value adds more noise to every discriminator update and gives stronger privacy protection
- **outlier** (`float` or `integer`, optional, default `10`) - Weight of the boundary penalty in the generator loss. A smaller value lets the generator produce more values near the extremes of each column; a larger value keeps generated values more tightly inside the column range
- **epochs** (`integer`, optional, default `200`) - Number of passes over the training data

{{< callout type="info" >}}
Write `epochs` and `sample_num_rows` as integers (for example `5`, not `5.0`). `sigma` and `outlier` accept integers or decimals.
{{< /callout >}}

## Pipeline Requirements

AdvGAN trains on a numeric tensor that contains every column, so use it between a Preprocessor and a Postprocessor:

| Module | Role for AdvGAN |
|---|---|
| **Preprocessor** (`method: 'default'`) | Handles missing values, encodes categorical columns as numbers and scales numeric columns, so that every column is numeric and complete |
| **Synthesizer** (`method: 'AdvGAN'`) | Learns from the preprocessed table and generates `sample_num_rows` synthetic rows |
| **Postprocessor** (`method: 'default'`) | Reverses scaling and encoding, returning synthetic data in original units with the original category labels |

The default Preprocessor sequence is described in [Preprocessor YAML](../../preprocessor-yaml/).

## Algorithm Principles

Let $x$ be a preprocessed training record with $F$ columns, and $z \sim \mathcal{N}(0, I)$ a 16-dimensional noise vector.

1. **Residual generation** - The generator $G$ takes the concatenation $[x, z]$ and outputs a perturbation that is added back to the record: $\tilde{x} = G([x, z]) + x$
2. **Private discriminator** - The discriminator $D$ separates real records from generated ones. It is trained with DP-SGD: every per-sample gradient is clipped to an L2 norm of $C = 1.0$, and Gaussian noise with standard deviation $\sigma C$ is added to the summed gradients before each update
3. **Boundary-aware generator** - The generator loss is the adversarial loss plus a boundary penalty weighted by `outlier` ($\lambda$): $L_G = \text{BCE}(D(\tilde{x}), 1) + \lambda \cdot \text{mean}\left(\max(0, \tilde{x} - x_{\max})^2 + \max(0, x_{\min} - \tilde{x})^2\right)$, where $x_{\max}$ and $x_{\min}$ are the column-wise maximum and minimum of the training data. While the generator is updated, per-sample gradient capture on the discriminator is paused
4. **Sampling** - To generate data, the synthesizer draws `sample_num_rows` training records at random, applies the trained generator with fresh noise, and clips every value to the column-wise range $[x_{\min}, x_{\max}]$

## Implementation Details

| Item | Setting |
|---|---|
| Generator | Fully connected: $F + 16 \rightarrow 64 \rightarrow 128 \rightarrow F$, LeakyReLU (0.2) |
| Discriminator | Fully connected: $F \rightarrow 128 \rightarrow 64 \rightarrow 1$, LeakyReLU (0.2); prepared with the Opacus `ModuleValidator` |
| Latent dimension | 16 |
| Batch size | 2 |
| Per-sample gradient clipping (`max_grad_norm`) | 1.0 |
| Optimizer | Adam for both networks, learning rate 0.0002, betas (0.5, 0.999) |
| Loss | Binary cross-entropy with logits (`BCEWithLogitsLoss`) |
| Device | CUDA GPU when available, otherwise CPU, selected automatically |

At the start of training the synthesizer prints the number of training rows and the device in use:

```
The number of data: <number of training rows>
Using device: cpu
```

## Parameter Tuning

| Goal | Adjustment |
|---|---|
| Confirm the pipeline quickly | Start with a small `epochs`, for example `5` |
| Full training | Increase `epochs` toward the default `200` |
| Stronger privacy protection | Increase `sigma` |
| More values near the extremes of each column | Decrease `outlier` |
| Values concentrated inside each column range | Increase `outlier` |
| Synthetic data of the same size as the training set | Set `sample_num_rows` to the number of training rows |

Training time grows with the number of rows and `epochs`, because each epoch processes the training data in batches of 2. For large tables, confirm the configuration with a few epochs first.

## Comparing with Other Synthesizers

Several synthesizers can run in the same YAML. The following block trains AdvGAN and the default Gaussian Copula on the same data; Postprocessor, Evaluator and Reporter then produce one result for each experiment:

```yaml
Synthesizer:
  advgan:
    method: 'AdvGAN'
    sample_num_rows: 1200
    sigma: 1.0
    outlier: 10
    epochs: 5
  gaussian-copula:
    method: 'default'
```

## Notes

- Use a Preprocessor before AdvGAN and a Postprocessor after it
- Set `sample_num_rows` explicitly as an integer
- `opacus` is installed with the `all` extra; PETsARD loads it only when `method: AdvGAN` is used
- A CUDA GPU is used automatically when available; all settings also run on a CPU
- For an end-to-end walkthrough with JSON input, evaluation and JSON output, see [JSON Data and AdvGAN Synthesis](../../../getting-started/json-advgan-synthesis)

## References

- Xiao, C., Li, B., Zhu, J.-Y., He, W., Liu, M., & Song, D. (2018). Generating Adversarial Examples with Adversarial Networks. Proceedings of the 27th International Joint Conference on Artificial Intelligence (IJCAI-18). https://arxiv.org/abs/1801.02610
- Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K., & Zhang, L. (2016). Deep Learning with Differential Privacy. Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security (CCS). https://doi.org/10.1145/2976749.2978318
- Yousefpour, A., et al. (2021). Opacus: User-Friendly Differential Privacy Library in PyTorch. arXiv:2109.12298. https://arxiv.org/abs/2109.12298
