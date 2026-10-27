---
title: "AdvGAN合成器"
type: docs
weight: 656
prev: docs/petsard-yaml/synthesizer-yaml/sdv-methods
next: docs/petsard-yaml/synthesizer-yaml/llm-synthesizer
math: true
---

以PyTorch實作的改良式AdvGAN合成器。鑑別器透過[Opacus](https://opacus.ai/)以差分隱私隨機梯度下降(DP-SGD)訓練；生成器為每筆紀錄學習擾動值，並以邊界懲罰項使合成值維持在訓練資料範圍內。本合成器適用於Loader載入各種資料表，包含半結構化JSON紀錄。

{{< callout type="info" >}}
AdvGAN合成器使用`opacus`套件，請依[擴充功能安裝](../../../installation/extended-features)完成安裝。
{{< /callout >}}

## 使用範例

```yaml
Loader:
  load_json:
    filepath: benchmark://customer-churn
Preprocessor:
  default:
    method: 'default'      # 處理缺失值、類別編碼、數值縮放
Synthesizer:
  advgan:
    method: 'AdvGAN'
    sample_num_rows: 1200  # 合成筆數(明確設定)
    sigma: 1.0             # 鑑別器DP-SGD雜訊倍數，預設1.0
    outlier: 10            # 邊界懲罰權重，預設10
    epochs: 5              # 訓練週期數，預設200
Postprocessor:
  default:
    method: 'default'      # 還原原始單位與類別標籤
Reporter:
  save_synthetic:
    method: save_data
    source: Postprocessor
    type: 'json'
```

範例以`epochs: 5`快速完成流程；完整訓練時，將`epochs`提高至預設值200。

## 參數說明

- **method** (`string`，必要) - 固定值：`AdvGAN`(不區分大小寫)
- **sample_num_rows** (`integer`，必要) - 合成資料筆數。使用AdvGAN時明確設定此參數，常見做法為設為訓練資料筆數
- **sigma** (`float`或`integer`，選填，預設`1.0`) - 鑑別器DP-SGD雜訊倍數(Opacus的`noise_multiplier`)。數值越大，每次更新鑑別器時加入的雜訊越多，隱私保護越強
- **outlier** (`float`或`integer`，選填，預設`10`) - 生成器損失中邊界懲罰項的權重。數值越小，生成器產生越多接近各欄位極端值的資料；數值越大，生成值越集中於各欄位範圍內
- **epochs** (`integer`，選填，預設`200`) - 以全部訓練資料訓練週期數

{{< callout type="info" >}}
`epochs`與`sample_num_rows`以整數設定(例如`5`，而非`5.0`)；`sigma`與`outlier`可使用整數或小數。
{{< /callout >}}

## 流程需求

AdvGAN以包含全部欄位的數值張量訓練，因此置於Preprocessor與Postprocessor之間使用：

| 模組 | 於AdvGAN流程的作用 |
|---|---|
| **Preprocessor** (`method: 'default'`) | 處理缺失值、將類別欄位編碼為數值並縮放數值欄位，使每個欄位皆為完整數值 |
| **Synthesizer** (`method: 'AdvGAN'`) | 由前處理後的資料表學習，並生成`sample_num_rows`筆合成資料 |
| **Postprocessor** (`method: 'default'`) | 執行反縮放與反編碼，將合成資料還原為原始單位與原始類別標籤 |

預設的前處理序列請參見[Preprocessor YAML](../../preprocessor-yaml/)。

## 演算法原理

令$x$為前處理後含$F$個欄位的訓練紀錄，$z \sim \mathcal{N}(0, I)$為16維雜訊向量。

1. **殘差式生成** - 生成器$G$以串接向量$[x, z]$為輸入，輸出擾動值並加回該筆紀錄：$\tilde{x} = G([x, z]) + x$
2. **差分隱私鑑別器** - 鑑別器$D$區分真實紀錄與生成紀錄，並以DP-SGD訓練：每筆樣本的梯度先裁剪至L2範數$C = 1.0$，再於梯度總和加入標準差為$\sigma C$的高斯雜訊後更新
3. **具邊界懲罰的生成器** - 生成器損失為對抗損失加上以`outlier`($\lambda$)加權的邊界懲罰項：$L_G = \text{BCE}(D(\tilde{x}), 1) + \lambda \cdot \text{mean}\left(\max(0, \tilde{x} - x_{\max})^2 + \max(0, x_{\min} - \tilde{x})^2\right)$，其中$x_{\max}$與$x_{\min}$為訓練資料各欄位的最大值與最小值。更新生成器時，暫停鑑別器的逐樣本梯度擷取
4. **取樣** - 生成資料時，隨機選取`sample_num_rows`筆訓練紀錄，以訓練完成的生成器搭配新抽樣的雜訊產生合成紀錄，再將每個值截斷至各欄位範圍$[x_{\min}, x_{\max}]$

## 實作細節

| 項目 | 設定 |
|---|---|
| 生成器 | 全連接網路：$F + 16 \rightarrow 64 \rightarrow 128 \rightarrow F$，LeakyReLU(0.2) |
| 鑑別器 | 全連接網路：$F \rightarrow 128 \rightarrow 64 \rightarrow 1$，LeakyReLU(0.2)；以Opacus的`ModuleValidator`完成相容設定 |
| 隱向量維度 | 16 |
| 批次大小 | 2 |
| 逐樣本梯度裁剪上限(`max_grad_norm`) | 1.0 |
| 最佳化器 | 兩個網路皆使用Adam，學習率0.0002，betas為(0.5, 0.999) |
| 損失函數 | 帶logits的二元交叉熵(`BCEWithLogitsLoss`) |
| 運算裝置 | 偵測到CUDA GPU時使用GPU，否則使用CPU，自動選擇 |

訓練開始時，終端機顯示訓練資料筆數與運算裝置：

```
The number of data: <訓練資料筆數>
Using device: cpu
```

## 參數調整

| 目的 | 調整方式 |
|---|---|
| 快速確認流程 | 先以較小的`epochs`執行，例如`5` |
| 完整訓練 | 將`epochs`提高至預設值`200` |
| 提高隱私保護強度 | 提高`sigma` |
| 合成更多接近各欄位極端值的資料 | 降低`outlier` |
| 使生成值集中於各欄位範圍內 | 提高`outlier` |
| 合成資料筆數與訓練集相同 | 將`sample_num_rows`設為訓練資料筆數 |

每個epoch以2筆為一批次處理全部訓練資料，訓練時間隨資料筆數與`epochs`增加。大型資料表先以少量epochs確認設定，再進行完整訓練。

## 與其他合成器比較

同一份YAML可執行多個合成器。下列設定以相同資料分別訓練AdvGAN與預設的Gaussian Copula，Postprocessor、Evaluator及Reporter即為每個實驗分別產生結果：

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

## 使用要點

- AdvGAN之前設定Preprocessor，之後設定Postprocessor
- 以整數明確設定`sample_num_rows`
- `opacus`隨套件群組`all`安裝，PETsARD僅於使用`method: AdvGAN`時載入
- 偵測到CUDA GPU時自動使用GPU；所有設定皆可以CPU執行
- 包含JSON讀取、評測與JSON輸出的完整流程，請參見[JSON資料與AdvGAN合成](../../../getting-started/json-advgan-synthesis)

## 參考文獻

- Xiao, C., Li, B., Zhu, J.-Y., He, W., Liu, M., & Song, D. (2018). Generating Adversarial Examples with Adversarial Networks. Proceedings of the 27th International Joint Conference on Artificial Intelligence (IJCAI-18). https://arxiv.org/abs/1801.02610
- Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K., & Zhang, L. (2016). Deep Learning with Differential Privacy. Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security (CCS). https://doi.org/10.1145/2976749.2978318
- Yousefpour, A., et al. (2021). Opacus: User-Friendly Differential Privacy Library in PyTorch. arXiv:2109.12298. https://arxiv.org/abs/2109.12298
