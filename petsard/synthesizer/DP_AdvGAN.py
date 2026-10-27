import pandas as pd
import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from torch.utils.data import DataLoader, TensorDataset

from petsard.synthesizer.synthesizer_base import BaseSynthesizer

# ---- 1. 生成器 / 鑑別器(全連接;對應改良式 AdvGAN) ----

class Generator(nn.Module):
    def __init__(self, LATENT, FEAT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEAT + LATENT, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, FEAT),
        )
    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, FEAT):
        super().__init__()
        # 注意:用 LayerNorm 而非 BatchNorm(Opacus 不支援 BatchNorm)
        self.net = nn.Sequential(
            nn.Linear(FEAT, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.net(x)
"""

class Generator(nn.Module):
    def __init__(self, LATENT, FEAT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEAT + LATENT, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, FEAT),
        )
    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, FEAT):
        super().__init__()
        # 注意:用 LayerNorm 而非 BatchNorm(Opacus 不支援 BatchNorm)
        self.net = nn.Sequential(
            nn.Linear(FEAT, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x)
"""

class AdvGANSynthesizer(BaseSynthesizer):

    def __init__(self, config: dict, metadata):
        """
        初始化合成器。
        參數：
            config   : dict，來自 YAML 的所有配置參數
            metadata : SchemaMetadata，資料表的 Schema 資訊
        """


        super().__init__(config, metadata)

        self.LATENT = 16
        self.LAMBDA_BOUND = 10
        self.sigma = 1.0
        self.EPOCHS = 200
        self.sample_num_rows = 100


        if "sigma" in self.config and isinstance(self.config.get("sigma"), (float,int)):
            self.sigma = self.config.get("sigma")
        if "outlier" in self.config and isinstance(self.config.get("outlier"), (float,int)):
            self.LAMBDA_BOUND = self.config.get("outlier")
        if "epochs" in self.config and isinstance(self.config.get("epochs"), int):
            self.EPOCHS = self.config.get("epochs")
        if "sample_num_rows" in self.config and isinstance(self.config.get("sample_num_rows"), int):
            self.sample_num_rows =  self.config.get("sample_num_rows")

    def _fit(self, data: pd.DataFrame):
        """
        從輸入資料學習模式（訓練階段）。
        參數：
            data : pd.DataFrame，已前處理的訓練資料
                   （類別欄位已編碼，數值欄位已正規化）
        回傳：無（void）
        """
        self._data = data.copy(deep=True)
        self.columns = self._data.columns
        self.feature_dim = len(self._data.columns)

        print("The number of data: " + str(len(self._data)))


        # 宣告裝置並儲存至 self 供 _sample 使用
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 將 data_max 與 data_min 移到 GPU
        self.data_max = torch.tensor(self._data.max().values, dtype=torch.float32, device=self.device)
        self.data_min = torch.tensor(self._data.min().values, dtype=torch.float32, device=self.device)

        # 1. 直接將整個 DataFrame 轉為 NumPy，再轉成 Tensor
        X_tensor = torch.tensor(self._data.values, dtype=torch.float32)

        # 2. 包裝並建立 DataLoader
        dataset = TensorDataset(X_tensor)
        self.real_loader = DataLoader(dataset, batch_size=2, shuffle=True)

        # 將模型移至 GPU
        self.G = Generator(self.LATENT, self.feature_dim).to(self.device)
        self.D = ModuleValidator.fix(Discriminator(self.feature_dim)).to(self.device)   # 自動修正不相容層(如有 BatchNorm)

        opt_G = torch.optim.Adam(self.G.parameters(), lr=2e-4, betas=(0.5, 0.999))
        opt_D = torch.optim.Adam(self.D.parameters(), lr=2e-4, betas=(0.5, 0.999))
        bce = nn.BCEWithLogitsLoss()

        # ---- 2. 只對鑑別器套 DP-SGD ----
        privacy_engine = PrivacyEngine()
        self.D, opt_D, self.real_loader = privacy_engine.make_private(
            module=self.D,
            optimizer=opt_D,
            data_loader=self.real_loader,
            noise_multiplier=self.sigma,   # sigma:越大越私密、品質越低
            max_grad_norm=1.0,      # C:逐樣本梯度裁剪上限
        )

        for _epoch in range(self.EPOCHS):
            for (real_batch,) in self.real_loader:
                # 將批次資料移動到 GPU
                real_batch = real_batch.to(self.device)
                bs = real_batch.size(0)

                # --- 訓練鑑別器(DP) ---
                opt_D.zero_grad()
                randNum = torch.randn(bs, self.LATENT, device=self.device)

                fake = self.G(torch.cat([real_batch, randNum], dim=1)) + real_batch

                # 建立在 GPU 上的 Target 標籤
                loss_D = bce(self.D(real_batch), torch.ones(bs, 1, device=self.device)) + \
                         bce(self.D(fake), torch.zeros(bs, 1, device=self.device))
                loss_D.backward()
                opt_D.step()                       # Opacus:逐樣本裁剪 + 加噪

                # --- 訓練生成器(一般;後處理) ---
                self.D.disable_hooks()          # 關鍵:產生器更新時不對鑑別器擷取逐樣本梯度
                opt_G.zero_grad()

                randNum = torch.randn(bs, self.LATENT, device=self.device)
                gen_fake = self.G(torch.cat([real_batch, randNum], dim=1)) + real_batch

                # B. 計算原始的對抗損失 (Adversarial Loss)
                loss_G_adv = bce(self.D(gen_fake), torch.ones(bs, 1, device=self.device))

                # C. 計算邊界懲罰項 (Boundary Penalty)
                over_upper = torch.relu(gen_fake - self.data_max)
                under_lower = torch.relu(self.data_min - gen_fake)
                bound_penalty = torch.mean(over_upper**2 + under_lower**2)

                # D. 組合最終的 Generator Loss
                loss_G = loss_G_adv + self.LAMBDA_BOUND * bound_penalty

                loss_G.backward()
                opt_G.step()
                self.D.enable_hooks()

        self._impl = True


    def _sample(self) -> pd.DataFrame:
        """
        生成合成資料（取樣階段）。
        參數：無
        回傳：
            pd.DataFrame，合成資料
            - 欄位名稱必須與 fit() 收到的 data 完全一致
            - 欄位順序應一致；資料型別應盡量與原始資料相符
        """

        # Synthesizer.fit() updates config['sample_num_rows'] after __init__ (manual value
        # or source row count), so prefer it over the value read in __init__.
        NUM_GEN_DATA = self.config.get("sample_num_rows") or self.sample_num_rows

        full_tensor_real_data = self.real_loader.dataset.tensors[0].to(self.device)
        num_real_data = len(full_tensor_real_data)

        rand_indices = torch.randint(0, num_real_data, (NUM_GEN_DATA,), device=self.device)
        sampled_real_data = full_tensor_real_data[rand_indices]


        with torch.no_grad():
            randNum = torch.randn(NUM_GEN_DATA, self.LATENT, device=self.device)
            synth = self.G(torch.cat([sampled_real_data, randNum], dim=1)) + sampled_real_data


        synth = torch.maximum(synth, self.data_min)
        synth = torch.minimum(synth, self.data_max)

        # 轉回 CPU 並轉成 NumPy 陣列，才能安全地給 pandas 建立 DataFrame
        synth_df = pd.DataFrame(synth.cpu().numpy(), columns=self.columns)


        return synth_df
