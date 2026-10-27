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
            nn.Linear(LATENT, 256),
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


        self.data_max = torch.tensor(self._data.max().values)
        self.data_min = torch.tensor(self._data.min().values)

        # 1. 直接將整個 DataFrame 轉為 NumPy，再轉成 Tensor
        X_tensor = torch.tensor(self._data.values, dtype=torch.float32)

        # 2. 包裝並建立 DataLoader
        dataset = TensorDataset(X_tensor)
        self.real_loader = DataLoader(dataset, batch_size=2, shuffle=True)

        self.G = Generator(self.LATENT, self.feature_dim)
        self.D = ModuleValidator.fix(Discriminator(self.feature_dim))   # 自動修正不相容層(如有 BatchNorm)

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
                bs = real_batch.size(0)
                # --- 訓練鑑別器(DP) ---
                opt_D.zero_grad()
                fake = self.G(torch.randn(bs, self.LATENT)).detach()
                loss_D = bce(self.D(real_batch), torch.ones(bs, 1)) + bce(self.D(fake), torch.zeros(bs, 1))
                loss_D.backward()
                opt_D.step()                       # Opacus:逐樣本裁剪 + 加噪
                # --- 訓練生成器(一般;後處理) ---
                self.D.disable_hooks()          # 關鍵:產生器更新時不對鑑別器擷取逐樣本梯度
                opt_G.zero_grad()

                gen_fake = self.G(torch.randn(bs, self.LATENT))

                # B. 計算原始的對抗損失 (Adversarial Loss)
                loss_G_adv = bce(self.D(gen_fake), torch.ones(bs, 1))

                # C. 計算邊界懲罰項 (Boundary Penalty)
                # 1. 抓出超過上界的數值 (超過 MAX_VAL 的部分會 > 0，未超過的會 <= 0 並被 relu 變成 0)
                over_upper = torch.relu(gen_fake - self.data_max)
                # 2. 抓出小於下界數值 (小於 MIN_VAL 的部分會 > 0)
                under_lower = torch.relu(self.data_min - gen_fake)
                # 3. 計算超出部分的的平方和平均 (MSE Penalty)
                bound_penalty = torch.mean(over_upper**2 + under_lower**2)

                # D. 組合最終的 Generator Loss
                loss_G = loss_G_adv + self.LAMBDA_BOUND * bound_penalty

                #loss_G = bce(self.D(self.G(torch.randn(bs, self.LATENT))), torch.ones(bs, 1))
                loss_G.backward()
                opt_G.step()
                self.D.enable_hooks()
            #if (epoch + 1) % 50 == 0:
            #    eps = privacy_engine.get_epsilon(delta=DELTA)
            #    print(f"epoch {epoch+1:3d} | loss_D={loss_D.item():.3f} loss_G={loss_G.item():.3f} "
            #        f"| (eps={eps:.2f}, delta={DELTA})-DP")



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

        #torch.manual_seed(42)

        synth_list = []

        with torch.no_grad():

            for _ in range(self.sample_num_rows):
                generated_row = self.G(torch.randn(1, self.LATENT))
                synth_list.append(generated_row)

        synth = torch.cat(synth_list, dim=0)

        synth = torch.maximum(synth, self.data_min)
        synth = torch.minimum(synth, self.data_max)

        synth_df = pd.DataFrame(synth, columns=self.columns)


        return synth_df
