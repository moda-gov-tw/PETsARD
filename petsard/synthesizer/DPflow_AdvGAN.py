"""DPflow_AdvGAN: AdvGAN's perturbation design with the whole flow under differential privacy. Column types come from
the user's public column declaration (column_spec), not from the data.

Output = seed + G(seed, z). Compared with the AdvGAN method (method: AdvGAN), where DP-SGD covers the discriminator:
1. Column types: read from `column_spec` (a YAML/JSON list, or a Markdown type document converted by
   DPflow_column_spec.py). Categorical columns get a softmax head and are decoded to their listed values; integer
   columns are rounded.
2. Seeds: built from noisy histograms (Gaussian mechanism), so sampling never
   reads a real record and every use of the real data is accounted.
3. Normal scores for the copula come from the released histograms.
4. Inputs to the networks are normal scores and one-hot codes, so lab values and 0/1 flags are on the same scale.
5. Budget: `eps` and `delta` set the whole budget. mu^2 is split 10% histograms, 60% copula, the rest (30%) DP-SGD, and the
   DP-SGD noise multiplier is solved so that the RDP total reaches eps without exceeding it. A privacy report is kept
   in self.privacy.

Neighbouring datasets: add or remove one record. Public inputs: column names, types, category lists, the row count N
and the fixed asinh bound used by the histograms.

PETsARD YAML:
  Synthesizer:
    advgan:
      method: DPflow_AdvGAN
      column_spec: demo/petsard-yaml/synthesizer-yaml/adult-income_column_spec.yaml   # or a type document (.md)
      epsilon: 8          # `eps` also works; .inf gives a noise-free baseline for comparison
      epochs: 5
      batch_size: 256     # `batch` also works
      sample_num_rows: 48842
      seed: 0             # optional: fixed seed to reproduce results; leave it out for a real release (OS randomness)
      keep_from_seed: [income]   # columns copied from the seeds unchanged, e.g. the label columns
      share_histograms: 0.1   # budget split in mu^2 terms; DP-SGD gets 1 - share_histograms - share_copula
      share_copula: 0.6       # defaults 0.1 / 0.6 (DP-SGD 0.3) since v6; v5 and earlier used 0.2 / 0.3
Use it without a Preprocessor: column types come from column_spec, and the synthesizer handles the data within its own privacy accounting.
Needs opacus and dp-accounting.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from torch.utils.data import DataLoader, TensorDataset

from petsard.exceptions import ConfigError
from petsard.synthesizer.DPflow_common import (
    Marginals,
    TwoStageMarginals,
    decode_scores,
    dp_sgd_plan,
    encode_scores,
    head_groups,
    load_column_spec,
    mu_for,
    release_copula_loading,
    resolve_seed,
    seed_report,
    solve_noise_multiplier,
)
from petsard.synthesizer.synthesizer_base import BaseSynthesizer


def load_types(source) -> list[dict]:
    """Declared column types: a .md type document, a column_spec .yaml/.json, or a list."""
    return load_column_spec(source)


# ---- 1. 生成器 / 鑑別器 ----

class Generator(nn.Module):
    def __init__(self, latent_dim, feat_dim, groups, keep=None):
        super().__init__()
        self.groups = groups
        self.keep = keep  # encoded positions copied from the seed unchanged (e.g. the label columns)
        self.net = nn.Sequential(
            nn.Linear(feat_dim + latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, feat_dim),
        )

    def forward(self, seed, z):
        # AdvGAN 殘差設計：seed + G(seed, z)；類別欄再經 softmax，輸出才會是合法的類別
        o = seed + self.net(torch.cat([seed, z], dim=1))
        out = o.clone()
        for _, idx in self.groups.items():
            out[:, idx.reshape(-1)] = torch.softmax(o[:, idx] * 4, 2).reshape(o.size(0), -1)
        if self.keep is not None:
            out[:, self.keep] = seed[:, self.keep]
        return out


class Discriminator(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


# ---- 2. DP-AdvGAN 合成器 ----

class DPflowAdvGANSynthesizer(BaseSynthesizer):
    LATENT = 16

    def __init__(self, config: dict, metadata=None):
        super().__init__(config, metadata)
        self.metadata = None  # column information comes from column_spec
        c = self.config
        if "column_spec" not in c:
            raise ConfigError("DPflow_AdvGAN needs 'column_spec': the column types you declare as public.")
        self.spec = load_types(c["column_spec"])
        self.eps = float(c.get("eps", c.get("epsilon", 1.0)))
        self.delta = float(c.get("delta", 1e-5))
        self.EPOCHS = int(c.get("epochs", 50))
        self.batch_size = int(c.get("batch_size", c.get("batch", 64)))
        self.sample_num_rows = int(c.get("sample_num_rows", 100))
        # seed is optional: set it to reproduce results; leave it out for a real release (operating-system randomness)
        self.seed, self.rng, self._torch_seed = resolve_seed(c)
        self.lr = float(c.get("lr", 5e-4))
        self.two_stage = bool(c.get("two_stage", True))
        self.k = int(c.get("k", 64))
        self.round_integers = bool(c.get("round_integers", True))
        # Budget split in mu^2 terms; DP-SGD gets the rest. Defaults 0.1 and 0.6 (DP-SGD 0.3); v5 and earlier used 0.2 and 0.3.
        self.share_hist = float(c.get("share_histograms", 0.1))
        self.share_copula = float(c.get("share_copula", 0.6))
        if not (0 < self.share_hist < 1 and 0 < self.share_copula < 1 and self.share_hist + self.share_copula < 1):
            raise ConfigError("share_histograms and share_copula must be in (0, 1) and sum to less than 1 (DP-SGD gets the rest).")
        self.keep_from_seed = list(c.get("keep_from_seed") or [])  # columns the generator must not change
        unknown = set(self.keep_from_seed) - {s["name"] for s in self.spec}
        if unknown:
            raise ConfigError(f"keep_from_seed names columns not in column_spec: {sorted(unknown)}")
        self.privacy: dict = {}

    def _check(self, data: pd.DataFrame) -> pd.DataFrame:
        names = [c["name"] for c in self.spec]
        if len(set(names)) != len(names):
            raise ConfigError("column_spec has duplicate names.")
        missing, extra = set(names) - set(data.columns), set(data.columns) - set(names)
        if missing or extra:
            raise ConfigError(f"column_spec and data differ. Missing in data: {sorted(missing)}; not in spec: {sorted(extra)}")
        raw = data[names].copy()
        for c in self.spec:
            if raw[c["name"]].isna().any():
                raise ConfigError(f"Column '{c['name']}' has missing values. Fill them with a public rule first.")
            if c["type"] == "num":
                raw[c["name"]] = raw[c["name"]].astype(float)
            else:
                raw[c["name"]] = raw[c["name"]].astype(object)
                if not set(raw[c["name"]].unique()) <= set(c["values"]):
                    raise ConfigError(f"Column '{c['name']}' has a value outside its 'values' list in the type document.")
        return raw

    def _seeds(self, m: int) -> torch.Tensor:
        """Seeds from the released copula and histograms only: post-processing, no real record is read."""
        z = self.rng.standard_normal((m, self.dim)) @ self.copula_a.T
        return torch.tensor(encode_scores(z, self.marg, self.spec)[0], dtype=torch.float32, device=self.device)

    def _fit(self, data: pd.DataFrame):
        from dp_accounting import dp_event
        from dp_accounting.rdp import rdp_privacy_accountant as rpa

        raw = self._check(data)
        n = len(raw)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self._torch_seed)

        # 1. 預算：mu^2依share_histograms(預設10%)給直方圖、share_copula(預設60%)給copula，其餘(預設30%)給DP-SGD
        q, steps = dp_sgd_plan(n, self.batch_size, self.EPOCHS)
        mu = mu_for(self.eps, self.delta)
        mu_m, mu_c = mu * math.sqrt(self.share_hist), mu * math.sqrt(self.share_copula)

        # 2. 加噪直方圖(依型別)與 DP copula
        self.marg = (TwoStageMarginals if self.two_stage else Marginals)(raw, self.spec, mu_m, self.k, self.rng)
        z = self.marg.scores(raw)
        self.dim = z.shape[1]
        self.copula_a, _ = release_copula_loading(z, mu_c, n, self.rng)

        def account(s):
            acc = rpa.RdpAccountant()
            acc.compose(dp_event.ComposedDpEvent([
                dp_event.GaussianDpEvent(1 / mu_m), dp_event.GaussianDpEvent(1 / mu_c),
                dp_event.SelfComposedDpEvent(dp_event.PoissonSampledDpEvent(q, dp_event.GaussianDpEvent(s)), steps)]))
            return acc.get_epsilon(self.delta)

        no_noise = math.isinf(self.eps)  # epsilon = inf: no noise anywhere, reference only
        sigma = 0.0 if no_noise else solve_noise_multiplier(account, self.eps, 0.3, 200)

        # 3. 判別器的輸入：數值欄是常態分數，類別欄是 one-hot
        enc, self.heads = encode_scores(z, self.marg, self.spec)
        del raw, z
        loader = DataLoader(TensorDataset(torch.tensor(enc, dtype=torch.float32)), batch_size=self.batch_size, shuffle=True)
        feat_dim = enc.shape[1]
        starts = np.cumsum([0] + [m for _, m in self.heads])
        keep = [p for j, c in enumerate(self.spec) if c["name"] in self.keep_from_seed for p in range(starts[j], starts[j + 1])]
        keep = torch.tensor(keep, dtype=torch.long, device=self.device) if keep else None
        self.G = Generator(self.LATENT, feat_dim, head_groups(self.heads), keep).to(self.device)
        self.D = ModuleValidator.fix(Discriminator(feat_dim)).to(self.device)
        opt_G = torch.optim.Adam(self.G.parameters(), lr=self.lr, betas=(0.5, 0.999))
        opt_D = torch.optim.Adam(self.D.parameters(), lr=self.lr, betas=(0.5, 0.999))
        bce = nn.BCEWithLogitsLoss()
        if not no_noise:
            self.D, opt_D, loader = PrivacyEngine().make_private(
                module=self.D, optimizer=opt_D, data_loader=loader, noise_multiplier=sigma, max_grad_norm=1.0)
        hooks = hasattr(self.D, "disable_hooks")

        # 4. 訓練：真實資料只經由 DP-SGD 的判別器
        for _ in range(self.EPOCHS):
            for (real_batch,) in loader:
                bs = real_batch.size(0)
                if bs == 0:  # Poisson sampling can draw an empty batch; still take the noisy step
                    opt_D.zero_grad()
                    opt_D.step()
                    continue
                real_batch = real_batch.to(self.device)
                opt_D.zero_grad()
                fake = self.G(self._seeds(bs), torch.randn(bs, self.LATENT, device=self.device)).detach()
                loss_D = bce(self.D(real_batch), torch.ones(bs, 1, device=self.device)) + \
                    bce(self.D(fake), torch.zeros(bs, 1, device=self.device))
                loss_D.backward()
                opt_D.step()

                if hooks:
                    self.D.disable_hooks()
                opt_G.zero_grad()
                gen_fake = self.G(self._seeds(bs), torch.randn(bs, self.LATENT, device=self.device))
                bce(self.D(gen_fake), torch.ones(bs, 1, device=self.device)).backward()
                opt_G.step()
                if hooks:
                    self.D.enable_hooks()

        self.privacy = {
            "method": "DPflow_AdvGAN", "target_epsilon": self.eps,
            "achieved_epsilon": math.inf if no_noise else float(account(sigma)),
            "delta": self.delta, "accounting": "none (no noise, reference run)" if no_noise else "RDP", "mu_histograms": mu_m, "mu_copula": mu_c,
            "share_histograms": self.share_hist, "share_copula": self.share_copula, "share_dp_sgd": 1 - self.share_hist - self.share_copula,
            "noise_multiplier": sigma, "q": q, "steps": steps, "n": n, **seed_report(self.seed),
            "privacy_unit": "one record", "column_types": "from the column type document (column_spec)",
        }
        if self.config.get("privacy_report"):
            report = Path(self.config["privacy_report"])
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps(self.privacy, ensure_ascii=False, indent=2), encoding="utf-8")
        self._impl = True  # the real data is not kept on this object

    def _sample(self) -> pd.DataFrame:
        """取樣只用已釋出的直方圖、copula 與生成器，不讀真實資料。"""
        with torch.no_grad():
            seeds = self._seeds(self.sample_num_rows)
            out = self.G(seeds, torch.randn(self.sample_num_rows, self.LATENT, device=self.device)).cpu().numpy()
        df = decode_scores(out, self.heads, self.marg, self.spec, self.rng)[[c["name"] for c in self.spec]]
        for c in self.spec:
            if c["type"] == "cat":
                if all(isinstance(v, (int, float, np.integer, np.floating)) for v in c["values"]):
                    df[c["name"]] = pd.to_numeric(df[c["name"]])
            elif c.get("integer") and self.round_integers:
                df[c["name"]] = np.rint(df[c["name"]].to_numpy(dtype=float)).astype("int64")
        return df
