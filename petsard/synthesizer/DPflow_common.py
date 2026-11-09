"""Shared parts of the complete-flow differentially private synthesizers.

Neighbouring datasets: add or remove one record. The row count N is treated as public.
Public inputs: column names and types, category lists, the fixed asinh transform and its bound B.
Every other use of the real data goes through a Gaussian mechanism or DP-SGD and is accounted.
No real minimum, maximum, mean or standard deviation is used directly.
Sampling never touches a real record, so all later steps are post-processing of the private outputs.
"""

from __future__ import annotations

import json
import math
import secrets
from abc import abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import brentq
from scipy.stats import norm

from petsard.exceptions import ConfigError
from petsard.synthesizer.synthesizer_base import BaseSynthesizer

B = math.asinh(1e7)  # public bound for numeric values on the asinh scale
DEFAULT_DELTA = 1e-5
MARGINAL_SHARE = 0.2  # share of mu^2 spent on the histograms


SEED_GIVEN = "seed set in the configuration (fixed; use it to reproduce reported results and keep it private in a real release)"
SEED_SYSTEM = "operating-system randomness (no seed set; every run draws new noise)"


def resolve_seed(config: dict) -> tuple[int | None, np.random.Generator, int]:
    """Random sources for the DP noise.

    With `seed` in the configuration: numpy uses seed + 1000 and torch uses seed, exactly as before, so a run can be
    reproduced. Without `seed` (or `seed: null`): numpy gets 128 bits and torch 64 bits from the operating system
    (`secrets`), so the noise cannot be recomputed from the code and configuration. Returns (seed or None, numpy
    generator, torch seed); the torch seed is never written to any report.
    """
    seed = config.get("seed")
    if seed is None:
        return None, np.random.default_rng(secrets.randbits(128)), secrets.randbits(64)
    seed = int(seed)
    return seed, np.random.default_rng(seed + 1000), seed


def seed_report(seed: int | None) -> dict:
    """Privacy-report fields for the seed: the value only when it was set in the configuration."""
    return {"seed": seed, "seed_source": SEED_GIVEN} if seed is not None else {"seed_source": SEED_SYSTEM}


def gdp_delta(mu: float, eps: float) -> float:
    return norm.cdf(-eps / mu + mu / 2) - math.exp(eps) * norm.cdf(-eps / mu - mu / 2)


def mu_for(eps: float, delta: float = DEFAULT_DELTA) -> float:
    """GDP mu for (eps, delta). eps = inf means no noise: mu = inf, so every noise scale below becomes 0."""
    if math.isinf(eps):
        return math.inf
    return brentq(lambda m: gdp_delta(m, eps) - delta, 1e-3, 50)


def eps_for_mu(mu: float, delta: float = DEFAULT_DELTA) -> float:
    if math.isinf(mu):
        return math.inf
    return brentq(lambda e: gdp_delta(mu, e) - delta, 0, 300)


def solve_noise_multiplier(account, eps: float, lo: float, hi: float, xtol: float = 1e-3) -> float:
    """DP-SGD noise multiplier whose accounted epsilon does not exceed the target `eps`.

    `account(s)` returns the epsilon of the whole flow for noise multiplier s; it decreases as s grows. brentq stops
    within xtol of the root, so its answer can lie on either side of it. If that answer already gives
    account(s) <= eps it is used unchanged, so a run with the same seed gives the same result as before this check
    was added. Otherwise the noise is raised just enough: in steps of xtol until the target is met, then by bisection
    to 1e-9, always keeping the side that meets the target.
    """
    sigma = brentq(lambda s: account(s) - eps, lo, hi, xtol=xtol)
    if account(sigma) <= eps:
        return sigma
    low, high = sigma, sigma + xtol
    while account(high) > eps:
        low, high = high, high + xtol
    while high - low > 1e-9:
        mid = (low + high) / 2
        if account(mid) <= eps:
            high = mid
        else:
            low = mid
    return high


class Marginals:
    """Noisy histograms for all columns, released together under one Gaussian mechanism."""

    def __init__(self, raw, spec, mu_m, k, rng):
        self.spec, self.k = spec, k
        noise = math.sqrt(len(spec)) / mu_m
        self.prob, self.order = {}, {}
        for c in spec:
            if c["type"] == "num":
                idx = self.num_bin(raw[c["name"]].to_numpy(dtype=float))
                m = k
            else:
                vals = list(c["values"])
                pos = {v: i for i, v in enumerate(vals)}
                idx = raw[c["name"]].map(pos).to_numpy(dtype=int)
                m = len(vals)
            h = np.bincount(idx, minlength=m).astype(float) + rng.normal(0, noise, m)
            h[h < (3 * noise if c["type"] == "num" else 0)] = 0  # post-processing: drop noise-only bins
            if h.sum() <= 0:
                h = np.ones(m)
            p = h / h.sum()
            if c["type"] == "cat":
                order = np.argsort(-p)
                self.order[c["name"]] = order  # order by released frequency: post-processing
                p = p[order]
            self.prob[c["name"]] = p

    def num_bin(self, x):
        return np.clip(((np.arcsinh(np.nan_to_num(x)) + B) / (2 * B) * self.k).astype(int), 0, self.k - 1)

    def scores(self, raw):
        out = np.zeros((len(raw), len(self.spec)))
        for j, c in enumerate(self.spec):
            p = self.prob[c["name"]]
            cdf = np.cumsum(p)
            if c["type"] == "num":
                i = self.num_bin(raw[c["name"]].to_numpy(dtype=float))
            else:
                pos = {v: r for r, v in enumerate(np.array(c["values"], dtype=object)[self.order[c["name"]]])}
                i = raw[c["name"]].map(pos).to_numpy(dtype=int)
            u = cdf[i] - 0.5 * p[i]
            out[:, j] = norm.ppf(np.clip(u, 1e-4, 1 - 1e-4))
        return out

    def invert(self, z, rng):
        out = {}
        for j, c in enumerate(self.spec):
            p = self.prob[c["name"]]
            cdf = np.cumsum(p)
            i = np.clip(np.searchsorted(cdf, norm.cdf(z[:, j])), 0, len(p) - 1)
            if c["type"] == "num":
                lo = i + rng.random(len(i))
                v = np.sinh(lo / self.k * 2 * B - B)
                # The bin holding exactly 0 maps back to 0: a public rule, so zero-heavy columns keep their spike.
                v[i == int((B / (2 * B)) * self.k)] = 0.0
                out[c["name"]] = v
            else:
                vals = np.array(c["values"], dtype=object)[self.order[c["name"]]]
                out[c["name"]] = vals[i]
        return pd.DataFrame(out)


class TwoStageMarginals(Marginals):
    """For wide tables with narrow numeric ranges. A public asinh grid first locates each column's window, then a
    finer histogram is released inside it. Both are Gaussian mechanisms: mu_m^2 = mu_a^2 + mu_b^2."""

    def __init__(self, raw, spec, mu_m, k, rng):
        self.spec, self.k = spec, k
        self.prob, self.order, self.window = {}, {}, {}
        mu_a = mu_b = mu_m / math.sqrt(2)
        nums = [c for c in spec if c["type"] == "num"]
        noise_a, noise_b = math.sqrt(len(spec)) / mu_a, math.sqrt(max(len(nums), 1)) / mu_b
        k1 = 128
        for c in spec:
            if c["type"] == "cat":
                vals = list(c["values"])
                pos = {v: i for i, v in enumerate(vals)}
                h = np.bincount(raw[c["name"]].map(pos).to_numpy(dtype=int), minlength=len(vals)).astype(float) + rng.normal(0, noise_a, len(vals))
                h[h < 0] = 0
                if h.sum() <= 0:
                    h = np.ones(len(vals))
                p = h / h.sum()
                order = np.argsort(-p)
                self.order[c["name"]], self.prob[c["name"]] = order, p[order]
            else:
                x = np.arcsinh(raw[c["name"]].to_numpy(dtype=float))
                i1 = np.clip(((x + B) / (2 * B) * k1).astype(int), 0, k1 - 1)
                h1 = np.bincount(i1, minlength=k1).astype(float) + rng.normal(0, noise_a, k1)
                keep = np.where(h1 > 3 * noise_a)[0]
                lo, hi = (keep.min(), keep.max() + 1) if len(keep) else (k1 // 2 - 1, k1 // 2 + 1)
                w = (lo / k1 * 2 * B - B, hi / k1 * 2 * B - B)
                self.window[c["name"]] = w
                i2 = self._fine(x, w)
                h2 = np.bincount(i2, minlength=k).astype(float) + rng.normal(0, noise_b, k)
                h2[h2 < 2 * noise_b] = 0
                if h2.sum() <= 0:
                    h2 = np.ones(k)
                self.prob[c["name"]] = h2 / h2.sum()

    def _fine(self, x, w):
        return np.clip(((x - w[0]) / max(w[1] - w[0], 1e-9) * self.k).astype(int), 0, self.k - 1)

    def scores(self, raw):
        out = np.zeros((len(raw), len(self.spec)))
        for j, c in enumerate(self.spec):
            p = self.prob[c["name"]]
            cdf = np.cumsum(p)
            if c["type"] == "num":
                i = self._fine(np.arcsinh(raw[c["name"]].to_numpy(dtype=float)), self.window[c["name"]])
            else:
                pos = {v: r for r, v in enumerate(np.array(c["values"], dtype=object)[self.order[c["name"]]])}
                i = raw[c["name"]].map(pos).to_numpy(dtype=int)
            out[:, j] = norm.ppf(np.clip(cdf[i] - 0.5 * p[i], 1e-4, 1 - 1e-4))
        return out

    def invert(self, z, rng):
        out = {}
        for j, c in enumerate(self.spec):
            p = self.prob[c["name"]]
            i = np.clip(np.searchsorted(np.cumsum(p), norm.cdf(z[:, j])), 0, len(p) - 1)
            if c["type"] == "num":
                w = self.window[c["name"]]
                out[c["name"]] = np.sinh(w[0] + (i + rng.random(len(i))) / self.k * (w[1] - w[0]))
            else:
                out[c["name"]] = np.array(c["values"], dtype=object)[self.order[c["name"]]][i]
        return pd.DataFrame(out)


def encode_scores(z, marg, spec):
    """Numeric columns stay as normal scores; categorical columns become one-hot in released-frequency order."""
    parts, heads = [], []
    for j, c in enumerate(spec):
        if c["type"] == "num":
            parts.append(z[:, [j]])
            heads.append(("num", 1))
        else:
            m = len(c["values"])
            cdf = np.cumsum(marg.prob[c["name"]])
            i = np.clip(np.searchsorted(cdf, norm.cdf(z[:, j])), 0, m - 1)
            parts.append(np.eye(m)[i])
            heads.append(("cat", m))
    return np.hstack(parts), heads


def decode_scores(o, heads, marg, spec, rng):
    zs, s = np.zeros((len(o), len(spec))), 0
    for j, (kind, m) in enumerate(heads):
        if kind == "num":
            zs[:, j] = o[:, s]
        else:
            p = marg.prob[spec[j]["name"]]
            i = o[:, s:s + m].argmax(1)
            zs[:, j] = norm.ppf(np.clip(np.cumsum(p)[i] - 0.5 * p[i], 1e-6, 1 - 1e-6))
        s += m
    return marg.invert(zs, rng)


def release_copula_loading(z, mu_c, n, rng):
    """Clip each row, release the second-moment matrix with Gaussian noise (Frobenius sensitivity d), return a loading."""
    d = z.shape[1]
    z = z * np.minimum(1, math.sqrt(d) / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12))
    e = rng.normal(0, d / mu_c, (d, d))
    cov = (z.T @ z + np.triu(e) + np.triu(e, 1).T) / n  # symmetric noise: the upper triangle carries the release
    w, v = np.linalg.eigh((cov + cov.T) / 2)
    cov = (v * np.clip(w, 1e-3, None)) @ v.T
    dg = np.sqrt(np.diag(cov))
    w, v = np.linalg.eigh(cov / np.outer(dg, dg))
    return v * np.sqrt(np.clip(w, 1e-6, None)), d / mu_c


def head_groups(heads):
    """Group categorical heads by size, so hundreds of softmax heads run as one tensor operation."""
    import torch

    groups, s = {}, 0
    for kind, m in heads:
        if kind == "cat":
            groups.setdefault(m, []).append(list(range(s, s + m)))
        s += m
    return {m: torch.tensor(v) for m, v in groups.items()}


def dp_sgd_plan(n, batch, epochs):
    """Opacus draws batches by Poisson sampling at rate 1/len(loader), so accounting uses that rate and step count."""
    steps = math.ceil(n / batch)
    return 1 / steps, epochs * steps


def load_column_spec(source) -> list[dict]:
    """A column spec is schema knowledge the user declares as public.

    Accepts a Markdown type document (.md, see DPflow_column_spec.py), a YAML or JSON path, or a list.
    """
    if isinstance(source, str | Path) and str(source).lower().endswith((".md", ".txt")):
        from petsard.synthesizer.DPflow_column_spec import read_report

        spec, errors, _ = read_report(Path(source))
        if errors:
            raise ConfigError("Column type document has problems: " + "; ".join(errors[:10]))
        source = spec
    if isinstance(source, str | Path):
        text = Path(source).read_text(encoding="utf-8")
        loaded = json.loads(text) if str(source).lower().endswith(".json") else yaml.safe_load(text)
    else:
        loaded = source
    if isinstance(loaded, dict):
        loaded = loaded.get("columns")
    if not isinstance(loaded, list) or not loaded:
        raise ConfigError("column_spec must be a non-empty list of columns, or a file holding one under 'columns'.")
    for c in loaded:
        if c.get("type") not in ("num", "cat") or "name" not in c:
            raise ConfigError(f"Every column needs a name and type 'num' or 'cat': {c}")
        if c["type"] == "cat" and not c.get("values"):
            raise ConfigError(f"Categorical column '{c['name']}' needs a public 'values' list.")
    return loaded


class DPflowBase(BaseSynthesizer):
    """Common behaviour: parameter handling, input checks, seeding, privacy report and output types.

    Subclasses implement _release(raw, spec) and _generate(n).
    Set the YAML without Preprocessor: the default PETsARD preprocessing computes statistics from the real data,
    which this privacy accounting does not cover.
    """

    NEEDS_TORCH = False

    def __init__(self, config: dict, metadata=None):
        super().__init__(config, metadata)
        # Schema statistics and precision are inferred from the real data, so they are not used here.
        self.metadata = None
        if "epsilon" not in config:
            raise ConfigError(f"{config['syn_method']} requires 'epsilon'.")
        if "column_spec" not in config:
            raise ConfigError(f"{config['syn_method']} requires 'column_spec', the columns you declare as public.")
        self.eps = float(config["epsilon"])
        self.delta = float(config.get("delta", DEFAULT_DELTA))
        # seed is optional: set it to reproduce results; leave it out for a real release (operating-system randomness)
        self.seed = None if config.get("seed") is None else int(config["seed"])
        self.two_stage = bool(config.get("two_stage", False))
        self.round_integers = bool(config.get("round_integers", True))
        self.spec = load_column_spec(config["column_spec"])
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
            name = c["name"]
            if raw[name].isna().any():
                raise ConfigError(f"Column '{name}' has missing values. Fill them with a public rule first.")
            if c["type"] == "num":
                raw[name] = raw[name].astype(float)
            else:
                raw[name] = raw[name].astype(object)
                if not set(raw[name].unique()) <= set(c["values"]):
                    raise ConfigError(f"Column '{name}' has a value outside its public 'values' list.")
        return raw

    def _marginals(self, raw, mu_m, k):
        return (TwoStageMarginals if self.two_stage else Marginals)(raw, self.spec, mu_m, k, self.rng)

    def _k(self, n):
        return int(self.config.get("k") or (512 if n < 5000 else 2048))

    def _fit(self, data: pd.DataFrame) -> None:
        raw = self._check(data)
        self.training_data_rows = len(raw)
        self.seed, self.rng, torch_seed = resolve_seed({"seed": self.seed})
        if self.NEEDS_TORCH:
            import torch

            torch.manual_seed(torch_seed)
            torch.set_num_threads(int(self.config.get("threads") or 1))
        self._release(raw, self.spec)
        self.privacy.update(method=self.config["syn_method"], target_epsilon=self.eps, delta=self.delta,
                            **seed_report(self.seed), n=len(raw))
        self._logger.info(f"Privacy report: {self.privacy}")
        if self.config.get("privacy_report"):
            report = Path(self.config["privacy_report"])
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps(self.privacy, ensure_ascii=False, indent=2), encoding="utf-8")
        self._impl = True  # the real data is not kept on this object

    def _sample(self) -> pd.DataFrame:
        n = self.config.get("sample_num_rows") or self.training_data_rows
        df = self._generate(int(n))
        df = df[[c["name"] for c in self.spec]]
        for c in self.spec:
            name = c["name"]
            if c["type"] == "cat":
                if all(isinstance(v, int | float | np.integer | np.floating) for v in c["values"]):
                    df[name] = pd.to_numeric(df[name])
            elif c.get("integer") and self.round_integers:
                df[name] = np.rint(df[name].to_numpy(dtype=float)).astype("int64")
        return df

    @abstractmethod
    def _release(self, raw: pd.DataFrame, spec: list[dict]) -> None:
        """Spend the privacy budget and keep only private outputs."""

    @abstractmethod
    def _generate(self, n: int) -> pd.DataFrame:
        """Produce n rows from the private outputs only."""
