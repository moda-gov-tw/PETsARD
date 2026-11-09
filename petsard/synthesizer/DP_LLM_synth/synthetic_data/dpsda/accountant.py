# Part of the differentially private LLM synthesizer added in 2026; not adapted from Aug-PE.
# Licensed under the MIT License of this repository (see LICENSE).

"""Privacy accounting for the DP LLM synthesizer (Gaussian differential privacy).

Every step that touches private data is a Gaussian mechanism with L2 sensitivity
normalised to 1 and noise multiplier sigma (noise std = sigma x sensitivity). A
Gaussian mechanism with multiplier sigma is (1/sigma)-GDP, and GDP composes exactly:
mechanisms with mu_1..mu_n give sqrt(sum mu_i^2)-GDP (Dong, Roth & Su, 2019).

mu-GDP implies (eps, delta(eps))-DP for every eps >= 0 with

    delta(eps) = Phi(-eps/mu + mu/2) - e^eps * Phi(-eps/mu - mu/2).

The profile's set-union release adds a separate failure probability delta_s (an item held
by a single record crossing the threshold), so the total delta is delta_G + delta_s.
"""

from __future__ import annotations

import math
from statistics import NormalDist

_SQRT2 = math.sqrt(2.0)
_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


def _log_ndtr(x: float) -> float:
    """log Phi(x), accurate in the far lower tail."""
    if x > -30.0:
        return math.log(0.5 * math.erfc(-x / _SQRT2))
    # asymptotic series: Phi(x) ~ phi(x)/(-x) * (1 - 1/x^2 + 3/x^4)
    x2 = x * x
    return -0.5 * x2 - math.log(-x) - _LOG_SQRT_2PI + math.log1p(-1.0 / x2 + 3.0 / (x2 * x2))


def gdp_delta(eps: float, mu: float) -> float:
    """delta(eps) of a mu-GDP mechanism."""
    if mu <= 0:
        return 0.0
    if math.isinf(mu):
        return 1.0
    a = -eps / mu + mu / 2.0
    b = -eps / mu - mu / 2.0
    first = 0.5 * math.erfc(-a / _SQRT2)
    second = math.exp(eps + _log_ndtr(b))
    return max(0.0, first - second)


def gdp_epsilon(mu: float, delta: float) -> float:
    """Smallest eps with delta(eps) <= delta for a mu-GDP mechanism."""
    if mu <= 0:
        return 0.0
    if math.isinf(mu):
        return math.inf
    if gdp_delta(0.0, mu) <= delta:
        return 0.0
    lo, hi = 0.0, 1.0
    while gdp_delta(hi, mu) > delta:
        hi *= 2.0
        if hi > 1e6:
            return math.inf
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if gdp_delta(mid, mu) > delta:
            lo = mid
        else:
            hi = mid
    return hi


def gdp_mu(eps: float, delta: float) -> float:
    """Largest mu whose mu-GDP guarantee gives (eps, delta)-DP."""
    if math.isinf(eps):
        return math.inf
    lo, hi = 0.0, 1.0
    while gdp_delta(eps, hi) < delta:
        hi *= 2.0
        if hi > 1e6:
            return math.inf
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if gdp_delta(eps, mid) < delta:
            lo = mid
        else:
            hi = mid
    return lo


def compose_mu(sigmas: list[float]) -> float:
    """mu of the composition of Gaussian mechanisms with the given noise multipliers."""
    if any(s == 0 for s in sigmas):
        return math.inf
    return math.sqrt(sum(1.0 / (s * s) for s in sigmas))


def threshold_counts(sigma: float, delta_s: float, cap: int) -> float:
    """Release threshold for thresholded noisy counts (a Gaussian set union).

    Every row adds 1 to each of at most `cap` items, so the L2 sensitivity is sqrt(cap)
    and the noise std is sigma * sqrt(cap). If t <= cap items belong to one row only (count
    1), none is released with probability Phi((tau - 1) / (sigma sqrt(cap)))^t; the threshold
    makes this at least 1 - delta_s for every t <= cap."""
    if sigma == 0:
        return 1.0
    return 1.0 + sigma * math.sqrt(cap) * NormalDist().inv_cdf((1.0 - delta_s) ** (1.0 / cap))


def plan_budget(
    epsilon: float,
    delta: float,
    epochs: int,
    profile_fraction: float = 0.2,
    delta_s: float = 1e-6,
    n_profile_queries: int = 4,
    label_fraction: float = 0.0,
    wording_fraction: float = 0.0,
) -> dict:
    """Split an (epsilon, delta) budget into the noise multipliers of every private step.

    The budget is split in mu^2: `profile_fraction` goes to the profile's queries (equal
    shares), `label_fraction` to the label vote that assigns the short column values of the
    initial texts (one query; 0 = no label vote), `wording_fraction` to the value-wording
    counts when that query has a budget of its own (0 = it shares the profile's), the rest
    to the `epochs` vote rounds
    (equal shares). epsilon = inf turns all noise off (a reference run for comparison).
    """
    if not epsilon > 0:
        raise ValueError("epsilon must be > 0 (use .inf for a no-noise reference run)")
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    if not 0 < delta_s < delta:
        raise ValueError("profile_delta_s must be in (0, delta)")
    if epochs < 0:
        raise ValueError("epochs must be >= 0")
    if not 0 <= label_fraction < 1 or not 0 <= wording_fraction < 1:
        raise ValueError("label_vote / value_wording budget fractions must be in [0, 1)")
    other = label_fraction + wording_fraction
    if epochs == 0:
        profile_fraction = 1.0 - other
    if not 0 < profile_fraction <= 1 - other:
        raise ValueError("profile_budget_fraction must be in (0, 1 - the other fractions]")
    if epochs > 0 and profile_fraction + other >= 1:
        raise ValueError("the profile, label-vote and wording fractions must leave budget for the votes")

    delta_gauss = delta - delta_s
    if math.isinf(epsilon):
        sigma_profile = 0.0
        sigma_vote = 0.0 if epochs else None
        sigma_label = 0.0 if label_fraction > 0 else None
        sigma_wording = 0.0 if wording_fraction > 0 else None
        mu = math.inf
    else:
        mu = gdp_mu(epsilon, delta_gauss)
        mu2_profile = profile_fraction * mu * mu / n_profile_queries
        sigma_profile = 1.0 / math.sqrt(mu2_profile)
        sigma_label = 1.0 / math.sqrt(label_fraction * mu * mu) if label_fraction > 0 else None
        sigma_wording = 1.0 / math.sqrt(wording_fraction * mu * mu) if wording_fraction > 0 else None
        rest = 1.0 - profile_fraction - label_fraction - wording_fraction
        sigma_vote = math.sqrt(epochs / (rest * mu * mu)) if epochs else None

    return {
        "epsilon": epsilon,
        "delta": delta,
        "delta_gauss": delta_gauss,
        "delta_s": delta_s,
        "mu": mu,
        "epochs": epochs,
        "profile_fraction": profile_fraction,
        "label_fraction": label_fraction,
        "wording_fraction": wording_fraction,
        "n_profile_queries": n_profile_queries,
        "sigma_profile": sigma_profile,
        "sigma_label": sigma_label,
        "sigma_wording": sigma_wording,
        "sigma_vote": sigma_vote,
    }


def realized_epsilon(sigma_profile: float, n_profile_queries: int, sigma_vote: float | None,
                     vote_rounds: int, delta_gauss: float, sigma_label: float | None = None,
                     label_votes: int = 0, sigma_wording: float | None = None,
                     wording_queries: int = 0) -> tuple[float, float]:
    """(mu, epsilon) actually spent by the steps that ran."""
    sigmas = [sigma_profile] * n_profile_queries
    if label_votes and sigma_label is not None:
        sigmas += [sigma_label] * label_votes
    if wording_queries and sigma_wording is not None:
        sigmas += [sigma_wording] * wording_queries
    if vote_rounds and sigma_vote is not None:
        sigmas += [sigma_vote] * vote_rounds
    mu = compose_mu(sigmas)
    return mu, gdp_epsilon(mu, delta_gauss)
