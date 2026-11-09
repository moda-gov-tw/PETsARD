"""DPflow_Copula: DP Gaussian copula. Not a GAN. Many-to-many.

Noisy per-column histograms (L2 sensitivity sqrt(d)) plus a noisy second-moment matrix of clipped normal scores
(Frobenius sensitivity d). Total privacy under Gaussian DP: mu^2 = mu_marg^2 + mu_corr^2.
"""

from __future__ import annotations

import math

from petsard.synthesizer.DPflow_common import MARGINAL_SHARE, DPflowBase, eps_for_mu, mu_for, release_copula_loading


class DPflowCopulaSynthesizer(DPflowBase):
    def _release(self, raw, spec):
        mu = mu_for(self.eps, self.delta)
        mu_m, mu_c = mu * math.sqrt(MARGINAL_SHARE), mu * math.sqrt(1 - MARGINAL_SHARE)
        self.marg = self._marginals(raw, mu_m, self._k(len(raw)))
        self.loading, sigma = release_copula_loading(self.marg.scores(raw), mu_c, len(raw), self.rng)
        self.privacy.update(achieved_epsilon=float(eps_for_mu(mu, self.delta)), accounting="Gaussian DP", moment_noise=round(sigma, 3))

    def _generate(self, n):
        d = self.loading.shape[1]
        return self.marg.invert(self.rng.standard_normal((n, d)) @ self.loading.T, self.rng)
