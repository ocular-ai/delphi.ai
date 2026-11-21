# Author: pstefanou12@
"""Truncated binary regression: a latent-variable classifier seen through phi.

The latent ``z = w^T x + b + e`` labels the sample ``y = 1{z >= 0}``, and
the pair (x, y) is kept with probability ``phi(z)``. Probit and logistic
regression differ only in the noise e.
"""

import math

import torch as ch

from delphi.configs.stats.truncated import classification
from delphi.projection_sets import latent_interval, projection_set
from delphi.stats.truncated.truncated_classification import (
    TruncatedClassification,
)
from delphi.truncated import sampling


class TruncatedBinaryRegression(
    TruncatedClassification[classification.TruncatedBinaryRegressionConfig]
):
    """Truncated binary regression with a known noise law.

    A subclass sets ``noise`` and ``loss``, the criterion over that noise.
    ``phi`` sees latents shaped (n, m, 1); alpha is at most 1/2.
    """

    defaults = classification.TruncatedBinaryRegressionConfig
    noise: sampling.LatentNoise
    sampler: sampling.LatentSampler
    default_sampler = sampling.StratifiedLatentSampler

    @staticmethod
    def _check_inputs(X: ch.Tensor, y: ch.Tensor) -> None:  # pylint: disable=invalid-name
        """Raise unless y holds both labels 0 and 1, in one column."""
        TruncatedClassification._check_inputs(X, y)
        if y.size(1) != 1 or not ((y == 0) | (y == 1)).all():
            raise ValueError("expected labels in {0, 1} shaped (n, 1).")
        if y.min() == y.max():
            raise ValueError("y holds one label; both are needed.")

    def _predicted_label(self, latent: ch.Tensor) -> ch.Tensor:
        return (latent >= 0).long()

    def _untruncated_nll(self, latent: ch.Tensor, y: ch.Tensor) -> ch.Tensor:
        return -self.noise.log_cdf((2.0 * y.double() - 1.0) * latent).mean()

    def _label_mass(self, linear: ch.Tensor) -> ch.Tensor:
        """Return each latent mean's per-label mass, from fixed draws."""
        with ch.no_grad(), ch.random.fork_rng():
            ch.manual_seed(self.rand_seed)
            return self.sampler.label_moments(linear, self.noise, self.phi).mass

    def latent_interval(self, radius: float) -> tuple[float, float]:
        """Return the hull of the latent means keeping e^-radius per label.

        Each label keeps at most its noise mass, so the search covers the
        means at which both noise masses reach e^-radius.

        Args:
            radius: The set's ``log(1 / alpha')``.

        Returns:
            (lower, upper), one grid step wider than the feasible grid
            points; (inf, -inf) when none is feasible.
        """
        floor = math.exp(-radius)
        if floor >= 0.5:
            return math.inf, -math.inf
        reach = -float(self.noise.icdf(ch.tensor(floor, dtype=ch.float64)))
        grid = ch.linspace(
            -reach, reach, self.hyperparameters.latent_grid_size, dtype=ch.float64
        )
        feasible = (self._label_mass(grid[:, None]).min(1).values >= floor).nonzero()
        if feasible.numel() == 0:
            return math.inf, -math.inf
        step = float(grid[1] - grid[0])
        return (
            max(float(grid[feasible.min()]) - step, -reach),
            min(float(grid[feasible.max()]) + step, reach),
        )

    def _label_mass_set(
        self,
        X: ch.Tensor,  # pylint: disable=invalid-name
        anchor: ch.Tensor,
        radius: float,
    ) -> projection_set.ProjectionSet:
        return latent_interval.LatentIntervalSet(
            X, anchor, radius, self.latent_interval
        )

    def predict_proba(self, X: ch.Tensor) -> ch.Tensor:  # pylint: disable=invalid-name
        """Return ``P(y = 1 | x)`` without truncation, shaped (n, 1).

        Args:
            X: Covariates, shaped (n, d).
        """
        return self.noise.cdf(super().predict(X))

    def predict(self, X: ch.Tensor) -> ch.Tensor:  # pylint: disable=invalid-name
        """Return the more likely label without truncation, shaped (n, 1).

        Args:
            X: Covariates, shaped (n, d).
        """
        return self._predicted_label(super().predict(X)).float()
