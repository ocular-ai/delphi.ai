# Author: pstefanou12@
"""A confidence ellipsoid, projected onto in the norm of a metric."""

from typing import cast

import torch as ch

from delphi.projection_sets.projection_set import ProjectionSet


class ConfidenceEllipsoid(ProjectionSet):
    """The ellipsoid ``{W : tr((W - W_0)^T Sigma_0 (W - W_0)) <= r^2}``.

    A projection returns the nearest point in the norm
    ``tr(D^T Sigma D)`` of a metric Sigma, not the Euclidean one: with
    ``Sigma_0 = C C^T`` and ``C^-1 Sigma C^-T = Q diag(e) Q^T``, the
    coordinates ``Q^T C^T D`` turn the set into a ball and the metric into
    ``diag(e)``, and the nearest point shrinks coordinate j by
    ``e_j / (e_j + mu)``, for the mu that lands it on the sphere.

    Attributes:
        sigma_0: The ellipsoid's shape, shaped (p, p).
        metric: Sigma, shaped (p, p); its owner may grow it in place.
        radius: The set's r.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        anchor: ch.Tensor,
        sigma_0: ch.Tensor,
        metric: ch.Tensor,
        radius: float = 1.0,
        bisection_steps: int = 60,
    ):
        """Build the ellipsoid around anchor.

        Args:
            anchor: Its center W_0, shaped (p, k).
            sigma_0: Its shape, positive definite, shaped (p, p).
            metric: The projection's Sigma, positive definite, shaped (p, p).
            radius: Its r.
            bisection_steps: Halvings of the interval mu is searched in.
        """
        super().__init__(anchor, radius, bisection_steps)
        self.sigma_0 = sigma_0
        self.metric = metric

    def contains(self, weight: ch.Tensor) -> bool:
        offset = (weight.detach() - self.anchor).double()
        spread = float((offset * (self.sigma_0.double() @ offset)).sum())
        return spread <= cast(float, self.radius) ** 2

    def project(self, weight: ch.Tensor) -> ch.Tensor:
        """Return the nearest point of the set to weight in the metric's norm."""
        if self.contains(weight):
            return weight
        chol = ch.linalg.cholesky(self.sigma_0.double())  # pylint: disable=not-callable
        half = ch.linalg.solve_triangular(chol, self.metric.double(), upper=False)  # pylint: disable=not-callable
        whitened = ch.linalg.solve_triangular(chol, half.T, upper=False)  # pylint: disable=not-callable
        spectrum, basis = ch.linalg.eigh((whitened + whitened.T) / 2.0)  # pylint: disable=not-callable
        target = basis.T @ chol.T @ (weight.detach() - self.anchor).double()
        mass = (spectrum[:, None] * target).pow(2).sum(1)
        radius = cast(float, self.radius)
        # At mu = high every coordinate already lies within the sphere.
        low, high = 0.0, float(mass.sum().sqrt()) / radius
        for _ in range(self.bisection_steps):
            middle = (low + high) / 2.0
            if float((mass / (spectrum + middle).pow(2)).sum()) > radius**2:
                low = middle
            else:
                high = middle
        shrunk = (spectrum / (spectrum + high))[:, None] * target
        offset = ch.linalg.solve_triangular(chol.T, basis @ shrunk, upper=True)  # pylint: disable=not-callable
        return (self.anchor + offset).to(weight.dtype)
