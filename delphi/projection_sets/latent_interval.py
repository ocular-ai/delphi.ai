# Author: pstefanou12@
"""Weights whose latent means all lie in one interval, as R_alpha does."""

from collections.abc import Callable
import math
from typing import Optional

import torch as ch

from delphi.projection_sets.projection_set import ProjectionSet

# Fraction of the segment a projection stops short of the boundary.
_INTERIOR = 1e-6


class LatentIntervalSet(ProjectionSet):
    """Weights whose latent means ``x_i^T w`` all lie in one interval.

    The interval is a function of the radius, recomputed when it changes;
    R_alpha, the weights at which every sample keeps mass alpha on each
    label, takes this form. The projection is exact along the segment to
    the anchor, so it skips the bisection.

    Attributes:
        lower: The interval's lower end at the current radius.
        upper: The interval's upper end at the current radius.
    """

    def __init__(
        self,
        X: ch.Tensor,  # pylint: disable=invalid-name
        anchor: ch.Tensor,
        radius: float,
        interval: Callable[[float], tuple[float, float]],
    ):
        """Build the set over the training covariates.

        Args:
            X: Design matrix, shaped (n, d).
            anchor: Weight of shape (d, 1) to retract towards.
            radius: The set's size, which interval maps to its bounds.
            interval: Maps a radius to the (lower, upper) bounds on every
                latent mean; lower > upper describes the empty set.
        """
        self._design = X.double()
        self._interval = interval
        self.lower, self.upper = interval(radius)
        super().__init__(anchor, radius)

    @property
    def radius(self) -> Optional[float]:
        """The set's size; setting it recomputes the interval."""
        return self._radius

    @radius.setter
    def radius(self, value: Optional[float]) -> None:
        self._radius = value
        if value is not None:
            self.lower, self.upper = self._interval(value)

    def contains(self, weight: ch.Tensor) -> bool:
        latent = self._design @ weight.detach().double().reshape(-1, 1)
        return bool(((latent >= self.lower) & (latent <= self.upper)).all())

    def project(self, weight: ch.Tensor) -> ch.Tensor:
        if self.contains(weight):
            return weight
        start = self._design @ self.anchor.double().reshape(-1, 1)
        step = self._design @ (weight.detach() - self.anchor).double().reshape(-1, 1)
        bound = ch.where(step > 0, self.upper, self.lower)
        ratio = ch.where(step != 0, (bound - start) / step, math.inf)
        # Inside the boundary, so rounding cannot leave the result outside.
        reach = (ratio.min().clamp(0.0, 1.0) * (1.0 - _INTERIOR)).item()
        return self.anchor + reach * (weight.detach() - self.anchor)
