# Author: pstefanou12@
"""The set of a linear regression's natural parameters with learned variance."""

import math
from typing import Optional, cast

import torch as ch

from delphi.projection_sets.projection_set import ProjectionSet


class UnknownVarianceSet(ProjectionSet):
    """The projection set D of a learned noise variance.

    It bounds the noise variance relative to the start's residual variance
    ``sigma_0^2``, and the weight by B, which in ``(v, lambda) =
    (w / sigma^2, 1 / sigma^2)`` is the convex set
    ``{a^2 / (96 sigma_0^2) <= lambda <= 8 (5 + 2 log(1 / a)) / sigma_0^2,
    ||v|| <= B lambda}``. Its radius is ``1 / a``, for a the survival mass.
    Each output's column ``(v, lambda)`` lies in its own copy, and the
    projection is Euclidean.

    Attributes:
        residual_var: Each output's ``sigma_0^2``, shaped (k,).
        radius: The set's ``1 / a``, at least 1.
        bound: Bound B on the weight's norm, or None for no ball.
    """

    def __init__(
        self,
        residual_var: ch.Tensor,
        radius: float,
        anchor: ch.Tensor,
        bound: Optional[float] = None,
    ):
        """Build D around the start's residual variance.

        Args:
            residual_var: Each output's sigma_0^2, shaped (k,).
            radius: The set's 1 / a.
            anchor: The start ``(v, lambda)``, shaped (d + 1, k) with lambda
                as the last row.
            bound: The set's B, or None for no ball.

        Raises:
            ValueError: If radius is below 1.
        """
        if radius < 1.0:
            raise ValueError(f"radius is 1 / alpha, at least 1; got {radius}.")
        super().__init__(anchor, radius)
        self.residual_var = residual_var
        self.bound = bound

    def precision_bounds(self) -> tuple[ch.Tensor, ch.Tensor]:
        """Return the lower and upper bounds on lambda, each shaped (k,)."""
        radius = cast(float, self.radius)
        lower = 1.0 / (96.0 * radius**2 * self.residual_var)
        upper = 8.0 * (5.0 + 2.0 * math.log(radius)) / self.residual_var
        return lower, upper

    def contains(self, weight: ch.Tensor) -> bool:
        lower, upper = self.precision_bounds()
        precision = weight[-1]
        if bool((precision < lower).any() or (precision > upper).any()):
            return False
        if self.bound is None:
            return True
        return bool((weight[:-1].norm(dim=0) <= self.bound * precision).all())

    def project(self, weight: ch.Tensor) -> ch.Tensor:
        """Return the nearest point of the set to weight."""
        if self.contains(weight):
            return weight
        lower, upper = self.precision_bounds()
        precision = ch.minimum(ch.maximum(weight[-1], lower), upper)
        if self.bound is None:
            return ch.cat([weight[:-1], precision[None]])
        projected = [
            self._project_column(weight[:, a], float(lower[a]), float(upper[a]))
            for a in range(weight.size(1))
        ]
        return ch.stack(projected, dim=1)

    def _project_column(
        self, column: ch.Tensor, lower: float, upper: float
    ) -> ch.Tensor:
        """Return the nearest point to one output's ``(v, lambda)``.

        The nearest point keeps v's direction, so the problem is planar in
        ``(||v||, lambda)``, over the trapezoid with corners ``(0, lower)``,
        ``(B lower, lower)``, ``(B upper, upper)`` and ``(0, upper)``; a
        point outside it is nearest to one of its edges.
        """
        bound = cast(float, self.bound)
        norm, precision = float(column[:-1].norm()), float(column[-1])
        if lower <= precision <= upper and norm <= bound * precision:
            return column
        point = ch.tensor([norm, precision], dtype=ch.float64)
        corners = ch.tensor(
            [
                [0.0, lower],
                [bound * lower, lower],
                [bound * upper, upper],
                [0.0, upper],
            ],
            dtype=ch.float64,
        )
        candidates = []
        for start, end in zip(corners, corners.roll(-1, 0), strict=True):
            edge = end - start
            share = float((point - start) @ edge / (edge @ edge))
            candidates.append(start + min(max(share, 0.0), 1.0) * edge)
        nearest = min(
            candidates, key=lambda candidate: float((candidate - point).norm())
        )
        direction = column[:-1] / norm if norm > 0.0 else ch.zeros_like(column[:-1])
        return ch.cat(
            [float(nearest[0]) * direction, column.new_tensor([float(nearest[1])])]
        )
