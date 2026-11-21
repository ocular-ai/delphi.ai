# Author: pstefanou12@
"""Sets that bound a linear model's residuals on its training data."""

import math
from typing import Optional, cast

import torch as ch

from delphi.projection_sets.projection_set import ProjectionSet


class ResidualBall(ProjectionSet):
    """The set ``E_r = {w : ||X w - y||_2 <= r sqrt(n)}``.

    A projection returns the Euclidean nearest point, ``(I + mu X^T X)^-1
    (w + mu X^T y)`` for the mu that lands it on the boundary: with
    ``X = U diag(s) V^T``, the residual there is
    ``||(s V^T w - U^T y) / (1 + mu s^2)||^2`` plus the part of y outside
    U's span, which is decreasing in mu, so mu is bisected.

    Attributes:
        radius: The set's r.
    """

    # Slack on the boundary, so float32 rounding of a projected point does
    # not leave it outside.
    _SLACK = 1e-5

    def __init__(
        self,
        X: ch.Tensor,  # pylint: disable=invalid-name
        y: ch.Tensor,
        radius: float,
        anchor: ch.Tensor,
        bisection_steps: int = 100,
    ):
        """Build E_r from the training data.

        Args:
            X: Design matrix, shaped (n, p).
            y: Responses, shaped (n, 1).
            radius: The set's r.
            anchor: A weight in the set, shaped (p, 1).
            bisection_steps: Halvings of the interval mu is searched in.
        """
        super().__init__(anchor, radius, bisection_steps)
        self._X, self._y = X.double(), y.double()  # pylint: disable=invalid-name
        u, self._spectrum, self._basis = ch.linalg.svd(self._X, full_matrices=False)  # pylint: disable=not-callable
        kept = self._spectrum > self._spectrum.max() * 1e-12
        u, self._spectrum = u[:, kept], self._spectrum[kept]
        self._basis = self._basis[kept]
        self._projected_y = u.T @ self._y
        self._outside = float((self._y.pow(2).sum() - self._projected_y.pow(2).sum()))

    def _bound(self) -> float:
        return cast(float, self.radius) * math.sqrt(self._X.size(0))

    def contains(self, weight: ch.Tensor) -> bool:
        residual = float((self._X @ weight.detach().double() - self._y).norm())
        return residual <= self._bound() * (1.0 + self._SLACK)

    def project(self, weight: ch.Tensor) -> ch.Tensor:
        """Return the Euclidean nearest point of the set to weight.

        Raises:
            ValueError: If the set is empty, as when y lies further than
                the bound from X's column span.
        """
        if self.contains(weight):
            return weight
        bound = self._bound()
        if self._outside > bound**2:
            raise ValueError(
                f"E_r is empty: no weight has a residual norm within {bound:.4f}."
            )
        point = weight.detach().double()
        offset = self._spectrum[:, None] * (self._basis @ point) - self._projected_y

        def residual(mu: float) -> float:
            shrunk = offset / (1.0 + mu * self._spectrum[:, None].pow(2))
            return float(shrunk.pow(2).sum()) + max(self._outside, 0.0)

        low, high = 0.0, 1.0
        while residual(high) > bound**2:
            high *= 2.0
        for _ in range(self.bisection_steps):
            middle = (low + high) / 2.0
            if residual(middle) > bound**2:
                low = middle
            else:
                high = middle
        spectrum = self._spectrum[:, None]
        step = -high * spectrum * offset / (1.0 + high * spectrum.pow(2))
        return (point + self._basis.T @ step).to(weight.dtype)


class ResidualSpectralSet(ProjectionSet):
    """The projection set D_{r,B} of a known noise variance.

    ``{w : sum_i ((y_i - w^T x_i) / sigma)^2 x_i x_i^T <= r sum_i x_i x_i^T,
    ||w|| <= B}``, in the Loewner order. In coordinates where
    ``sum_i x_i x_i^T`` is the identity the first constraint bounds the
    largest eigenvalue of ``sum_i rho_i^2 u_i u_i^T`` by r. ``rho_i`` is
    linear in ``v = (-w, 1)``, so that matrix is ``sum_pq v_p v_q Q_pq`` for
    a tensor Q summed over the data once, and a membership test costs
    O(d^4) instead of O(n d^2) whenever d^2 is below n and Q fits in memory.

    Attributes:
        radius: The set's r, bounding the residual-weighted covariance
            relative to the covariance.
        bound: Bound B on the weight's norm, or None for no ball.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        X: ch.Tensor,  # pylint: disable=invalid-name
        y: ch.Tensor,
        noise_var: float,
        radius: float,
        anchor: ch.Tensor,
        bound: Optional[float] = None,
        bisection_steps: int = 40,
        max_quadratic_entries: int = 2**24,
    ):
        """Build D_{r,B} from the training data.

        Args:
            X: Design matrix, shaped (n, d).
            y: Responses, shaped (n, 1).
            noise_var: Known noise variance sigma^2.
            radius: The set's r.
            anchor: Weight of shape (d, 1) to retract towards.
            bound: The set's B, or None for no ball.
            bisection_steps: As ``ProjectionSet``.
            max_quadratic_entries: Largest precomputed quadratic form, in
                float64 entries; 2^24 is 128 MB. A larger one is skipped for
                the O(n d^2) test.
        """
        super().__init__(anchor, radius, bisection_steps)
        self.bound = bound
        scale = math.sqrt(noise_var)
        chol = ch.linalg.cholesky(X.T @ X)  # pylint: disable=not-callable
        whitened = ch.linalg.solve_triangular(chol, X.T, upper=False).T  # pylint: disable=not-callable
        joint = ch.cat([-X, y], dim=1) / scale
        self._direct: Optional[tuple[ch.Tensor, ch.Tensor]] = None
        self._quadratic: Optional[ch.Tensor] = None
        entries = ((X.size(1) + 1) * X.size(1)) ** 2
        if (X.size(1) + 1) ** 2 <= X.size(0) and entries <= max_quadratic_entries:
            whitened, joint = whitened.double(), joint.double()
            self._quadratic = ch.einsum(
                "np,nq,na,nb->pqab", joint, joint, whitened, whitened
            )
        else:
            self._direct = (joint, whitened)

    def contains(self, weight: ch.Tensor) -> bool:
        if self.bound is not None and float(weight.norm()) > self.bound:
            return False
        v = ch.cat([weight.flatten(), ch.ones(1)])
        if self._quadratic is not None:
            v = v.double()
            weighted = ch.einsum("p,q,pqab->ab", v, v, self._quadratic)
        else:
            joint, whitened = cast(tuple[ch.Tensor, ch.Tensor], self._direct)
            weighted = (whitened * (joint @ v)[:, None].pow(2)).T @ whitened
        return float(ch.linalg.eigvalsh(weighted)[-1]) <= cast(float, self.radius)  # pylint: disable=not-callable
