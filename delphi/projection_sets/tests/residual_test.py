# Author: pstefanou12@
"""Tests for the sets that bound a linear model's residuals."""

import pytest
import torch as ch

from delphi.projection_sets import residual


def _ball(seed: int = 0):
    """Return a wide design, responses and E_r at radius 1 around them."""
    generator = ch.Generator().manual_seed(seed)
    X = ch.randn(30, 60, dtype=ch.float64, generator=generator)
    y = ch.randn(30, 1, dtype=ch.float64, generator=generator)
    anchor = ch.linalg.lstsq(X, y).solution  # pylint: disable=not-callable
    ball = residual.ResidualBall(X, y, 1.0, anchor)
    return X, y, ball


def test_projection_lands_on_the_boundary_at_the_nearest_point():
    X, y, ball = _ball()
    outside = 3.0 * ch.randn(60, 1, dtype=ch.float64)
    assert not ball.contains(outside)
    projected = ball.project(outside)
    residual = X @ projected - y
    ch.testing.assert_close(float(residual.norm()), 30**0.5, rtol=1e-6, atol=0.0)
    # Nearest: the step back is along the constraint's gradient X^T r.
    step = outside - projected
    gradient = X.T @ residual
    cosine = float((step.T @ gradient) / (step.norm() * gradient.norm()))
    assert cosine == pytest.approx(1.0, abs=1e-8)


def test_projection_keeps_points_inside():
    X, y, ball = _ball(seed=1)
    inside = ch.linalg.lstsq(X, y).solution  # pylint: disable=not-callable
    inside = inside + 1e-3 * ch.randn(60, 1, dtype=ch.float64)
    assert ball.contains(inside)
    assert ball.project(inside) is inside


def test_an_empty_set_cannot_be_projected_onto():
    generator = ch.Generator().manual_seed(2)
    X = ch.randn(200, 3, dtype=ch.float64, generator=generator)
    y = 5.0 * ch.randn(200, 1, dtype=ch.float64, generator=generator)
    ball = residual.ResidualBall(X, y, 1.0, ch.zeros(3, 1))
    with pytest.raises(ValueError, match="empty"):
        ball.project(ch.ones(3, 1, dtype=ch.float64))
