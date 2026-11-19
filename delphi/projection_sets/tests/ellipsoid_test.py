# Author: pstefanou12@
"""Tests for the confidence ellipsoid."""

import math

import torch as ch

from delphi.projection_sets import ellipsoid


def test_confidence_ellipsoid_projects_in_the_metric_norm():
    ch.manual_seed(0)
    factors = ch.randn(2, 4, 4, dtype=ch.float64)
    sigma_0, metric = factors @ factors.mT + 0.1 * ch.eye(4, dtype=ch.float64)
    anchor = ch.randn(4, 2, dtype=ch.float64)
    projection = ellipsoid.ConfidenceEllipsoid(anchor, sigma_0, metric)
    inside = anchor + 1e-3 * ch.randn(4, 2, dtype=ch.float64)
    assert projection.project(inside) is inside
    point = anchor + 10.0 * ch.randn(4, 2, dtype=ch.float64)
    projected = projection.project(point)
    offset = projected - anchor
    assert math.isclose(float((offset * (sigma_0 @ offset)).sum()), 1.0, rel_tol=1e-6)
    # Nearest in the metric's norm: its pull is a positive multiple of the
    # ellipsoid's normal there.
    pull, normal = metric @ (point - projected), sigma_0 @ offset
    scale = float((pull * normal).sum() / (normal * normal).sum())
    assert scale > 0.0
    assert ch.allclose(pull, scale * normal, atol=1e-8 * float(pull.abs().max()))
