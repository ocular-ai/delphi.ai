# Author: pstefanou12@
"""The sublevel set of a loss, which the exponential families project onto."""

from collections.abc import Callable

import torch as ch

from delphi.projection_sets.projection_set import ProjectionSet


class SublevelSet(ProjectionSet):
    """The weights whose loss is at most ``level + radius``.

    The exponential families keep the untruncated NLL within a budget of its
    value at the start, per Karatapanis et al. (2025), Algorithm 2. A
    sublevel set of a general loss has no closed-form projection, so it
    projects with ``ProjectionSet.project``'s bisection towards the anchor,
    which needs only ``contains``.

    Attributes:
        loss: Maps a weight to the loss the set bounds.
        level: The loss at the anchor, which the radius is a budget above.
    """

    def __init__(
        self,
        anchor: ch.Tensor,
        loss: Callable[[ch.Tensor], float],
        radius: float,
        bisection_steps: int = 50,
    ):
        """Initialize the set.

        Args:
            anchor: The weight the budget is measured from.
            loss: Maps a weight to the loss the set bounds.
            radius: The budget above the anchor's loss.
            bisection_steps: As the base class; 50 leaves 2^-50 of the
                segment.
        """
        super().__init__(anchor, radius, bisection_steps)
        self.loss = loss
        self.level = loss(self.anchor)

    @property
    def threshold(self) -> float:
        """The largest loss inside the set."""
        return self.level + float(self.radius or 0.0)

    def contains(self, weight: ch.Tensor) -> bool:
        return self.loss(weight) <= self.threshold
