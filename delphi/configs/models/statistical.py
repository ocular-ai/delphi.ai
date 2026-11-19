# Author: pstefanou12@
"""Configuration models for statistical models and their Newton steps."""

import pydantic

from delphi.configs.base import DelphiConfig


class NewtonRefinementConfig(pydantic.BaseModel):
    """Configuration of the Newton steps that refine a fitted estimate.

    Setting ``newton_steps`` enables them. From the fit's averaged iterate,
    each step is ``theta - H^-1 grad`` on the same sample, with ``H`` the
    model's ``hessian`` at the current iterate, which moves the estimate to
    the sample's MLE: the k-step Newton estimator, not a further fit on new
    data. See ``StatisticalModel.refine``.

    Attributes:
        newton_steps: Newton steps; 0 skips the stage.
        newton_avg_burn_in: Fraction of the steps left out of the average,
            which cancels the Monte Carlo noise in a sampled gradient.
        newton_damped: Shrink each step by ``1 / (1 + delta)``, for delta
            the Newton decrement ``sqrt(grad^T H^-1 grad)``: Nesterov's
            damped Newton, which converges from a start where full steps
            overshoot.
    """

    newton_steps: int = pydantic.Field(default=0, ge=0)
    newton_avg_burn_in: float = pydantic.Field(default=0.5, ge=0.0, lt=1.0)
    newton_damped: bool = False


class StatisticalModelConfig(DelphiConfig, NewtonRefinementConfig):
    """Base config of the statistical models, which may refine their fit."""

    def refinement_overlay(self) -> "StatisticalModelConfig":
        """Return this config as the refinement's: unit-rate, unscheduled SGD.

        Returns:
            A copy that takes ``newton_steps`` Newton steps.
        """
        return self.model_copy(
            update={
                "optimizer": "sgd",
                "lr": 1.0,
                "sgd_momentum": 0.0,
                "weight_decay": 0.0,
                "scheduler": None,
                "constant": True,
                "max_grad_norm": None,
                "epochs": None,
                "iterations": self.newton_steps,
                "avg_burn_in": self.newton_avg_burn_in,
                "max_phases": 1,
                "warm_start_iterations": None,
                "warm_start_epochs": None,
            }
        )
