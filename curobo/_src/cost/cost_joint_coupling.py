# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from curobo._src.cost.cost_base import BaseCost
from curobo._src.state.state_joint import JointState
from curobo._src.util.logging import log_and_raise

if TYPE_CHECKING:
    from curobo._src.cost.cost_joint_coupling_cfg import JointCouplingCostCfg


class JointCouplingCost(BaseCost):
    def __init__(self, config: JointCouplingCostCfg):
        if not hasattr(config, "joint_indices"):
            log_and_raise(
                "JointCouplingCostCfg must be initialized from a transition model before use"
            )
        super().__init__(config)

    def forward(self, joint_state: JointState) -> torch.Tensor:
        q = joint_state.position.index_select(-1, self.config.joint_indices)
        values = torch.matmul(q, self.config.coefficients.transpose(0, 1))
        violations = torch.zeros_like(values)

        if self.config.target is not None:
            error = torch.abs(values - self.config.target.view(1, 1, -1))
            violations = violations + torch.clamp(error - self.config.tolerance, min=0.0)
        if self.config.lower_bound is not None:
            lower = self.config.lower_bound.view(1, 1, -1)
            violations = violations + torch.clamp(lower - values, min=0.0)
        if self.config.upper_bound is not None:
            upper = self.config.upper_bound.view(1, 1, -1)
            violations = violations + torch.clamp(values - upper, min=0.0)

        if self._weight.numel() == 1:
            return violations * self._weight.view(1, 1, 1)
        return violations * self._weight.view(1, 1, -1)
