# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from curobo._src.cost.cost_base import BaseCost
from curobo._src.state.state_joint import JointState
from curobo._src.util.logging import log_and_raise

if TYPE_CHECKING:
    from curobo._src.cost.cost_joint_bound_cfg import JointBoundCostCfg


class JointBoundCost(BaseCost):
    def __init__(self, config: JointBoundCostCfg):
        if not hasattr(config, "joint_indices"):
            log_and_raise("JointBoundCostCfg must be initialized from a transition model")
        super().__init__(config)

    def forward(self, joint_state: JointState) -> torch.Tensor:
        q = joint_state.position.index_select(-1, self.config.joint_indices)
        violations = torch.zeros_like(q)

        if self.config.lower_bound is not None:
            lower = self.config.lower_bound.view(1, 1, -1)
            violations = violations + torch.clamp(lower - q, min=0.0)
        if self.config.upper_bound is not None:
            upper = self.config.upper_bound.view(1, 1, -1)
            violations = violations + torch.clamp(q - upper, min=0.0)

        if self._weight.numel() == 1:
            return violations * self._weight.view(1, 1, 1)
        return violations * self._weight.view(1, 1, -1)
