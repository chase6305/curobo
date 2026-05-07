# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from curobo._src.cost.cost_base import BaseCost
from curobo._src.state.state_joint import JointState
from curobo._src.util.logging import log_and_raise

if TYPE_CHECKING:
    from curobo._src.cost.cost_joint_posture_cfg import JointPostureCostCfg


class JointPostureCost(BaseCost):
    def __init__(self, config: JointPostureCostCfg):
        if not hasattr(config, "joint_indices"):
            log_and_raise("JointPostureCostCfg must be initialized from a transition model")
        super().__init__(config)

    def forward(self, joint_state: JointState) -> torch.Tensor:
        q = joint_state.position.index_select(-1, self.config.joint_indices)
        error = torch.abs(q - self.config.target.view(1, 1, -1))
        if self.config.deadband > 0.0:
            error = torch.clamp(error - self.config.deadband, min=0.0)
        cost = error * error

        if self._weight.numel() == 1:
            return cost * self._weight.view(1, 1, 1)
        return cost * self._weight.view(1, 1, -1)
