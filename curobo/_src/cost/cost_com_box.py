# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from curobo._src.cost.cost_base import BaseCost
from curobo._src.robot.kinematics.kinematics_state import KinematicsState
from curobo._src.util.logging import log_and_raise

if TYPE_CHECKING:
    from curobo._src.cost.cost_com_box_cfg import ComBoxCostCfg


class ComBoxCost(BaseCost):
    def __init__(self, config: ComBoxCostCfg):
        if not hasattr(config, "axes_tensor"):
            log_and_raise("ComBoxCostCfg must be initialized from a transition model before use")
        super().__init__(config)

    def forward(self, kinematics_state: KinematicsState) -> torch.Tensor:
        if kinematics_state.robot_com is None:
            log_and_raise("ComBoxCost requires robot_com from kinematics")

        com = kinematics_state.robot_com[..., :3].index_select(-1, self.config.axes_tensor)
        violations = torch.zeros_like(com)

        if self.config.lower_bound is not None:
            lower = self.config.lower_bound.view(1, 1, -1)
            violations = violations + torch.clamp(lower - com, min=0.0)
        if self.config.upper_bound is not None:
            upper = self.config.upper_bound.view(1, 1, -1)
            violations = violations + torch.clamp(com - upper, min=0.0)

        if self._weight.numel() == 1:
            return violations * self._weight.view(1, 1, 1)
        return violations * self._weight.view(1, 1, -1)
