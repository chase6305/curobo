# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from curobo._src.cost.cost_base import BaseCost
from curobo._src.types.tool_pose import ToolPose
from curobo._src.util.logging import log_and_raise

if TYPE_CHECKING:
    from curobo._src.cost.cost_link_pose_relation_cfg import LinkPoseRelationCostCfg


class LinkPoseRelationCost(BaseCost):
    def __init__(self, config: LinkPoseRelationCostCfg):
        if not hasattr(config, "target_indices"):
            log_and_raise("LinkPoseRelationCostCfg must be initialized from a transition model")
        super().__init__(config)

    def forward(self, tool_poses: ToolPose) -> torch.Tensor:
        target_pos = tool_poses.position.index_select(2, self.config.target_indices)
        reference_pos = tool_poses.position.index_select(2, self.config.reference_indices)

        axes = self.config.axes_tensor.view(1, 1, -1, 1)
        axes = axes.expand(target_pos.shape[0], target_pos.shape[1], -1, 1)
        target_values = torch.gather(target_pos, -1, axes).squeeze(-1)
        reference_values = torch.gather(reference_pos, -1, axes).squeeze(-1)
        relation = target_values - reference_values

        violations = torch.zeros_like(relation)
        if self.config.lower_bound is not None:
            lower = self.config.lower_bound.view(1, 1, -1)
            violations = violations + torch.clamp(lower - relation, min=0.0)
        if self.config.upper_bound is not None:
            upper = self.config.upper_bound.view(1, 1, -1)
            violations = violations + torch.clamp(relation - upper, min=0.0)

        cost = violations * violations
        if self._weight.numel() == 1:
            return cost * self._weight.view(1, 1, 1)
        return cost * self._weight.view(1, 1, -1)
