# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Type

import torch

from curobo._src.cost.cost_base_cfg import BaseCostCfg
from curobo._src.cost.cost_link_pose_relation import LinkPoseRelationCost
from curobo._src.transition.robot_state_transition import RobotStateTransition
from curobo._src.util.logging import log_and_raise


@dataclass
class LinkPoseRelationCostCfg(BaseCostCfg):
    class_type: Type[LinkPoseRelationCost] = LinkPoseRelationCost
    target_link_names: Optional[List[str]] = None
    reference_link_names: Optional[List[str]] = None
    axes: Optional[List[int]] = None
    lower_bound: Optional[List[float]] = None
    upper_bound: Optional[List[float]] = None

    def __post_init__(self):
        if self.target_link_names is None or self.reference_link_names is None:
            log_and_raise("LinkPoseRelationCostCfg requires target and reference links")
        if len(self.target_link_names) != len(self.reference_link_names):
            log_and_raise("target_link_names must match reference_link_names length")
        if self.lower_bound is None and self.upper_bound is None:
            log_and_raise("LinkPoseRelationCostCfg requires lower_bound or upper_bound")
        if self.axes is None:
            self.axes = [2] * len(self.target_link_names)
        if len(self.axes) != len(self.target_link_names):
            log_and_raise("axes must match target_link_names length")
        if any(axis < 0 or axis > 2 for axis in self.axes):
            log_and_raise("axes must contain only 0, 1, or 2")
        super().__post_init__()

    def initialize_from_transition_model(self, transition_model: RobotStateTransition):
        tool_frames = transition_model.robot_model.tool_frames
        missing = [
            name for name in self.target_link_names + self.reference_link_names
            if name not in tool_frames
        ]
        if missing:
            log_and_raise(f"LinkPoseRelationCostCfg unknown tool frames: {missing}")

        self.target_indices = torch.tensor(
            [tool_frames.index(name) for name in self.target_link_names],
            device=self.device_cfg.device,
            dtype=torch.long,
        )
        self.reference_indices = torch.tensor(
            [tool_frames.index(name) for name in self.reference_link_names],
            device=self.device_cfg.device,
            dtype=torch.long,
        )
        self.axes_tensor = torch.tensor(
            self.axes,
            device=self.device_cfg.device,
            dtype=torch.long,
        )
        size = len(self.target_link_names)
        self.lower_bound = self._optional_vector(self.lower_bound, size, "lower_bound")
        self.upper_bound = self._optional_vector(self.upper_bound, size, "upper_bound")
        if self.weight.numel() not in (1, size):
            log_and_raise("LinkPoseRelationCostCfg.weight must be scalar or match relation count")

    def _optional_vector(self, value, size: int, name: str):
        if value is None:
            return None
        tensor = self.device_cfg.to_device(value)
        if tensor.shape != (size,):
            log_and_raise(f"LinkPoseRelationCostCfg.{name} must have shape ({size},)")
        return tensor
