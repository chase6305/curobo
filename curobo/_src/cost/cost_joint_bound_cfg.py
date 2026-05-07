# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Type

import torch

from curobo._src.cost.cost_base_cfg import BaseCostCfg
from curobo._src.cost.cost_joint_bound import JointBoundCost
from curobo._src.transition.robot_state_transition import RobotStateTransition
from curobo._src.util.logging import log_and_raise


@dataclass
class JointBoundCostCfg(BaseCostCfg):
    class_type: Type[JointBoundCost] = JointBoundCost
    joint_names: Optional[List[str]] = None
    lower_bound: Optional[List[float]] = None
    upper_bound: Optional[List[float]] = None

    def __post_init__(self):
        if self.joint_names is None or len(self.joint_names) == 0:
            log_and_raise("JointBoundCostCfg.joint_names must contain at least one joint")
        if self.lower_bound is None and self.upper_bound is None:
            log_and_raise("JointBoundCostCfg requires lower_bound or upper_bound")
        super().__post_init__()

    def initialize_from_transition_model(self, transition_model: RobotStateTransition):
        missing = [n for n in self.joint_names if n not in transition_model.joint_names]
        if missing:
            log_and_raise(f"JointBoundCostCfg unknown joints: {missing}")

        self.joint_indices = torch.tensor(
            [transition_model.joint_names.index(n) for n in self.joint_names],
            device=self.device_cfg.device,
            dtype=torch.long,
        )
        size = len(self.joint_names)
        self.lower_bound = self._optional_vector(self.lower_bound, size, "lower_bound")
        self.upper_bound = self._optional_vector(self.upper_bound, size, "upper_bound")
        if self.weight.numel() not in (1, size):
            log_and_raise("JointBoundCostCfg.weight must be scalar or match len(joint_names)")

    def _optional_vector(self, value, size: int, name: str):
        if value is None:
            return None
        tensor = self.device_cfg.to_device(value)
        if tensor.shape != (size,):
            log_and_raise(f"JointBoundCostCfg.{name} must have shape ({size},)")
        return tensor
