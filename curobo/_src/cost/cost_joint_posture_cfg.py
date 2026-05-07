# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Type

import torch

from curobo._src.cost.cost_base_cfg import BaseCostCfg
from curobo._src.cost.cost_joint_posture import JointPostureCost
from curobo._src.transition.robot_state_transition import RobotStateTransition
from curobo._src.util.logging import log_and_raise


@dataclass
class JointPostureCostCfg(BaseCostCfg):
    class_type: Type[JointPostureCost] = JointPostureCost
    joint_names: Optional[List[str]] = None
    target: Optional[List[float]] = None
    deadband: float = 0.0

    def __post_init__(self):
        if self.joint_names is None or len(self.joint_names) == 0:
            log_and_raise("JointPostureCostCfg.joint_names must contain at least one joint")
        if self.target is None:
            log_and_raise("JointPostureCostCfg.target is required")
        if len(self.target) != len(self.joint_names):
            log_and_raise("JointPostureCostCfg.target must match len(joint_names)")
        super().__post_init__()

    def initialize_from_transition_model(self, transition_model: RobotStateTransition):
        missing = [n for n in self.joint_names if n not in transition_model.joint_names]
        if missing:
            log_and_raise(f"JointPostureCostCfg unknown joints: {missing}")

        self.joint_indices = torch.tensor(
            [transition_model.joint_names.index(n) for n in self.joint_names],
            device=self.device_cfg.device,
            dtype=torch.long,
        )
        self.target = self.device_cfg.to_device(self.target)
        if self.weight.numel() not in (1, len(self.joint_names)):
            log_and_raise("JointPostureCostCfg.weight must be scalar or match len(joint_names)")
