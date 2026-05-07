# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Type

import torch

from curobo._src.cost.cost_base_cfg import BaseCostCfg
from curobo._src.cost.cost_joint_coupling import JointCouplingCost
from curobo._src.transition.robot_state_transition import RobotStateTransition
from curobo._src.util.logging import log_and_raise


@dataclass
class JointCouplingCostCfg(BaseCostCfg):
    class_type: Type[JointCouplingCost] = JointCouplingCost
    coefficients: Optional[List[List[float]]] = None
    joint_names: Optional[List[str]] = None
    target: Optional[List[float]] = None
    lower_bound: Optional[List[float]] = None
    upper_bound: Optional[List[float]] = None
    tolerance: float = 0.0

    def __post_init__(self):
        if self.coefficients is None or len(self.coefficients) == 0:
            log_and_raise("JointCouplingCostCfg.coefficients must contain at least one row")
        super().__post_init__()

    def initialize_from_transition_model(self, transition_model: RobotStateTransition):
        device_cfg = self.device_cfg
        dof = transition_model.action_dim
        coeff = device_cfg.to_device(self.coefficients)
        if coeff.ndim != 2:
            log_and_raise("JointCouplingCostCfg.coefficients must be a 2D list")

        if self.joint_names is None:
            if coeff.shape[1] != dof:
                log_and_raise(
                    "JointCouplingCostCfg.coefficients width must match robot dof when "
                    "joint_names is not provided"
                )
            joint_indices = torch.arange(dof, device=device_cfg.device, dtype=torch.long)
        else:
            if coeff.shape[1] != len(self.joint_names):
                log_and_raise(
                    "JointCouplingCostCfg.coefficients width must match len(joint_names)"
                )
            missing = [n for n in self.joint_names if n not in transition_model.joint_names]
            if missing:
                log_and_raise(f"JointCouplingCostCfg unknown joints: {missing}")
            joint_indices = torch.tensor(
                [transition_model.joint_names.index(n) for n in self.joint_names],
                device=device_cfg.device,
                dtype=torch.long,
            )

        self.coefficients = coeff
        self.joint_indices = joint_indices
        self.target = self._optional_vector(self.target, coeff.shape[0], "target")
        self.lower_bound = self._optional_vector(self.lower_bound, coeff.shape[0], "lower_bound")
        self.upper_bound = self._optional_vector(self.upper_bound, coeff.shape[0], "upper_bound")

    def _optional_vector(self, value, size: int, name: str):
        if value is None:
            return None
        tensor = self.device_cfg.to_device(value)
        if tensor.shape != (size,):
            log_and_raise(f"JointCouplingCostCfg.{name} must have shape ({size},)")
        return tensor
