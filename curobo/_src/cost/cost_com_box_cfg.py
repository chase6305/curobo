# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Type

from curobo._src.cost.cost_base_cfg import BaseCostCfg
from curobo._src.cost.cost_com_box import ComBoxCost
from curobo._src.transition.robot_state_transition import RobotStateTransition
from curobo._src.util.logging import log_and_raise


@dataclass
class ComBoxCostCfg(BaseCostCfg):
    class_type: Type[ComBoxCost] = ComBoxCost
    lower_bound: Optional[List[float]] = None
    upper_bound: Optional[List[float]] = None
    axes: Optional[List[int]] = None

    def __post_init__(self):
        if self.lower_bound is None and self.upper_bound is None:
            log_and_raise("ComBoxCostCfg requires lower_bound or upper_bound")
        if self.axes is None:
            size = len(self.lower_bound if self.lower_bound is not None else self.upper_bound)
            self.axes = list(range(size))
        if any(axis < 0 or axis > 2 for axis in self.axes):
            log_and_raise("ComBoxCostCfg.axes must contain only 0, 1, or 2")
        super().__post_init__()

    def initialize_from_transition_model(self, transition_model: RobotStateTransition):
        size = len(self.axes)
        self.axes_tensor = self.device_cfg.to_device(self.axes).long()
        self.lower_bound = self._optional_vector(self.lower_bound, size, "lower_bound")
        self.upper_bound = self._optional_vector(self.upper_bound, size, "upper_bound")

    def _optional_vector(self, value, size: int, name: str):
        if value is None:
            return None
        tensor = self.device_cfg.to_device(value)
        if tensor.shape != (size,):
            log_and_raise(f"ComBoxCostCfg.{name} must have shape ({size},)")
        return tensor
