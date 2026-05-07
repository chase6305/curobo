# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration dataclass for the robot cost manager.

Holds sub-configs for each cost term (tool-pose, collision, self-collision, cspace,
cspace-dist) and provides the factory method :meth:`RobotCostManagerCfg.from_dict`
that constructs all cost configs from a flat dictionary.
"""
from __future__ import annotations

# Standard Library
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional

# CuRobo
from curobo._src.cost.cost_cspace_cfg import CSpaceCostCfg
from curobo._src.cost.cost_cspace_dist_cfg import CSpaceDistCostCfg
from curobo._src.cost.cost_com_box_cfg import ComBoxCostCfg
from curobo._src.cost.cost_joint_bound_cfg import JointBoundCostCfg
from curobo._src.cost.cost_joint_coupling_cfg import JointCouplingCostCfg
from curobo._src.cost.cost_joint_posture_cfg import JointPostureCostCfg
from curobo._src.cost.cost_link_pose_relation_cfg import LinkPoseRelationCostCfg
from curobo._src.cost.cost_scene_collision_cfg import SceneCollisionCostCfg
from curobo._src.cost.cost_self_collision_cfg import SelfCollisionCostCfg
from curobo._src.cost.cost_tool_pose_cfg import ToolPoseCostCfg
from curobo._src.types.device_cfg import DeviceCfg

if TYPE_CHECKING:
    # CuRobo
    from curobo._src.geom.collision.collision_scene import SceneCollision


@dataclass
class RobotCostManagerCfg:
    """Configuration for :class:`RobotCostManager`.

    Holds optional configs for each cost component (self-collision, scene
    collision, c-space, c-space distance, tool pose) used during robot
    motion planning.
    """

    #: Concrete :class:`RobotCostManager` class. Set automatically in
    #: ``__post_init__``.
    class_type: type = None

    # Collision
    #: Self-collision cost configuration. None disables self-collision
    #: penalties.
    self_collision_cfg: Optional[SelfCollisionCostCfg] = None
    #: Scene (environment) collision cost configuration. None disables
    #: scene collision penalties.
    scene_collision_cfg: Optional[SceneCollisionCostCfg] = None

    # Joint space
    #: Configuration-space (joint limit, velocity, acceleration)
    #: regularization cost. None disables c-space penalties.
    cspace_cfg: Optional[CSpaceCostCfg] = None
    #: C-space distance cost relative to the start configuration. Used
    #: to penalize deviation from the initial joint state. None disables.
    start_cspace_dist_cfg: Optional[CSpaceDistCostCfg] = None
    #: C-space distance cost relative to the target configuration. Used
    #: to penalize deviation from the goal joint state. None disables.
    target_cspace_dist_cfg: Optional[CSpaceDistCostCfg] = None

    # Pose tracking
    #: Tool (end-effector) pose tracking cost configuration. None
    #: disables pose-based objectives.
    tool_pose_cfg: Optional[ToolPoseCostCfg] = None

    #: Linear joint-coupling cost/constraint for robot-specific IK modes.
    joint_coupling_cfg: Optional[JointCouplingCostCfg] = None

    #: Extra selected-joint bounds beyond global URDF limits.
    joint_bound_cfg: Optional[JointBoundCostCfg] = None

    #: Center-of-mass box cost/constraint. Requires transition compute_com=True.
    com_box_cfg: Optional[ComBoxCostCfg] = None

    #: Soft target posture for selected joints, useful for redundant IK preferences.
    joint_posture_cfg: Optional[JointPostureCostCfg] = None

    #: Soft relative link-position preference, e.g. elbow below shoulder/wrist.
    link_pose_relation_cfg: Optional[LinkPoseRelationCostCfg] = None

    def __post_init__(self):
        from .cost_manager_robot import RobotCostManager

        self.class_type = RobotCostManager

    @staticmethod
    def create(
        data_dict: Dict,
        scene_collision_checker: Optional[SceneCollision] = None,
        device_cfg: DeviceCfg = DeviceCfg(),
    ) -> RobotCostManagerCfg:
        """Create a :class:`RobotCostManagerCfg` from a raw dictionary.

        Instantiates each cost-component config present in *data_dict* and
        attaches the scene collision checker when provided.
        """
        cost_key_map = {
            "self_collision_cfg": SelfCollisionCostCfg,
            "cspace_cfg": CSpaceCostCfg,
            "scene_collision_cfg": SceneCollisionCostCfg,
            "start_cspace_dist_cfg": CSpaceDistCostCfg,
            "target_cspace_dist_cfg": CSpaceDistCostCfg,
            "tool_pose_cfg": ToolPoseCostCfg,
            "joint_bound_cfg": JointBoundCostCfg,
            "joint_coupling_cfg": JointCouplingCostCfg,
            "joint_posture_cfg": JointPostureCostCfg,
            "link_pose_relation_cfg": LinkPoseRelationCostCfg,
            "com_box_cfg": ComBoxCostCfg,
        }
        data = {}
        for k, cfg_class in cost_key_map.items():
            if k in data_dict:
                data[k] = cfg_class(**data_dict[k], device_cfg=device_cfg)

        if "scene_collision_cfg" in data and scene_collision_checker is not None:
            data["scene_collision_cfg"].scene_collision_checker = scene_collision_checker

        return RobotCostManagerCfg(**data)

    def update_collision_activation_distance(self, distance: float):
        if self.scene_collision_cfg is not None:
            self.scene_collision_cfg.activation_distance[:] = distance

    def disable_self_collision(self):
        """Zero out the self-collision cost weight so the solver ignores self-collision."""
        if self.self_collision_cfg is not None:
            self.self_collision_cfg.weight[:] = 0.0

    def update_regularization_weight(
        self, l2_weight: Optional[float] = None, distance_weight: Optional[float] = None
    ):
        pass
