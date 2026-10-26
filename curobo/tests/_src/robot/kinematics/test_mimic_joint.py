# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import torch

# CuRobo
from curobo._src.robot.kinematics.kinematics import Kinematics, KinematicsCfg
from curobo._src.state.state_joint import JointState


def test_mimic_config():
    cfg = KinematicsCfg.from_robot_yaml_file("simple_mimic_robot.yml", ["ee_link"])

    robot_model = Kinematics(cfg)
    q = JointState.from_position(
        robot_model.default_joint_position.view(-1).clone(), joint_names=robot_model.joint_names
    )
    q = robot_model.get_full_js(q)

    q_mimic = robot_model.get_mimic_js(q)
    assert len(q_mimic.joint_names) == 3


def test_unpinned_lock_updates_mimic_children():
    """Mimic links must follow an unpinned lock joint, not stay at bake pose.

    ``simple_mimic_robot.yml`` pins ``chain_1_active_joint_1``; overriding the
    value with ``None`` makes it an unpinned lock (value comes from the state).
    Its mimic link has to end up at the same pose it would have if the active
    joint were still a planning DOF — otherwise only the active link moves and
    every mimicking link (e.g. the second Robotiq finger) stays open.
    """
    active_joint = "chain_1_active_joint_1"
    tool_frames = ["chain_1_link_2", "ee_link"]
    mimic_link = "chain_1_link_2"

    locked = Kinematics(
        KinematicsCfg.from_robot_yaml_file(
            "simple_mimic_robot.yml",
            tool_frames=tool_frames,
            lock_joints={active_joint: None},
        )
    )
    locked.update_lock_joints({active_joint: 0.7})
    locked_pose = locked.get_link_poses(
        torch.zeros(1, locked.dof, device=locked.device_cfg.device), [mimic_link]
    )

    unlocked = Kinematics(
        KinematicsCfg.from_robot_yaml_file(
            "simple_mimic_robot.yml", tool_frames=tool_frames, lock_joints={}
        )
    )
    unlocked_pose = unlocked.get_link_poses(
        torch.tensor([[0.7, 0.0]], device=unlocked.device_cfg.device), [mimic_link]
    )

    assert torch.allclose(locked_pose.position, unlocked_pose.position, atol=1e-5)
    assert torch.allclose(locked_pose.quaternion, unlocked_pose.quaternion, atol=1e-5)
