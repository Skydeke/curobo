# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for unpinned lock joints: locks whose value is read from the state."""

import pytest
import torch

from curobo._src.state.state_joint import JointState
from curobo._src.state.state_joint_ops import augment_joint_state

#: All articulated joints, in the order the augmented state is reordered to.
ALL_JOINTS = ["arm_1", "arm_2", "lock_a", "ptu_pan"]
#: The locked joints: lock_a is pinned, ptu_pan is unpinned (value read from the state).
LOCK_JOINTS = ["lock_a", "ptu_pan"]
#: The lock joints that read their value from the supplied state.
UNPINNED_LOCK_JOINTS = ["ptu_pan"]


def _state(position, joint_names):
    """A single-config joint state from a list of positions."""
    return JointState.from_position(
        torch.tensor([position], dtype=torch.float32), joint_names=joint_names
    )


def _values(full_state):
    """Augmented state as ``{joint_name: value}``."""
    return dict(zip(full_state.joint_names, full_state.position[0].tolist()))


def test_pinned_locked_joint_in_state_raises():
    """A pinned joint must not appear in the incoming state (unchanged contract)."""
    cspace = _state([0.1, 0.2, 0.5], ["arm_1", "arm_2", "lock_a"])
    locks = _state([0.5, 0.0], LOCK_JOINTS)
    with pytest.raises(ValueError):
        augment_joint_state(cspace, ALL_JOINTS, locks, UNPINNED_LOCK_JOINTS)


def test_unpinned_lock_reads_value_from_state():
    """An unpinned lock may be supplied with the state; its value wins over the fallback."""
    cspace = _state([0.1, 0.2, 0.9], ["arm_1", "arm_2", "ptu_pan"])
    locks = _state([0.5, 0.0], LOCK_JOINTS)
    values = _values(augment_joint_state(cspace, ALL_JOINTS, locks, UNPINNED_LOCK_JOINTS))
    assert values["arm_1"] == pytest.approx(0.1)
    assert values["lock_a"] == pytest.approx(0.5)  # still pinned
    assert values["ptu_pan"] == pytest.approx(0.9)  # from the state, not the fallback


def test_unpinned_lock_falls_back_when_not_supplied():
    """Before the first state arrives the configured fallback value is used."""
    cspace = _state([0.1, 0.2], ["arm_1", "arm_2"])
    locks = _state([0.5, 0.25], LOCK_JOINTS)
    values = _values(augment_joint_state(cspace, ALL_JOINTS, locks, UNPINNED_LOCK_JOINTS))
    assert values["ptu_pan"] == pytest.approx(0.25)


def test_no_unpinned_locks_keeps_the_pinned_contract():
    """Without unpinned locks the lock set is fully pinned."""
    cspace = _state([0.1, 0.2], ["arm_1", "arm_2"])
    locks = _state([0.5, 0.25], LOCK_JOINTS)
    values = _values(augment_joint_state(cspace, ALL_JOINTS, locks))
    assert values["lock_a"] == pytest.approx(0.5)
    assert values["ptu_pan"] == pytest.approx(0.25)
