# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for the GPU robot segmentation kernel.

Tests ``mask_depth_by_spheres`` in
``curobo._src.perception.mapper.kernel.wp_robot_mask``:
- Pixels within ``threshold`` of an enabled sphere are zeroed.
- Disabled spheres (negative radius, cuRobo's sentinel) are ignored instead of
  acting as a ~100 m reach that masks the whole bounding box.
"""

# Third Party
import numpy as np
import pytest
import torch

# CuRobo
from curobo._src.perception.mapper.kernel.wp_robot_mask import (
    mask_depth_by_spheres,
)
from curobo._src.util.warp import init_warp

# Initialize Warp at module load
init_warp()


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def device():
    """Get CUDA device (Warp kernels in this module require it)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for robot mask tests")
    return torch.device("cuda")


# ============================================================================
# Helpers
# ============================================================================


def _depth_plane(height: int, width: int, distance: float) -> np.ndarray:
    """A constant-depth image (a plane at ``distance`` in metres)."""
    return np.full((height, width), distance, dtype=np.float32)


def _reference_mask(
    depth: np.ndarray,
    spheres: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    threshold: float,
) -> np.ndarray:
    """Host reference: True where a pixel is robot (must be zeroed).

    Disabled spheres (``radius < 0``) are ignored, matching cuRobo's collision
    check.
    """
    height, width = depth.shape
    rows, cols = np.mgrid[0:height, 0:width]
    points = np.stack(
        [
            (cols - cx) * depth / fx,
            (rows - cy) * depth / fy,
            depth,
        ],
        axis=-1,
    ).astype(np.float32)

    enabled = spheres[spheres[:, 3] >= 0.0]
    if enabled.shape[0] == 0:
        return np.zeros((height, width), dtype=bool)

    distance = (
        np.linalg.norm(points[..., None, :] - enabled[None, None, :, :3], axis=-1)
        - enabled[None, None, :, 3]
    )
    return (distance.min(axis=-1) <= threshold) & (depth > 0.0)


# ============================================================================
# Tests
# ============================================================================


def test_masks_only_points_within_threshold(device):
    """Pixels near an enabled sphere are zeroed; the rest keep their depth."""
    height, width = 48, 64
    fx = fy = 100.0
    cx, cy = 32.0, 24.0
    threshold = 0.02
    depth = _depth_plane(height, width, 2.0)
    spheres_np = np.array([[0.0, 0.0, 2.0, 0.1]], dtype=np.float32)
    spheres = torch.as_tensor(spheres_np, device=device)

    masked = mask_depth_by_spheres(
        depth, spheres, [0, 0, 0, 1, 0, 0, 0], (fx, fy, cx, cy), threshold
    )
    got = masked == 0.0
    expected = _reference_mask(depth, spheres_np, fx, fy, cx, cy, threshold)

    assert got.sum() > 0
    np.testing.assert_array_equal(got, expected)


def test_disabled_spheres_do_not_mask_the_bounding_box(device):
    """A negative-radius sphere must not mask points inside the spheres' box.

    cuRobo's loader fills ``extra_collision_spheres`` links (e.g.
    ``attached_object``) with ``radius=-100`` placeholders. Squaring
    ``radius + threshold`` would turn that sentinel into a ~100 m reach, masking
    every pixel inside the (axis-aligned) box spanning the real spheres.
    """
    height, width = 80, 160
    fx = fy = 300.0
    cx, cy = 80.0, 40.0
    threshold = 0.025
    depth = _depth_plane(height, width, 3.0)
    spheres_np = np.array(
        [
            [-0.2, 0.0, 3.0, 0.02],
            [0.2, 0.0, 3.0, 0.02],
            [0.0, 0.0, 3.0, -100.0],
        ],
        dtype=np.float32,
    )
    spheres = torch.as_tensor(spheres_np, device=device)

    masked = mask_depth_by_spheres(
        depth, spheres, [0, 0, 0, 1, 0, 0, 0], (fx, fy, cx, cy), threshold
    )
    got = masked == 0.0
    expected = _reference_mask(depth, spheres_np, fx, fy, cx, cy, threshold)

    assert got.sum() > 0
    np.testing.assert_array_equal(got, expected)
    # The disabled placeholder is at the middle of the two real spheres; a point
    # between them is inside the box but far from any robot surface.
    assert not got[height // 2, width // 2]


def test_all_spheres_disabled_keeps_valid_depth(device):
    """With every sphere disabled nothing is segmented (matches the host path)."""
    height, width = 24, 32
    depth = _depth_plane(height, width, 1.5)
    spheres = torch.tensor(
        [[0.0, 0.0, 1.5, -100.0], [0.1, 0.0, 1.5, -100.0]],
        dtype=torch.float32,
        device=device,
    )

    masked = mask_depth_by_spheres(
        depth, spheres, [0, 0, 0, 1, 0, 0, 0], (50.0, 50.0, 16.0, 12.0), 0.025
    )

    np.testing.assert_array_equal(masked, depth)


def test_invalid_depth_is_zeroed(device):
    """Invalid input pixels (0, negative) stay zeroed, as the host path did."""
    height, width = 16, 16
    depth = _depth_plane(height, width, 2.0)
    depth[0:4, :] = 0.0
    depth[4:8, :] = -1.0
    # Sphere outside the camera frustum: no valid pixel is robot.
    spheres = torch.tensor([[5.0, 5.0, 2.0, 0.05]], dtype=torch.float32, device=device)

    masked = mask_depth_by_spheres(
        depth, spheres, [0, 0, 0, 1, 0, 0, 0], (40.0, 40.0, 8.0, 8.0), 0.02
    )

    assert np.all(masked[0:8, :] == 0.0)
    assert np.all(masked[8:, :] == 2.0)
