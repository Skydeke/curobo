# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Robot segmentation on the GPU: mask a depth image against the robot's bodies.

Replaces the host-side loop that walked every valid pixel in torch: that form
materialised the point cloud, filtered it against a bounding box, then measured
every surviving point against every robot sphere in Python chunks - a few dozen
torch ops, several host<->device transfers and a stream synchronisation per
frame. On the real stack a single 800x600 frame cost ~65 ms that way, which is
GPU-idle time in the middle of the perception lane.

This version is one kernel launch: per pixel it unprojects with the camera
intrinsics, transforms into the base frame with the camera pose, and zeroes the
pixel when it lies within ``threshold`` of any robot sphere. Pixels with an
invalid depth (0, NaN, inf) are zeroed, exactly as the host version did, so the
Mapper sees the same image.

The sphere set is the robot's collision spheres for the current joint state, in
the base frame, as ``(x, y, z, radius)`` rows - the same tensor FK already
produces, so no repacking happens here.
"""

from typing import Sequence, Tuple

import numpy as np
import torch
import warp as wp

from curobo._src.util.warp import get_warp_device_stream


@wp.kernel
def robot_sphere_bounds_kernel(
    spheres: wp.array(dtype=wp.vec4),
    bounds: wp.array(dtype=wp.float32),
):
    """Bounding box of the robot's spheres, expanded by each radius.

    ``bounds`` holds ``[min_x, min_y, min_z, max_x, max_y, max_z]`` and must be
    pre-filled with +inf / -inf. A point outside that box cannot be within any
    sphere's radius, so the mask kernel skips the per-sphere loop for it - the
    room, the floor and the far wall cost one comparison.

    Args:
        spheres: ``(S, 4)`` collision spheres ``(x, y, z, radius)``.
        bounds: ``(6,)`` float32 accumulator, seeded with +inf/-inf.
    """
    tid = wp.tid()
    if tid >= spheres.shape[0]:
        return
    sphere = spheres[tid]
    radius = sphere[3]
    wp.atomic_min(bounds, 0, sphere[0] - radius)
    wp.atomic_min(bounds, 1, sphere[1] - radius)
    wp.atomic_min(bounds, 2, sphere[2] - radius)
    wp.atomic_max(bounds, 3, sphere[0] + radius)
    wp.atomic_max(bounds, 4, sphere[1] + radius)
    wp.atomic_max(bounds, 5, sphere[2] + radius)


@wp.kernel
def robot_mask_depth_kernel(
    depth_in: wp.array3d(dtype=wp.float32),
    depth_out: wp.array3d(dtype=wp.float32),
    spheres: wp.array(dtype=wp.vec4),
    bounds: wp.array(dtype=wp.float32),
    cam_to_base: wp.transform,
    fx: wp.float32,
    fy: wp.float32,
    cx: wp.float32,
    cy: wp.float32,
    threshold: wp.float32,
):
    """Zero depth pixels that fall inside (or within threshold of) a robot sphere.

    Args:
        depth_in: Depth images ``(B, H, W)`` in metres.
        depth_out: Masked depth, same shape.
        spheres: Robot collision spheres in the base frame, ``(S, 4)`` as
            ``(x, y, z, radius)``.
        bounds: ``(6,)`` sphere bounding box from :func:`robot_sphere_bounds_kernel`.
        cam_to_base: Camera -> base transform, applied to the unprojected pixel.
        fx: Focal length x (px).
        fy: Focal length y (px).
        cx: Principal point x (px).
        cy: Principal point y (px).
        threshold: Distance (m) within which a point counts as robot.
    """
    b, i, j = wp.tid()
    if b >= depth_in.shape[0] or i >= depth_in.shape[1] or j >= depth_in.shape[2]:
        return

    z = depth_in[b, i, j]
    # Invalid depth (0 / negative / NaN / inf) is written as 0, matching the
    # host implementation's `valid` mask: a pixel with no return must not keep a
    # stale value.
    if not (z > 0.0 and z < 1.0e30):
        depth_out[b, i, j] = 0.0
        return

    # Pixel -> camera (z-forward optical convention), then camera -> base. Row
    # index is the y axis and column index the x axis, as in the host version.
    point_cam = wp.vec3(
        (wp.float32(j) - cx) * z / fx,
        (wp.float32(i) - cy) * z / fy,
        z,
    )
    point_base = wp.transform_point(cam_to_base, point_cam)

    # Outside the robot's (radius-expanded) box nothing can be within threshold
    # of a sphere: keep the pixel without touching the sphere list.
    if (
        point_base[0] < bounds[0] - threshold or point_base[1] < bounds[1] - threshold or
        point_base[2] < bounds[2] - threshold or point_base[0] > bounds[3] + threshold or
        point_base[1] > bounds[4] + threshold or point_base[2] > bounds[5] + threshold
    ):
        depth_out[b, i, j] = z
        return

    # A point is robot when |p - c| - r <= threshold, i.e. |p - c|^2 <= (r +
    # threshold)^2 - compared squared to keep the sqrt out of the inner loop.
    n_spheres = spheres.shape[0]
    for s in range(n_spheres):
        sphere = spheres[s]
        delta = point_base - wp.vec3(sphere[0], sphere[1], sphere[2])
        reach = sphere[3] + threshold
        if wp.dot(delta, delta) <= reach * reach:
            depth_out[b, i, j] = 0.0
            return
    depth_out[b, i, j] = z


def _warp_device_for(reference: torch.Tensor):
    """Warp device/stream for a tensor, tolerating a device without an index.

    ``get_warp_device_stream`` reads the CUDA index off the tensor, which is
    ``None`` for a tensor created on the bare ``"cuda"`` device. The bridge's
    callers pass buffers built with an explicit index, so that is the primary
    path; the fallback keeps the helper usable from plain tests and scripts.
    """
    try:
        return get_warp_device_stream(reference)
    except Exception:
        if reference.is_cuda:
            index = reference.device.index or 0
            return wp.get_device("cuda:%d" % index), None
        return wp.get_device("cpu"), None


def mask_depth_by_spheres(
    depth: np.ndarray,
    spheres: torch.Tensor,
    cam_pose: Sequence[float],
    intrinsics: Tuple[float, float, float, float],
    threshold: float,
) -> np.ndarray:
    """Mask one depth image against the robot's spheres.

    Args:
        depth: ``(H, W)`` float32 depth in metres (host array).
        spheres: ``(S, 4)`` collision spheres ``(x, y, z, radius)`` in the base
            frame, on the same device the kernels run on.
        cam_pose: Camera -> base pose as ``[x, y, z, qw, qx, qy, qz]``.
        intrinsics: ``(fx, fy, cx, cy)`` in pixels.
        threshold: Distance (m) within which a point counts as robot.

    Returns:
        ``(H, W)`` float32 masked depth on the host.
    """
    depth_np = np.ascontiguousarray(depth, dtype=np.float32)
    height, width = int(depth_np.shape[0]), int(depth_np.shape[1])
    fx, fy, cx, cy = (float(v) for v in intrinsics)

    device, stream = _warp_device_for(spheres)
    depth_in = wp.array(
        depth_np.reshape(1, height, width), dtype=wp.float32, device=device)
    depth_out = wp.empty((1, height, width), dtype=wp.float32, device=device)
    spheres_wp = wp.from_torch(spheres.detach().view(-1, 4), dtype=wp.vec4)
    # Bounding box of the spheres, computed on the device (one tiny kernel) so no
    # host synchronisation is needed to build the early-out.
    bounds = wp.array(
        [float("inf")] * 3 + [float("-inf")] * 3, dtype=wp.float32, device=device)
    wp.launch(
        kernel=robot_sphere_bounds_kernel,
        dim=max(1, int(spheres.shape[0])),
        inputs=[spheres_wp, bounds],
        stream=stream,
        device=device,
    )
    cam_to_base = wp.transform(
        wp.vec3(float(cam_pose[0]), float(cam_pose[1]), float(cam_pose[2])),
        # wp.quat takes (x, y, z, w); the pose arrives w-first.
        wp.quat(
            float(cam_pose[4]), float(cam_pose[5]), float(cam_pose[6]),
            float(cam_pose[3])),
    )

    wp.launch(
        kernel=robot_mask_depth_kernel,
        dim=(1, height, width),
        inputs=[
            depth_in,
            depth_out,
            spheres_wp,
            bounds,
            cam_to_base,
            fx,
            fy,
            cx,
            cy,
            float(threshold),
        ],
        stream=stream,
        device=device,
    )
    return depth_out.numpy()[0]
