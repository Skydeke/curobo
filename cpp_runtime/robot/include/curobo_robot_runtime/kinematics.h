// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>
#include <string>

#include "curobo_robot_runtime/api.h"

#include <vector>

namespace curobo_robot_runtime {
/** Replace the fixed transforms in every live native stage using this artifact. */
CUROBO_ROBOT_API void update_artifact_fixed_transforms(
    const std::string& artifact_dir, int device_ordinal,
    const std::vector<float>& data);
}  // namespace curobo_robot_runtime

namespace curobo::robot {

/**
 * The artifact's collision-sphere model, copied to host memory.
 *
 * This is the same data the native collision kernels consume: link-local
 * spheres, the self-collision pair list, per-sphere padding, and the
 * sphere-to-link map. It is not re-fitted from the URDF.
 */
struct SphereModel {
  int num_spheres{0};
  int num_self_collision_pairs{0};
  std::vector<float> link_spheres;            // [num_spheres, 4]
  std::vector<float> self_collision_padding;  // [num_spheres]
  std::vector<std::int16_t> self_collision_pairs;  // [num_pairs, 2]
  std::vector<std::int16_t> link_sphere_map;  // [num_spheres]
};

/** Typed C++ interface for artifact-backed forward kinematics and Jacobians. */
class Kinematics final {
 public:
  explicit Kinematics(const std::string& artifact_directory, int device_ordinal = 0);
  ~Kinematics();

  Kinematics(const Kinematics&) = delete;
  Kinematics& operator=(const Kinematics&) = delete;
  Kinematics(Kinematics&& other) noexcept;
  Kinematics& operator=(Kinematics&& other) noexcept;

  CuroboRobotInfo info() const;
  std::string joint_name(int joint_index) const;
  std::string tool_frame_name(int tool_frame_index) const;

  /**
   * Enqueue fused FK and Jacobian computation on stream.
   *
   * All pointers refer to contiguous CUDA float32 memory. See
   * CuroboRobotKinematicsRequest for tensor shapes and conventions.
   */
  void forward(const float* joint_positions, float* link_positions,
               float* link_quaternions, float* link_jacobians, int batch_size,
               int horizon, cudaStream_t stream = nullptr) const;

  /** Copy the artifact's static collision-sphere model to host memory. */
  SphereModel sphere_model() const;

  /**
   * Enqueue fused collision-sphere FK on stream.
   *
   * joint_positions: [batch_size, horizon, num_dof]
   * robot_spheres:   [batch_size, horizon, num_spheres, 4]
   *
   * The number of spheres is available from `sphere_model().num_spheres`.
   */
  void forward_spheres(const float* joint_positions, float* robot_spheres,
                       int batch_size, int horizon,
                       cudaStream_t stream = nullptr) const;

 private:
  CuroboRobotRuntime* runtime_{nullptr};
};

}  // namespace curobo::robot
