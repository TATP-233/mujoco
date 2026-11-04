// Copyright 2026 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MUJOCO_SRC_ENGINE_ENGINE_DERIVATIVE_ADJOINT_H_
#define MUJOCO_SRC_ENGINE_ENGINE_DERIVATIVE_ADJOINT_H_

#include <mujoco/mjadjoint.h>

#ifdef __cplusplus
extern "C" {
#endif

// VJP of qfrc_actuator with respect to ctrl for gated scalar JOINT actuators.
// d must contain a completed forward evaluation. Saturated clamps use the outward zero derivative.
MJAPI int mjd_actuatorCtrlAdjoint(const mjModel* m, const mjData* d,
                                  const mjtNum* bar_qfrc_actuator, mjtNum* dbar_ctrl);

// Position gradients use nv-dimensional tangent coordinates, like the fused step API.
MJAPI int mjd_actuatorStateAdjoint(const mjModel* m, const mjData* d,
                                   const mjtNum* bar_qfrc_actuator,
                                   mjtNum* dbar_qpos, mjtNum* dbar_qvel);

// VJP of linear scalar-joint spring/damper passive forces; other passive terms are gated.
MJAPI int mjd_passiveStateAdjoint(const mjModel* m, const mjData* d,
                                  const mjtNum* bar_qfrc_passive,
                                  mjtNum* dbar_qpos, mjtNum* dbar_qvel);

// qDeriv is caller-owned nD scratch; d's persistent qDeriv is not modified.
MJAPI int mjd_rneBiasVelAdjoint(const mjModel* m, mjData* d,
                                const mjtNum* bar_qfrc_bias,
                                mjtNum* qDeriv, mjtNum* dbar_qvel);

// VJP of qfrc_bias(qpos,qvel) for scalar hinge/slide trees with an optional free root.
MJAPI int mjd_rneBiasPosAdjoint(const mjModel* m, mjData* d,
                                const mjtNum* bar_qfrc_bias, mjtNum* dbar_qpos);

// VJP of mj_integratePos with input/output qpos cotangents in their local nv tangent charts.
MJAPI int mjd_integratePosAdjoint(const mjModel* m, const mjtNum* qvel, mjtNum dt,
                                  const mjtNum* dbar_qpos_out, mjtNum* dbar_qpos_in,
                                  mjtNum* dbar_qvel);

// y = Q^-T * bar using the current qH factor; exact input/output aliasing is supported.
MJAPI int mjd_implicitFastSolveAdjoint(const mjModel* m, const mjData* d,
                                       const mjtNum* bar, mjtNum* y);

// qDeriv is caller-owned nD scratch for S = d(qfrc_smooth)/d(qvel).
MJAPI int mjd_smoothVelAdjoint(const mjModel* m, const mjData* d,
                               const mjtNum* bar_qfrc_smooth,
                               mjtNum* qDeriv, mjtNum* dbar_qvel);

// Euler qacc_smooth VJP: qDeriv is nD scratch and y is nv scratch for M^-T*bar.
MJAPI int mjd_smoothAccVelAdjoint(const mjModel* m, const mjData* d,
                                  const mjtNum* bar_qacc_smooth,
                                  mjtNum* qDeriv, mjtNum* y, mjtNum* dbar_qvel);

// Euler qacc_smooth control VJP; y is nv scratch for M^-T*bar.
MJAPI int mjd_smoothAccCtrlAdjoint(const mjModel* m, const mjData* d,
                                   const mjtNum* bar_qacc_smooth,
                                   mjtNum* y, mjtNum* dbar_ctrl);

// Position VJP of one static plane-sphere contact with active-set cotangents.
MJAPI int mjd_planeSpherePosAdjoint(const mjModel* m, const mjData* d,
                                    const mjtNum* y, const mjtNum* lambda, const mjtNum* mu,
                                    const mjtNum* hinv, const mjtNum* qacc_adv,
                                    mjtNum* dbar_qpos);

MJAPI int mjd_planeSphereStepPosAdjoint(const mjModel* m, const mjData* d,
                                        const mjtNum* y, const mjtNum* lambda, const mjtNum* mu,
                                        const mjtNum* hinv, const mjtNum* qacc_adv,
                                        mjtNum* dbar_qpos);

// Position VJP of one capsule-capsule / sphere-capsule / sphere-sphere contact
// (non-parallel capsule axes, interior closest points) with active-set cotangents.
// Clipped-witness configurations are NON_SMOOTH.
MJAPI int mjd_capsuleCapsulePosAdjoint(const mjModel* m, const mjData* d,
                                       const mjtNum* y, const mjtNum* lambda, const mjtNum* mu,
                                       const mjtNum* hinv, const mjtNum* qacc_adv,
                                       mjtNum* dbar_qpos);

MJAPI int mjd_capsuleCapsuleStepPosAdjoint(const mjModel* m, const mjData* d,
                                           const mjtNum* y, const mjtNum* lambda,
                                           const mjtNum* mu, const mjtNum* hinv,
                                           const mjtNum* qacc_adv, mjtNum* dbar_qpos);

// VJP of y^T M(qpos) w for scalar hinge/slide trees with an optional free root.
MJAPI int mjd_massPosAdjoint(const mjModel* m, mjData* d,
                             const mjtNum* y, const mjtNum* w, mjtNum* dbar_qpos);

// VJP of y^T M(qpos) w for one free joint with an arbitrary CoM offset.
MJAPI int mjd_freeMassPosAdjoint(const mjModel* m, const mjData* d,
                                 const mjtNum y[6], const mjtNum w[6], mjtNum dbar_qpos[6]);

// VJP of qfrc_bias(qpos,qvel) for a standalone free body.
MJAPI int mjd_freeBiasPosAdjoint(const mjModel* m, const mjData* d,
                                 const mjtNum bar_qfrc_bias[6], mjtNum dbar_qpos[6]);

// qacc_smooth position VJP for fixed trees using d's current qLD factor; y is nv scratch.
MJAPI int mjd_fixedSmoothAccPosAdjoint(const mjModel* m, mjData* d,
                                       const mjtNum* bar_qacc_smooth,
                                       mjtNum* y, mjtNum* dbar_qpos);

// Euler qacc_smooth position VJP for a standalone free body; y is nv scratch.
MJAPI int mjd_freeSmoothAccPosAdjoint(const mjModel* m, const mjData* d,
                                      const mjtNum bar_qacc_smooth[6],
                                      mjtNum y[6], mjtNum dbar_qpos[6]);

#ifdef __cplusplus
}
#endif

#endif  // MUJOCO_SRC_ENGINE_ENGINE_DERIVATIVE_ADJOINT_H_
