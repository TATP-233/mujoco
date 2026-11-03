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

#ifndef MUJOCO_MJADJOINT_H_
#define MUJOCO_MJADJOINT_H_

#include <stdint.h>

#include <mujoco/mjdata.h>
#include <mujoco/mjexport.h>
#include <mujoco/mjmodel.h>

#ifdef __cplusplus
extern "C" {
#endif

#define mjADJOINT_ABI_VERSION 1

#define mjADJOINT_CAP_STEP_VJP       (1ULL << 0)
#define mjADJOINT_CAP_EULER          (1ULL << 1)
#define mjADJOINT_CAP_IMPLICITFAST   (1ULL << 2)
#define mjADJOINT_CAP_CONTACTS       (1ULL << 3)

typedef enum mjtAdjointStatus {
  mjADJ_OK = 0,
  mjADJ_STOP_GRADIENT,
  mjADJ_UNSUPPORTED,
  mjADJ_NON_SMOOTH,
  mjADJ_NOT_CONVERGED,
  mjADJ_INVALID_ARGUMENT,
  mjADJ_ENGINE_ERROR
} mjtAdjointStatus;

typedef struct mjAdjContext_ mjAdjContext;

MJAPI int mjd_adjointVersion(void);
MJAPI uint64_t mjd_adjointCapabilities(void);
MJAPI const char* mjd_adjointStatusString(int status);

// The returned status is UNSUPPORTED until every model gate is satisfied.
MJAPI int mjd_checkAdjointModel(const mjModel* m);

// A context is owned by exactly one worker. make/delete/reserve are cold-path operations.
MJAPI mjAdjContext* mj_makeAdjContext(const mjModel* m);
MJAPI void mj_deleteAdjContext(mjAdjContext* ctx);
MJAPI int mj_reserveAdjContext(mjAdjContext* ctx, int ncon, int nefc, int nJ, int nA);

// Input and output position cotangents use nv-dimensional tangent coordinates.
// An output may exactly alias its corresponding input cotangent. All other overlap is invalid.
// Before forward, errors leave mjData and gradient outputs unchanged. After a successful forward,
// any later error leaves mjData committed and gradient outputs unwritten.
MJAPI int mjd_stepAdjoint(const mjModel* m, mjData* d, mjAdjContext* ctx,
                          const mjtNum* dbar_qpos, const mjtNum* dbar_qvel,
                          mjtNum* dbar_qpos_in, mjtNum* dbar_qvel_in,
                          mjtNum* dbar_ctrl);

#ifdef __cplusplus
}
#endif

#endif  // MUJOCO_MJADJOINT_H_
