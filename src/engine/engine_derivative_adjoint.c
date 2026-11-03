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

#include "engine/engine_derivative_adjoint.h"

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <mujoco/mjmacro.h>
#include <mujoco/mujoco.h>
#include "engine/engine_callback.h"
#include "engine/engine_core_smooth.h"
#include "engine/engine_core_util.h"
#include "engine/engine_derivative.h"
#include "engine/engine_memory.h"
#include "engine/engine_util_errmem.h"
#include "engine/engine_util_solve.h"
#include "engine/engine_util_spatial.h"

struct mjAdjContext_ {
  uint32_t magic;
  int abi_version;
  int nq;
  int nv;
  int nu;
  mjtSize nM;
  mjtSize nC;
  mjtSize nD;
  uint64_t layout_signature;
  uint64_t generation;
  mjtNum* numbers;
  size_t numbers_size;
  mjtNum* qpos;
  mjtNum* qvel;
  mjtNum* ctrl;
  mjtNum* bar_qpos;
  mjtNum* bar_qvel;
  mjtNum* grad_qpos;
  mjtNum* grad_qvel;
  mjtNum* grad_ctrl;
  mjtNum* sparse_scratch;
  mjtNum* step_work;
  int capacity_ncon;
  int capacity_nefc;
  int capacity_nJ;
  int capacity_nA;
  mjtNum* tape_numbers;
  int* tape_ints;
};

static const uint32_t kContextMagic = 0x4d4a4144;  // MJAD

static int checked_add(size_t* value, size_t increment) {
  if (increment > SIZE_MAX - *value) {
    return 0;
  }
  *value += increment;
  return 1;
}

static int checked_bytes(size_t count, size_t element_size, size_t* bytes) {
  if (element_size && count > SIZE_MAX / element_size) {
    return 0;
  }
  *bytes = count * element_size;
  return 1;
}

static uint64_t hash_bytes(uint64_t hash, const void* data, size_t size) {
  const unsigned char* bytes = (const unsigned char*)data;
  for (size_t i = 0; i < size; ++i) {
    hash ^= bytes[i];
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

static uint64_t hash_ints(uint64_t hash, const int* values, mjtSize count) {
  if (count > 0) {
    size_t bytes;
    if (!checked_bytes((size_t)count, sizeof(int), &bytes)) {
      return 0;
    }
    hash = hash_bytes(hash, values, bytes);
  }
  return hash;
}

static uint64_t layout_signature(const mjModel* m) {
  uint64_t hash = UINT64_C(1469598103934665603);
  hash = hash_bytes(hash, &m->nq, sizeof(m->nq));
  hash = hash_bytes(hash, &m->nv, sizeof(m->nv));
  hash = hash_bytes(hash, &m->nu, sizeof(m->nu));
  hash = hash_bytes(hash, &m->nM, sizeof(m->nM));
  hash = hash_bytes(hash, &m->nC, sizeof(m->nC));
  hash = hash_bytes(hash, &m->nD, sizeof(m->nD));
  hash = hash_ints(hash, m->dof_parentid, m->nv);
  hash = hash_ints(hash, m->M_rownnz, m->nv);
  hash = hash_ints(hash, m->M_rowadr, m->nv);
  hash = hash_ints(hash, m->M_colind, m->nC);
  hash = hash_ints(hash, m->mapM2M, m->nC);
  hash = hash_ints(hash, m->D_rownnz, m->nv);
  hash = hash_ints(hash, m->D_rowadr, m->nv);
  hash = hash_ints(hash, m->D_diag, m->nv);
  hash = hash_ints(hash, m->D_colind, m->nD);
  hash = hash_ints(hash, m->mapM2D, m->nD);
  hash = hash_ints(hash, m->mapD2M, m->nC);
  return hash;
}

static int valid_context(const mjAdjContext* ctx) {
  return ctx && ctx->magic == kContextMagic && ctx->abi_version == mjADJOINT_ABI_VERSION;
}

int mjd_adjointVersion(void) {
  return mjADJOINT_ABI_VERSION;
}

uint64_t mjd_adjointCapabilities(void) {
  return mjADJOINT_CAP_STEP_VJP | mjADJOINT_CAP_EULER | mjADJOINT_CAP_IMPLICITFAST;
}

const char* mjd_adjointStatusString(int status) {
  switch (status) {
    case mjADJ_OK: return "OK";
    case mjADJ_STOP_GRADIENT: return "STOP_GRADIENT";
    case mjADJ_UNSUPPORTED: return "UNSUPPORTED";
    case mjADJ_NON_SMOOTH: return "NON_SMOOTH";
    case mjADJ_NOT_CONVERGED: return "NOT_CONVERGED";
    case mjADJ_INVALID_ARGUMENT: return "INVALID_ARGUMENT";
    case mjADJ_ENGINE_ERROR: return "ENGINE_ERROR";
    default: return "UNKNOWN";
  }
}

int mjd_checkAdjointModel(const mjModel* m) {
  if (!m) {
    return mjADJ_INVALID_ARGUMENT;
  }
  if (m->na != 0 || m->neq != 0 || m->ntendon != 0 || m->nflex != 0 ||
      m->nplugin != 0 || m->nhistory != 0) {
    return mjADJ_UNSUPPORTED;
  }
  if (m->opt.solver != mjSOL_NEWTON ||
      (m->opt.integrator != mjINT_EULER && m->opt.integrator != mjINT_IMPLICITFAST) ||
      m->opt.noslip_iterations != 0 ||
      !(m->opt.disableflags & mjDSBL_ISLAND) ||
      (m->opt.enableflags & (mjENBL_DIAGEXACT | mjENBL_SLEEP))) {
    return mjADJ_UNSUPPORTED;
  }
  if (m->opt.density != 0 || m->opt.viscosity != 0 ||
      m->flg_adhesion || m->flg_surfacevel) {
    return mjADJ_UNSUPPORTED;
  }
  // mjcb_time only measures time and does not alter physics; Python installs it by default.
  if (mjcb_passive || mjcb_control || mjcb_contactfilter || mjcb_sensor ||
      mjcb_act_dyn || mjcb_act_gain || mjcb_act_bias) {
    return mjADJ_UNSUPPORTED;
  }
  for (int i = 0; i < m->ngeom * mjNFLUID; ++i) {
    if (m->geom_fluid[i] != 0) {
      return mjADJ_UNSUPPORTED;
    }
  }
  for (int i = 0; i < m->nsensor; ++i) {
    if (m->sensor_type[i] == mjSENS_USER) {
      return mjADJ_UNSUPPORTED;
    }
  }
  for (int i = 0; i < m->nactuator; ++i) {
    if (m->actuator_dyntype[i] != mjDYN_NONE ||
        m->actuator_gaintype[i] != mjGAIN_FIXED ||
        (m->actuator_biastype[i] != mjBIAS_NONE &&
         m->actuator_biastype[i] != mjBIAS_AFFINE) ||
        m->actuator_delay[i] != 0 ||
        m->actuator_trntype[i] != mjTRN_JOINT ||
        m->actuator_plugin[i] >= 0) {
      return mjADJ_UNSUPPORTED;
    }
    int joint = m->actuator_trnid[2*i];
    if (joint < 0 || joint >= m->njnt ||
        (m->jnt_type[joint] != mjJNT_HINGE && m->jnt_type[joint] != mjJNT_SLIDE)) {
      return mjADJ_UNSUPPORTED;
    }
  }
  if (m->opt.integrator == mjINT_IMPLICITFAST) {
    for (int i = 0; i < m->njnt; ++i) {
      if (m->jnt_type[i] == mjJNT_FREE && mj_isFreeBody(m, m->jnt_bodyid[i])) {
        return mjADJ_UNSUPPORTED;
      }
    }
  }
  return mjADJ_OK;
}

static int scalar_tree_kind(const mjModel* m) {
  int kind = 1;
  for (int body = 1; body < m->nbody; ++body) {
    if (m->body_jntnum[body] > 1) return 0;
    if (!m->body_jntnum[body]) continue;
    mjtJoint type = (mjtJoint)m->jnt_type[m->body_jntadr[body]];
    if (type == mjJNT_FREE) {
      if (body != 1 || kind == 2) return 0;
      kind = 2;
    } else if (type != mjJNT_HINGE && type != mjJNT_SLIDE) {
      return 0;
    }
  }
  return kind;
}

static int sphere_box_face_pos_adjoint(const mjModel* m, const mjData* d,
                                       const mjtNum* y, const mjtNum* lambda,
                                       const mjtNum* mu, const mjtNum* hinv,
                                       const mjtNum* qacc_adv,
                                       mjtNum impedance_sign, mjtNum* dbar_qpos);
static int sphere_cylinder_pos_adjoint(const mjModel* m, const mjData* d,
                                       const mjtNum* y, const mjtNum* lambda,
                                       const mjtNum* mu, const mjtNum* hinv,
                                       const mjtNum* qacc_adv,
                                       mjtNum impedance_sign, mjtNum* dbar_qpos);

static int plane_sphere_cone_hessian(const mjModel* m, const mjData* d, mjtNum* H) {
  int dim = d->contact[0].dim;
  if (dim < 3 || dim > 6 || d->nefc != dim) return mjADJ_UNSUPPORTED;
  mjtNum jar[6] = {0, 0, 0, 0, 0, 0};
  mj_mulJacVec(m, d, jar, d->qacc);
  mjtNum mu = d->contact[0].mu;
  mjtNum N = mu*(jar[0]-d->efc_aref[0]);
  mjtNum U[5];
  for (int k = 1; k < dim; ++k) {
    U[k-1] = d->contact[0].friction[k-1]*(jar[k]-d->efc_aref[k]);
  }
  mjtNum T = mju_norm(U, dim-1);
  if (T <= 0 || mu <= 0)     return mjADJ_NON_SMOOTH;
  mjtNum Dm = d->efc_D[0]/(mu*mu*(1+mu*mu));
  mju_zero(H, dim*dim);
  H[0] = Dm*mu*mu;
  for (int k = 1; k < dim; ++k) {
    mjtNum friction_k = d->contact[0].friction[k-1];
    H[k] = -Dm*mu*mu*friction_k*U[k-1]/T;
    H[dim*k] = H[k];
    for (int j = k; j < dim; ++j) {
      mjtNum friction_j = d->contact[0].friction[j-1];
      mjtNum value = mu*N*U[j-1]*U[k-1]/(T*T*T);
      if (j == k) value += mu*mu-mu*N/T;
      H[dim*k+j] = Dm*friction_k*friction_j*value;
      H[dim*j+k] = H[dim*k+j];
    }
  }
  return mjADJ_OK;
}

static int contact_pair_status(const mjModel* m, const mjData* d, int* pair_kind) {
  if (pair_kind) *pair_kind = 0;
  if (!d->ncon) return mjADJ_OK;
  if (d->ne || d->nf || d->ncon != 1) return mjADJ_UNSUPPORTED;
  int dim = d->contact[0].dim;
  if (!((dim == 1 && d->nefc == 1) ||
        (dim >= 3 && m->opt.cone == mjCONE_PYRAMIDAL && d->nefc == 2*(dim-1)) ||
        (dim >= 3 && dim <= 6 && m->opt.cone == mjCONE_ELLIPTIC && d->nefc == dim))) {
    return mjADJ_UNSUPPORTED;
  }
  for (int row = 0; row < d->nefc; ++row) {
    if (d->efc_state[row] != mjCNSTRSTATE_QUADRATIC &&
        d->efc_state[row] != mjCNSTRSTATE_SATISFIED &&
        d->efc_state[row] != mjCNSTRSTATE_CONE) {
          return mjADJ_NON_SMOOTH;
    }
  }
  int type1 = m->geom_type[d->contact[0].geom1];
  int type2 = m->geom_type[d->contact[0].geom2];
  if (((type1 == mjGEOM_CAPSULE || type1 == mjGEOM_SPHERE) &&
       (type2 == mjGEOM_CAPSULE || type2 == mjGEOM_SPHERE))) {
    if (pair_kind) *pair_kind = 1;
    return mjADJ_OK;
  }
  int t1b = m->geom_type[d->contact[0].geom1];
  int t2b = m->geom_type[d->contact[0].geom2];
  if ((t1b == mjGEOM_SPHERE || t1b == mjGEOM_CAPSULE) && t2b == mjGEOM_BOX) {
    if (pair_kind) *pair_kind = 2;
    return mjADJ_OK;
  }
  if (t1b == mjGEOM_SPHERE && t2b == mjGEOM_CYLINDER) {
    if (pair_kind) *pair_kind = 3;
    return mjADJ_OK;
  }
  int sphere = d->contact[0].geom1;
  int plane = d->contact[0].geom2;
  if (m->geom_type[plane] != mjGEOM_PLANE) {
    sphere = d->contact[0].geom2;
    plane = d->contact[0].geom1;
  }
  int moving_type = m->geom_type[sphere];
  if ((moving_type != mjGEOM_SPHERE && moving_type != mjGEOM_CAPSULE &&
       moving_type != mjGEOM_BOX && moving_type != mjGEOM_CYLINDER &&
       moving_type != mjGEOM_ELLIPSOID) ||
      m->geom_type[plane] != mjGEOM_PLANE || m->geom_bodyid[plane] != 0) {
    return mjADJ_UNSUPPORTED;
  }
  return mjADJ_OK;
}

mjAdjContext* mj_makeAdjContext(const mjModel* m) {
  if (mjd_checkAdjointModel(m) != mjADJ_OK ||
      m->nq > INT_MAX || m->nv > INT_MAX || m->nu > INT_MAX) {
    return NULL;
  }
  size_t count = 0;
  size_t bytes;
  if (!checked_add(&count, (size_t)m->nq) ||
      !checked_add(&count, 15 * (size_t)m->nv) ||
      !checked_add(&count, 2 * (size_t)m->nu) ||
      !checked_add(&count, (size_t)m->nM) ||
      !checked_add(&count, (size_t)m->nC) ||
      !checked_add(&count, (size_t)m->nD) ||
      !checked_bytes(count, sizeof(mjtNum), &bytes)) {
    return NULL;
  }
  mjAdjContext* ctx = (mjAdjContext*)mju_malloc(sizeof(mjAdjContext));
  if (!ctx) {
    return NULL;
  }
  memset(ctx, 0, sizeof(*ctx));
  ctx->numbers = (mjtNum*)mju_malloc(bytes);
  if (bytes && !ctx->numbers) {
    mju_free(ctx);
    return NULL;
  }
  ctx->magic = kContextMagic;
  ctx->abi_version = mjADJOINT_ABI_VERSION;
  ctx->nq = (int)m->nq;
  ctx->nv = (int)m->nv;
  ctx->nu = (int)m->nu;
  ctx->nM = m->nM;
  ctx->nC = m->nC;
  ctx->nD = m->nD;
  ctx->layout_signature = layout_signature(m);
  ctx->numbers_size = count;
  mjtNum* cursor = ctx->numbers;
  ctx->qpos = cursor; cursor += ctx->nq;
  ctx->qvel = cursor; cursor += ctx->nv;
  ctx->ctrl = cursor; cursor += ctx->nu;
  ctx->bar_qpos = cursor; cursor += ctx->nv;
  ctx->bar_qvel = cursor; cursor += ctx->nv;
  ctx->grad_qpos = cursor; cursor += ctx->nv;
  ctx->grad_qvel = cursor; cursor += ctx->nv;
  ctx->grad_ctrl = cursor; cursor += ctx->nu;
  ctx->sparse_scratch = cursor; cursor += ctx->nM + ctx->nC + ctx->nD;
  ctx->step_work = cursor;
  return ctx;
}

void mj_deleteAdjContext(mjAdjContext* ctx) {
  if (!valid_context(ctx)) {
    return;
  }
  ctx->magic = 0;
  mju_free(ctx->numbers);
  mju_free(ctx->tape_numbers);
  mju_free(ctx->tape_ints);
  mju_free(ctx);
}

int mj_reserveAdjContext(mjAdjContext* ctx, int ncon, int nefc, int nJ, int nA) {
  if (!valid_context(ctx) || ncon < 0 || nefc < 0 || nJ < 0 || nA < 0) {
    return mjADJ_INVALID_ARGUMENT;
  }
  size_t number_count = 0;
  size_t int_count = 0;
  size_t number_bytes;
  size_t int_bytes;
  if (!checked_add(&number_count, 16 * (size_t)ncon) ||
      !checked_add(&number_count, 3 * (size_t)nefc) ||
      !checked_add(&number_count, (size_t)nJ) ||
      !checked_add(&number_count, (size_t)nA) ||
      !checked_add(&int_count, 12 * (size_t)ncon) ||
      !checked_add(&int_count, (size_t)nefc) ||
      !checked_add(&int_count, (size_t)nJ) ||
      !checked_add(&int_count, (size_t)nA) ||
      !checked_bytes(number_count, sizeof(mjtNum), &number_bytes) ||
      !checked_bytes(int_count, sizeof(int), &int_bytes)) {
    return mjADJ_INVALID_ARGUMENT;
  }
  mjtNum* numbers = (mjtNum*)mju_malloc(number_bytes);
  if (number_bytes && !numbers) {
    return mjADJ_ENGINE_ERROR;
  }
  int* ints = (int*)mju_malloc(int_bytes);
  if (int_bytes && !ints) {
    mju_free(numbers);
    return mjADJ_ENGINE_ERROR;
  }
  mju_free(ctx->tape_numbers);
  mju_free(ctx->tape_ints);
  ctx->tape_numbers = numbers;
  ctx->tape_ints = ints;
  ctx->capacity_ncon = ncon;
  ctx->capacity_nefc = nefc;
  ctx->capacity_nJ = nJ;
  ctx->capacity_nA = nA;
  return mjADJ_OK;
}

static int ranges_overlap(const void* first, size_t first_size,
                          const void* second, size_t second_size) {
  if (!first_size || !second_size) {
    return 0;
  }
  uintptr_t first_begin = (uintptr_t)first;
  uintptr_t second_begin = (uintptr_t)second;
  if (first_size > UINTPTR_MAX - first_begin || second_size > UINTPTR_MAX - second_begin) {
    return 1;
  }
  return first_begin < second_begin + second_size && second_begin < first_begin + first_size;
}

static int valid_arguments(const mjModel* m, const mjData* d, const mjAdjContext* ctx,
                           const mjtNum* bar_qpos, const mjtNum* bar_qvel,
                           const mjtNum* grad_qpos, const mjtNum* grad_qvel,
                           const mjtNum* grad_ctrl) {
  if (!m || !d || !valid_context(ctx) || ctx->layout_signature != layout_signature(m) ||
      ctx->nq != m->nq || ctx->nv != m->nv || ctx->nu != m->nu ||
      ctx->nM != m->nM || ctx->nC != m->nC || ctx->nD != m->nD ||
      (m->nq && !d->qpos) || (m->nv && !d->qvel) || (m->nu && !d->ctrl) ||
      (m->nv && (!bar_qpos || !bar_qvel || !grad_qpos || !grad_qvel)) ||
      (m->nu && !grad_ctrl) || !d->arena || d->narena <= 0 || d->parena > (size_t)d->narena) {
    return 0;
  }
  size_t tangent_bytes = (size_t)m->nv * sizeof(mjtNum);
  size_t ctrl_bytes = (size_t)m->nu * sizeof(mjtNum);
  const void* inputs[] = {bar_qpos, bar_qvel};
  const void* outputs[] = {grad_qpos, grad_qvel, grad_ctrl};
  size_t sizes[] = {tangent_bytes, tangent_bytes, ctrl_bytes};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      if (i != j && ranges_overlap(outputs[i], sizes[i], outputs[j], sizes[j])) {
        return 0;
      }
    }
    for (int j = 0; j < 2; ++j) {
      if (ranges_overlap(inputs[j], tangent_bytes, outputs[i], sizes[i]) &&
          !(i == j && inputs[j] == outputs[i])) {
        return 0;
      }
    }
  }
  if (ranges_overlap(bar_qpos, tangent_bytes, bar_qvel, tangent_bytes)) {
    return 0;
  }
  return 1;
}

int mjd_stepAdjoint(const mjModel* m, mjData* d, mjAdjContext* ctx,
                    const mjtNum* dbar_qpos, const mjtNum* dbar_qvel,
                    mjtNum* dbar_qpos_in, mjtNum* dbar_qvel_in,
                    mjtNum* dbar_ctrl) {
  if (!valid_arguments(m, d, ctx, dbar_qpos, dbar_qvel,
                       dbar_qpos_in, dbar_qvel_in, dbar_ctrl)) {
    return mjADJ_INVALID_ARGUMENT;
  }
  int model_status = mjd_checkAdjointModel(m);
  if (model_status != mjADJ_OK) {
    return model_status;
  }
  if (d->ncon > ctx->capacity_ncon || d->nefc > ctx->capacity_nefc ||
      d->nJ > ctx->capacity_nJ || d->nA > ctx->capacity_nA) {
    return mjADJ_ENGINE_ERROR;
  }
  int free_body = m->nbody == 2 && m->njnt == 1 && mj_isFreeBody(m, 1);
  int tree_kind = scalar_tree_kind(m);
  int fixed_tree = tree_kind == 1;
  int free_tree = tree_kind == 2 && m->nbody > 2;
  if ((!free_body && !fixed_tree && !free_tree) ||
      (m->opt.integrator != mjINT_EULER && m->opt.integrator != mjINT_IMPLICITFAST) ||
      (free_body && m->nactuator != 0)) {
    return mjADJ_UNSUPPORTED;
  }
  int euler_damp = 0;
  if (m->opt.integrator == mjINT_EULER &&
      !mjDISABLED(mjDSBL_EULERDAMP) && !mjDISABLED(mjDSBL_DAMPER)) {
    for (int dof = 0; dof < m->nv; ++dof) {
      if (m->dof_damping[dof] > 0 ||
          !mju_isZero(m->dof_dampingpoly+mjNPOLY*dof, mjNPOLY) ||
          m->jnt_actuatorid[m->dof_jntid[dof]] != -1) {
        euler_damp = 1;
        break;
      }
    }
  }
  int implicit_factor = (fixed_tree || free_tree) &&
      (m->opt.integrator == mjINT_IMPLICITFAST || euler_damp);
  mjtNum* integrate_pos = ctx->step_work;
  mjtNum* integrate_vel = integrate_pos+m->nv;
  mjtNum* adj_qacc = integrate_vel+m->nv;
  mjtNum* y = adj_qacc+m->nv;
  mjtNum* y_vel = y+m->nv;
  mjtNum* contact_z = y_vel+m->nv;
  mjtNum* contact_y = contact_z+m->nv;
  mjtNum* qacc_adv = contact_y+m->nv;
  mjtNum* extra_pos = qacc_adv+m->nv;
  mjtNum* extra_vel = extra_pos+m->nv;
  mju_zero(adj_qacc, m->nv);
  int status = mjd_passiveStateAdjoint(m, d, adj_qacc, y, y_vel);
  if (status != mjADJ_OK) {
    return status;
  }
  memcpy(ctx->qpos, d->qpos, (size_t)m->nq * sizeof(mjtNum));
  memcpy(ctx->qvel, d->qvel, (size_t)m->nv * sizeof(mjtNum));
  if (m->nu) {
    memcpy(ctx->ctrl, d->ctrl, (size_t)m->nu * sizeof(mjtNum));
  }
  memcpy(ctx->bar_qpos, dbar_qpos, (size_t)m->nv * sizeof(mjtNum));
  memcpy(ctx->bar_qvel, dbar_qvel, (size_t)m->nv * sizeof(mjtNum));
  ++ctx->generation;
  mj_step(m, d);
  if (d->ncon > ctx->capacity_ncon || d->nefc > ctx->capacity_nefc ||
      d->nJ > ctx->capacity_nJ || d->nA > ctx->capacity_nA) {
    return mjADJ_ENGINE_ERROR;
  }
  int pair_kind = 0;
  status = contact_pair_status(m, d, &pair_kind);
  if (status != mjADJ_OK) {
    return status;
  }
  mjData backward = *d;
  backward.qpos = ctx->qpos;
  backward.qvel = ctx->qvel;
  backward.ctrl = ctx->ctrl;
  status = mjd_integratePosAdjoint(m, d->qvel, m->opt.timestep, ctx->bar_qpos,
                                   integrate_pos, integrate_vel);
  if (status != mjADJ_OK) return status;
  for (int i = 0; i < m->nv; ++i) {
    adj_qacc[i] = m->opt.timestep*(ctx->bar_qvel[i] + integrate_vel[i]);
    qacc_adv[i] = (d->qvel[i]-ctx->qvel[i])/m->opt.timestep;
  }
  const mjtNum* contact_lambda = NULL;
  const mjtNum* contact_q = NULL;
  mjtNum* contact_mu = NULL;
  if (d->ncon) {
    int nv = m->nv;
    int nefc = d->nefc;
    int width = nv > nefc ? nv : nefc;
    size_t mass_size = (size_t)nv*nv;
    size_t cone_size = (size_t)nefc*nefc;
    size_t block_size = mass_size > cone_size ? mass_size : cone_size;
    size_t required = 2*block_size+cone_size+2*width+nefc;
    if ((size_t)ctx->capacity_nA < required) return mjADJ_ENGINE_ERROR;
    int cone_contact = m->opt.cone == mjCONE_ELLIPTIC &&
                       d->efc_state[0] == mjCNSTRSTATE_CONE;
    contact_lambda = backward.efc_force;
    mjtNum* contact_ar = ctx->tape_numbers;
    mjtNum* contact_hessian = contact_ar+block_size;
    mjtNum* contact_factor = contact_hessian+cone_size;
    contact_mu = contact_factor+block_size;
    mjtNum* contact_solution = contact_mu+width;
    mjtNum* contact_q_data = contact_solution+width;
    mju_zero(contact_hessian, cone_size);
    if (cone_contact) {
      int hessian_status = plane_sphere_cone_hessian(m, &backward, contact_hessian);
      if (hessian_status != mjADJ_OK) return hessian_status;
    } else {
      for (int row = 0; row < nefc; ++row) {
        if (d->efc_state[row] == mjCNSTRSTATE_QUADRATIC) {
          contact_hessian[row*nefc+row] = d->efc_D[row];
        }
      }
    }
    if (cone_contact) {
    mj_fullM(m, &backward, contact_ar);
    for (int row = 0; row < nefc; ++row) {
      mju_zero(contact_mu, width);
      contact_mu[row] = 1;
      mju_zero(y_vel, nv);
      mj_mulJacTVec(m, &backward, y_vel, contact_mu);
      for (int col = 0; col < nefc; ++col) {
        mjtNum scale = contact_hessian[row*nefc+col];
        if (!scale) continue;
        mju_zero(contact_mu, width);
        contact_mu[col] = 1;
        mju_zero(contact_z, nv);
        mj_mulJacTVec(m, &backward, contact_z, contact_mu);
        for (int i = 0; i < nv; ++i) {
          for (int j = 0; j < nv; ++j) {
            contact_ar[i*nv+j] += scale*y_vel[i]*contact_z[j];
          }
        }
      }
    }
    memcpy(contact_factor, contact_ar, mass_size*sizeof(mjtNum));
    if (mju_cholFactor(contact_factor, nv, 0) != nv) return mjADJ_ENGINE_ERROR;
    if (implicit_factor) {
      memcpy(y, adj_qacc, (size_t)nv*sizeof(mjtNum));
      mj_solveLD(y, backward.qH, backward.qHDiagInv, nv, 1,
                 m->M_rownnz, m->M_rowadr, m->M_colind, NULL);
      mj_mulM(m, &backward, contact_mu, y);
    } else {
      memcpy(contact_mu, adj_qacc, (size_t)nv*sizeof(mjtNum));
    }
    mju_cholSolve(contact_solution, contact_factor, contact_mu, nv);
    mj_mulJacVec(m, &backward, contact_q_data, contact_solution);
    if (cone_contact) {
      for (int row = 0; row < nefc; ++row) {
        contact_z[row] = mju_dot(contact_hessian+row*nefc, contact_q_data, nefc);
      }
      for (int row = 0; row < nefc; ++row) {
        contact_mu[row] = -contact_z[row];
      }
      contact_q = contact_q_data;
      mju_zero(y_vel, nv);
      mj_mulJacTVec(m, &backward, y_vel, contact_mu);
      if (implicit_factor) {
        memcpy(contact_y, contact_solution, (size_t)nv*sizeof(mjtNum));
        for (int i = 0; i < nv; ++i) contact_z[i] = contact_y[i]-y[i];
      } else {
        for (int i = 0; i < nv; ++i) contact_y[i] = adj_qacc[i]+y_vel[i];
      }
    } else {
      for (int row = 0; row < nefc; ++row) {
        if (d->efc_state[row] == mjCNSTRSTATE_QUADRATIC) {
          contact_q_data[row] *= d->efc_D[row];
          contact_mu[row] = -d->efc_D[row]*contact_q_data[row];
        } else {
          contact_q_data[row] = contact_mu[row] = 0;
        }
      }
    }
    } else {
      size_t matrix_size = (size_t)nefc*nefc;
      mju_zero(contact_ar, matrix_size);
      for (int row = 0; row < nefc; ++row) {
        mju_zero(contact_mu, width);
        contact_mu[row] = 1;
        mju_zero(y_vel, nv);
        mj_mulJacTVec(m, &backward, y_vel, contact_mu);
        memcpy(contact_z, y_vel, (size_t)nv*sizeof(mjtNum));
        mj_solveLD(contact_z, backward.qLD, backward.qLDiagInv, nv, 1,
                   m->M_rownnz, m->M_rowadr, m->M_colind, NULL);
        mj_mulJacVec(m, &backward, contact_mu, contact_z);
        for (int col = 0; col < nefc; ++col) contact_ar[row*nefc+col] += contact_mu[col];
        contact_ar[row*nefc+row] += d->efc_R[row];
      }
      for (int row = 0; row < nefc; ++row) {
        if (d->efc_state[row] == mjCNSTRSTATE_QUADRATIC) continue;
        for (int col = 0; col < nefc; ++col) {
          contact_ar[row*nefc+col] = row == col;
          contact_ar[col*nefc+row] = row == col;
        }
      }
      const mjtNum* factor = implicit_factor ? backward.qH : backward.qLD;
      const mjtNum* diag_inv = implicit_factor ? backward.qHDiagInv : backward.qLDiagInv;
      memcpy(y, adj_qacc, (size_t)nv*sizeof(mjtNum));
      mj_solveLD(y, factor, diag_inv, nv, 1,
                 m->M_rownnz, m->M_rowadr, m->M_colind, NULL);
      mj_mulJacVec(m, &backward, contact_mu, y);
      memcpy(contact_factor, contact_ar, matrix_size*sizeof(mjtNum));
      if (!mju_factorLU(contact_factor, nefc, ctx->tape_ints)) return mjADJ_ENGINE_ERROR;
      mju_solveLU(contact_solution, contact_factor, contact_mu, ctx->tape_ints, nefc);
      memcpy(contact_mu, contact_solution, (size_t)nefc*sizeof(mjtNum));
      for (int row = 0; row < nefc; ++row) {
        contact_mu[row] = d->efc_state[row] == mjCNSTRSTATE_SATISFIED ? 0 : -contact_mu[row];
      }
      mju_zero(y_vel, nv);
      mj_mulJacTVec(m, &backward, y_vel, contact_mu);
      if (implicit_factor) {
        memcpy(contact_z, y_vel, (size_t)nv*sizeof(mjtNum));
        mj_solveLD(contact_z, backward.qLD, backward.qLDiagInv, nv, 1,
                   m->M_rownnz, m->M_rowadr, m->M_colind, NULL);
        for (int i = 0; i < nv; ++i) contact_y[i] = y[i]+contact_z[i];
      } else {
        for (int i = 0; i < nv; ++i) contact_y[i] = adj_qacc[i]+y_vel[i];
      }
    }
  }
  const mjtNum* smooth_bar = d->ncon && !implicit_factor ? contact_y : adj_qacc;
  mjData factor_backward = backward;
  factor_backward.qacc_smooth = qacc_adv;
  if (implicit_factor) {
    factor_backward.qLD = backward.qH;
    factor_backward.qLDiagInv = backward.qHDiagInv;
  }
  if (free_body) {
    status = mjd_freeSmoothAccPosAdjoint(m, &factor_backward, smooth_bar, y, ctx->grad_qpos);
  } else {
    status = mjd_fixedSmoothAccPosAdjoint(m, &factor_backward, smooth_bar, y, ctx->grad_qpos);
  }
  if (status != mjADJ_OK) return status;
  if (d->ncon && implicit_factor) {
    status = mjd_massPosAdjoint(m, &backward, contact_z, backward.qacc, extra_pos);
    if (status == mjADJ_OK) {
      for (int i = 0; i < m->nv; ++i) ctx->grad_qpos[i] -= extra_pos[i];
      status = mjd_rneBiasPosAdjoint(m, &backward, contact_z, extra_pos);
    }
    if (status == mjADJ_OK) {
      for (int i = 0; i < m->nv; ++i) ctx->grad_qpos[i] -= extra_pos[i];
      status = mjd_passiveStateAdjoint(m, &backward, contact_z, extra_pos, extra_vel);
    }
    if (status == mjADJ_OK) {
      for (int i = 0; i < m->nv; ++i) ctx->grad_qpos[i] += extra_pos[i];
      status = mjd_actuatorStateAdjoint(m, &backward, contact_z, extra_pos, extra_vel);
    }
    if (status != mjADJ_OK) return status;
    for (int i = 0; i < m->nv; ++i) ctx->grad_qpos[i] += extra_pos[i];
  }
  if (d->ncon) {
    const mjtNum* pos_bar = implicit_factor ? contact_y : y;
    const mjtNum* solver_qacc = implicit_factor ? backward.qacc : qacc_adv;
    status = pair_kind == 3 ?
        sphere_cylinder_pos_adjoint(m, &backward, pos_bar, contact_lambda, contact_mu,
                                    contact_q, solver_qacc, -1, ctx->grad_qpos) :
        pair_kind == 2 ?
        sphere_box_face_pos_adjoint(m, &backward, pos_bar, contact_lambda, contact_mu,
                                    contact_q, solver_qacc, -1, ctx->grad_qpos) :
        pair_kind == 1 ?
        mjd_capsuleCapsuleStepPosAdjoint(m, &backward, pos_bar, contact_lambda, contact_mu,
                                         contact_q, solver_qacc, ctx->grad_qpos) :
        mjd_planeSphereStepPosAdjoint(m, &backward, pos_bar, contact_lambda, contact_mu,
                                      contact_q, solver_qacc, ctx->grad_qpos);
    if (status != mjADJ_OK) return status;
  }
  const mjtNum* force_bar = d->ncon && implicit_factor ? contact_y : y;
  if (m->nu) {
    status = mjd_actuatorCtrlAdjoint(m, &backward, force_bar, ctx->grad_ctrl);
    if (status != mjADJ_OK) return status;
  }
  if (implicit_factor) {
    status = mjd_smoothVelAdjoint(m, &backward, force_bar, ctx->sparse_scratch, ctx->grad_qvel);
  } else {
    status = mjd_smoothAccVelAdjoint(m, &backward, smooth_bar, ctx->sparse_scratch,
                                     y_vel, ctx->grad_qvel);
  }
  if (status != mjADJ_OK) return status;
  if (d->ncon) {
    for (int row = 0; row < d->nefc; ++row) contact_mu[row] *= d->efc_KBIP[4*row+1];
    mju_zero(extra_vel, m->nv);
    mj_mulJacTVec(m, &backward, extra_vel, contact_mu);
    for (int i = 0; i < m->nv; ++i) ctx->grad_qvel[i] += extra_vel[i];
  }
  for (int i = 0; i < m->nv; ++i) {
    ctx->grad_qpos[i] += integrate_pos[i];
    ctx->grad_qvel[i] += ctx->bar_qvel[i] + integrate_vel[i];
  }
  memcpy(dbar_qpos_in, ctx->grad_qpos, (size_t)m->nv*sizeof(mjtNum));
  memcpy(dbar_qvel_in, ctx->grad_qvel, (size_t)m->nv*sizeof(mjtNum));
  if (m->nu) memcpy(dbar_ctrl, ctx->grad_ctrl, (size_t)m->nu*sizeof(mjtNum));
  return mjADJ_OK;
}

int mjd_actuatorCtrlAdjoint(const mjModel* m, const mjData* d,
                            const mjtNum* bar_qfrc_actuator, mjtNum* dbar_ctrl) {
  if (!m || !d || (m->nv && !bar_qfrc_actuator) || (m->nu && !dbar_ctrl)) {
    return mjADJ_INVALID_ARGUMENT;
  }
  int model_status = mjd_checkAdjointModel(m);
  if (model_status != mjADJ_OK) {
    return model_status;
  }
  if (ranges_overlap(bar_qfrc_actuator, (size_t)m->nv * sizeof(mjtNum),
                     dbar_ctrl, (size_t)m->nu * sizeof(mjtNum))) {
    return mjADJ_INVALID_ARGUMENT;
  }
  memset(dbar_ctrl, 0, (size_t)m->nu * sizeof(mjtNum));
  if (mjDISABLED(mjDSBL_ACTUATION)) {
    return mjADJ_OK;
  }
  for (int i = 0; i < m->nactuator; ++i) {
    int group = m->actuator_group[i];
    if ((group >= 0 && group <= 30 && (m->opt.disableactuator & (1 << group)))) {
      continue;
    }
    int uadr = m->actuator_ctrladr[i];
    int oadr = m->actuator_outadr[i];
    int joint = m->actuator_trnid[2*i];
    int dof = m->jnt_dofadr[joint];
    if (m->actuator_ctrlnum[i] != 1 || m->actuator_outnum[i] != 1 ||
        uadr < 0 || uadr >= m->nu || oadr < 0 || oadr >= m->nout) {
      return mjADJ_UNSUPPORTED;
    }
    mjtNum ctrl = d->ctrl[uadr];
    mjtNum ctrl_scale = 1;
    if (!mjDISABLED(mjDSBL_CLAMPCTRL) && m->actuator_ctrllimited[uadr]) {
      const mjtNum* range = m->actuator_ctrlrange + 2*uadr;
      if (ctrl <= range[0] || ctrl >= range[1]) {
        ctrl_scale = 0;
      }
      ctrl = mju_clip(ctrl, range[0], range[1]);
    }
    const mjtNum* biasprm = m->actuator_biasprm + mjNBIAS*i;
    mjtNum bias = 0;
    if (m->actuator_biastype[i] == mjBIAS_AFFINE) {
      bias = biasprm[0] + biasprm[1]*d->actuator_length[oadr] +
             biasprm[2]*d->actuator_velocity[oadr];
    }
    mjtNum force = m->actuator_gainprm[mjNGAIN*i]*ctrl + bias;
    if (m->actuator_forcelimited[i]) {
      const mjtNum* range = m->actuator_forcerange + 2*i;
      if (force <= range[0] || force >= range[1]) {
        continue;
      }
    }
    if (m->jnt_actfrclimited[joint]) {
      const mjtNum* range = m->jnt_actfrcrange + 2*joint;
      if (d->qfrc_actuator[dof] <= range[0] || d->qfrc_actuator[dof] >= range[1]) {
        continue;
      }
    }
    mjtNum lambda = 0;
    int rownnz = d->moment_rownnz[oadr];
    int rowadr = d->moment_rowadr[oadr];
    for (int k = 0; k < rownnz; ++k) {
      lambda += d->actuator_moment[rowadr+k]*bar_qfrc_actuator[d->moment_colind[rowadr+k]];
    }
    dbar_ctrl[uadr] += lambda*m->actuator_gainprm[mjNGAIN*i]*ctrl_scale;
  }
  return mjADJ_OK;
}

int mjd_actuatorStateAdjoint(const mjModel* m, const mjData* d,
                             const mjtNum* bar_qfrc_actuator,
                             mjtNum* dbar_qpos, mjtNum* dbar_qvel) {
  if (!m || !d || (m->nv && (!bar_qfrc_actuator || !dbar_qpos || !dbar_qvel))) {
    return mjADJ_INVALID_ARGUMENT;
  }
  int model_status = mjd_checkAdjointModel(m);
  if (model_status != mjADJ_OK) {
    return model_status;
  }
  size_t tangent_bytes = (size_t)m->nv * sizeof(mjtNum);
  const mjtNum* inputs[] = {bar_qfrc_actuator, dbar_qpos, dbar_qvel};
  for (int i = 0; i < 3; ++i) {
    for (int j = i + 1; j < 3; ++j) {
      if (ranges_overlap(inputs[i], tangent_bytes, inputs[j], tangent_bytes)) {
        return mjADJ_INVALID_ARGUMENT;
      }
    }
  }
  memset(dbar_qpos, 0, tangent_bytes);
  memset(dbar_qvel, 0, tangent_bytes);
  if (mjDISABLED(mjDSBL_ACTUATION)) {
    return mjADJ_OK;
  }
  for (int i = 0; i < m->nactuator; ++i) {
    int group = m->actuator_group[i];
    if ((group >= 0 && group <= 30 && (m->opt.disableactuator & (1 << group))) ||
        m->actuator_biastype[i] == mjBIAS_NONE) {
      continue;
    }
    int uadr = m->actuator_ctrladr[i];
    int oadr = m->actuator_outadr[i];
    int joint = m->actuator_trnid[2*i];
    int dof = m->jnt_dofadr[joint];
    mjtNum ctrl = d->ctrl[uadr];
    if (!mjDISABLED(mjDSBL_CLAMPCTRL) && m->actuator_ctrllimited[uadr]) {
      const mjtNum* range = m->actuator_ctrlrange + 2*uadr;
      ctrl = mju_clip(ctrl, range[0], range[1]);
    }
    const mjtNum* biasprm = m->actuator_biasprm + mjNBIAS*i;
    mjtNum bias = biasprm[0] + biasprm[1]*d->actuator_length[oadr] +
                  biasprm[2]*d->actuator_velocity[oadr];
    mjtNum force = m->actuator_gainprm[mjNGAIN*i]*ctrl + bias;
    if (m->actuator_forcelimited[i]) {
      const mjtNum* range = m->actuator_forcerange + 2*i;
      if (force <= range[0] || force >= range[1]) {
        continue;
      }
    }
    if (m->jnt_actfrclimited[joint]) {
      const mjtNum* range = m->jnt_actfrcrange + 2*joint;
      if (d->qfrc_actuator[dof] <= range[0] || d->qfrc_actuator[dof] >= range[1]) {
        continue;
      }
    }
    mjtNum lambda = 0;
    int rownnz = d->moment_rownnz[oadr];
    int rowadr = d->moment_rowadr[oadr];
    for (int k = 0; k < rownnz; ++k) {
      lambda += d->actuator_moment[rowadr+k]*bar_qfrc_actuator[d->moment_colind[rowadr+k]];
    }
    mjtNum gear = m->actuator_gear[6*oadr];
    dbar_qpos[dof] += lambda*biasprm[1]*gear;
    dbar_qvel[dof] += lambda*biasprm[2]*gear;
  }
  return mjADJ_OK;
}

int mjd_passiveStateAdjoint(const mjModel* m, const mjData* d,
                            const mjtNum* bar_qfrc_passive,
                            mjtNum* dbar_qpos, mjtNum* dbar_qvel) {
  if (!m || !d || (m->nv && (!bar_qfrc_passive || !dbar_qpos || !dbar_qvel))) {
    return mjADJ_INVALID_ARGUMENT;
  }
  int model_status = mjd_checkAdjointModel(m);
  if (model_status != mjADJ_OK || m->flg_gravcomp) {
    return mjADJ_UNSUPPORTED;
  }
  size_t tangent_bytes = (size_t)m->nv * sizeof(mjtNum);
  const mjtNum* inputs[] = {bar_qfrc_passive, dbar_qpos, dbar_qvel};
  for (int i = 0; i < 3; ++i) {
    for (int j = i + 1; j < 3; ++j) {
      if (ranges_overlap(inputs[i], tangent_bytes, inputs[j], tangent_bytes)) {
        return mjADJ_INVALID_ARGUMENT;
      }
    }
  }
  for (int joint = 0; joint < m->njnt; ++joint) {
    const mjtNum* poly = m->jnt_stiffnesspoly + mjNPOLY*joint;
    if (poly[0] != 0 || poly[1] != 0 ||
        (m->jnt_stiffness[joint] != 0 && m->jnt_type[joint] != mjJNT_HINGE &&
         m->jnt_type[joint] != mjJNT_SLIDE)) {
      return mjADJ_UNSUPPORTED;
    }
  }
  for (int dof = 0; dof < m->nv; ++dof) {
    mjtNum poly[mjNPOLY] = {m->dof_dampingpoly[mjNPOLY*dof],
                            m->dof_dampingpoly[mjNPOLY*dof+1]};
    mjtNum damping = m->dof_damping[dof] +
                     mj_actuatorDamping(m, mjOBJ_JOINT, m->dof_jntid[dof], poly);
    int joint_type = m->jnt_type[m->dof_jntid[dof]];
    if (poly[0] != 0 || poly[1] != 0 ||
        (damping != 0 && joint_type != mjJNT_HINGE && joint_type != mjJNT_SLIDE)) {
      return mjADJ_UNSUPPORTED;
    }
  }
  memset(dbar_qpos, 0, tangent_bytes);
  memset(dbar_qvel, 0, tangent_bytes);
  if (!mjDISABLED(mjDSBL_SPRING)) {
    for (int joint = 0; joint < m->njnt; ++joint) {
      int dof = m->jnt_dofadr[joint];
      dbar_qpos[dof] -= bar_qfrc_passive[dof]*m->jnt_stiffness[joint];
    }
  }
  if (!mjDISABLED(mjDSBL_DAMPER)) {
    for (int dof = 0; dof < m->nv; ++dof) {
      mjtNum poly[mjNPOLY] = {m->dof_dampingpoly[mjNPOLY*dof],
                              m->dof_dampingpoly[mjNPOLY*dof+1]};
      mjtNum damping = m->dof_damping[dof] +
                       mj_actuatorDamping(m, mjOBJ_JOINT, m->dof_jntid[dof], poly);
      dbar_qvel[dof] -= bar_qfrc_passive[dof]*damping;
    }
  }
  return mjADJ_OK;
}

int mjd_rneBiasVelAdjoint(const mjModel* m, mjData* d,
                          const mjtNum* bar_qfrc_bias,
                          mjtNum* qDeriv, mjtNum* dbar_qvel) {
  if (!m || !d || (m->nv && (!bar_qfrc_bias || !dbar_qvel)) || (m->nD && !qDeriv)) {
    return mjADJ_INVALID_ARGUMENT;
  }
  int model_status = mjd_checkAdjointModel(m);
  if (model_status != mjADJ_OK) {
    return model_status;
  }
  size_t tangent_bytes = (size_t)m->nv * sizeof(mjtNum);
  size_t derivative_bytes = (size_t)m->nD * sizeof(mjtNum);
  if (ranges_overlap(bar_qfrc_bias, tangent_bytes, dbar_qvel, tangent_bytes) ||
      ranges_overlap(bar_qfrc_bias, tangent_bytes, qDeriv, derivative_bytes) ||
      ranges_overlap(dbar_qvel, tangent_bytes, qDeriv, derivative_bytes)) {
    return mjADJ_INVALID_ARGUMENT;
  }
  memset(qDeriv, 0, derivative_bytes);
  memset(dbar_qvel, 0, tangent_bytes);
  mjd_rne_vel(m, d, qDeriv);
  for (int row = 0; row < m->nv; ++row) {
    int rowadr = m->D_rowadr[row];
    for (int k = 0; k < m->D_rownnz[row]; ++k) {
      dbar_qvel[m->D_colind[rowadr+k]] -= qDeriv[rowadr+k]*bar_qfrc_bias[row];
    }
  }
  return mjADJ_OK;
}

int mjd_integratePosAdjoint(const mjModel* m, const mjtNum* qvel, mjtNum dt,
                            const mjtNum* dbar_qpos_out, mjtNum* dbar_qpos_in,
                            mjtNum* dbar_qvel) {
  if (!m || (m->nv && (!qvel || !dbar_qpos_out || !dbar_qpos_in || !dbar_qvel))) {
    return mjADJ_INVALID_ARGUMENT;
  }
  size_t tangent_bytes = (size_t)m->nv * sizeof(mjtNum);
  const mjtNum* buffers[] = {qvel, dbar_qpos_out, dbar_qpos_in, dbar_qvel};
  for (int i = 0; i < 4; ++i) {
    for (int j = i + 1; j < 4; ++j) {
      if (ranges_overlap(buffers[i], tangent_bytes, buffers[j], tangent_bytes)) {
        return mjADJ_INVALID_ARGUMENT;
      }
    }
  }
  for (int joint = 0; joint < m->njnt; ++joint) {
    int dof = m->jnt_dofadr[joint];
    if (m->jnt_type[joint] == mjJNT_FREE) {
      for (int k = 0; k < 3; ++k) {
        dbar_qpos_in[dof+k] = dbar_qpos_out[dof+k];
        dbar_qvel[dof+k] = dt*dbar_qpos_out[dof+k];
      }
    } else if (m->jnt_type[joint] == mjJNT_HINGE || m->jnt_type[joint] == mjJNT_SLIDE) {
      dbar_qpos_in[dof] = dbar_qpos_out[dof];
      dbar_qvel[dof] = dt*dbar_qpos_out[dof];
    }
    if (m->jnt_type[joint] == mjJNT_BALL || m->jnt_type[joint] == mjJNT_FREE) {
      int rotation = dof + 3*(m->jnt_type[joint] == mjJNT_FREE);
      mjtNum Dquat[9];
      mjtNum Dvel[9];
      mjd_quatIntegrate(qvel+rotation, dt, Dquat, Dvel, NULL);
      for (int row = 0; row < 3; ++row) {
        dbar_qpos_in[rotation+row] = 0;
        dbar_qvel[rotation+row] = 0;
        for (int col = 0; col < 3; ++col) {
          dbar_qpos_in[rotation+row] += Dquat[3*col+row]*dbar_qpos_out[rotation+col];
          dbar_qvel[rotation+row] += dt*Dvel[3*col+row]*dbar_qpos_out[rotation+col];
        }
      }
    }
  }
  return mjADJ_OK;
}

int mjd_implicitFastSolveAdjoint(const mjModel* m, const mjData* d,
                                 const mjtNum* bar, mjtNum* y) {
  if (!m || !d || (m->nv && (!bar || !y || !d->qH || !d->qHDiagInv))) {
    return mjADJ_INVALID_ARGUMENT;
  }
  int model_status = mjd_checkAdjointModel(m);
  if (model_status != mjADJ_OK || m->opt.integrator != mjINT_IMPLICITFAST || d->efm_active) {
    return mjADJ_UNSUPPORTED;
  }
  size_t tangent_bytes = (size_t)m->nv * sizeof(mjtNum);
  if (bar != y && ranges_overlap(bar, tangent_bytes, y, tangent_bytes)) {
    return mjADJ_INVALID_ARGUMENT;
  }
  if (bar != y) {
    memcpy(y, bar, tangent_bytes);
  }
  mj_solveLD(y, d->qH, d->qHDiagInv, m->nv, 1,
             m->M_rownnz, m->M_rowadr, m->M_colind, NULL);
  return mjADJ_OK;
}

int mjd_smoothVelAdjoint(const mjModel* m, const mjData* d,
                         const mjtNum* bar_qfrc_smooth,
                         mjtNum* qDeriv, mjtNum* dbar_qvel) {
  if (!m || !d || (m->nv && (!bar_qfrc_smooth || !dbar_qvel)) || (m->nD && !qDeriv)) {
    return mjADJ_INVALID_ARGUMENT;
  }
  int model_status = mjd_checkAdjointModel(m);
  if (model_status != mjADJ_OK) {
    return model_status;
  }
  size_t tangent_bytes = (size_t)m->nv * sizeof(mjtNum);
  size_t derivative_bytes = (size_t)m->nD * sizeof(mjtNum);
  if (ranges_overlap(bar_qfrc_smooth, tangent_bytes, dbar_qvel, tangent_bytes) ||
      ranges_overlap(bar_qfrc_smooth, tangent_bytes, qDeriv, derivative_bytes) ||
      ranges_overlap(dbar_qvel, tangent_bytes, qDeriv, derivative_bytes)) {
    return mjADJ_INVALID_ARGUMENT;
  }
  memset(dbar_qvel, 0, tangent_bytes);
  mjData derivative_data = *d;
  derivative_data.qDeriv = qDeriv;
  mjd_smooth_vel(m, &derivative_data, 1);
  for (int row = 0; row < m->nv; ++row) {
    int rowadr = m->D_rowadr[row];
    for (int k = 0; k < m->D_rownnz[row]; ++k) {
      dbar_qvel[m->D_colind[rowadr+k]] += qDeriv[rowadr+k]*bar_qfrc_smooth[row];
    }
  }
  return mjADJ_OK;
}

int mjd_smoothAccVelAdjoint(const mjModel* m, const mjData* d,
                            const mjtNum* bar_qacc_smooth,
                            mjtNum* qDeriv, mjtNum* y, mjtNum* dbar_qvel) {
  if (!m || !d || (m->nv && (!bar_qacc_smooth || !y || !dbar_qvel)) || (m->nD && !qDeriv)) {
    return mjADJ_INVALID_ARGUMENT;
  }
  int model_status = mjd_checkAdjointModel(m);
  if (model_status != mjADJ_OK || m->opt.integrator != mjINT_EULER || d->efm_active) {
    return mjADJ_UNSUPPORTED;
  }
  size_t tangent_bytes = (size_t)m->nv * sizeof(mjtNum);
  size_t derivative_bytes = (size_t)m->nD * sizeof(mjtNum);
  const mjtNum* buffers[] = {bar_qacc_smooth, qDeriv, y, dbar_qvel};
  size_t sizes[] = {tangent_bytes, derivative_bytes, tangent_bytes, tangent_bytes};
  for (int i = 0; i < 4; ++i) {
    for (int j = i + 1; j < 4; ++j) {
      if (ranges_overlap(buffers[i], sizes[i], buffers[j], sizes[j])) {
        return mjADJ_INVALID_ARGUMENT;
      }
    }
  }
  memcpy(y, bar_qacc_smooth, tangent_bytes);
  mj_solveLD(y, d->qLD, d->qLDiagInv, m->nv, 1,
             m->M_rownnz, m->M_rowadr, m->M_colind, NULL);
  return mjd_smoothVelAdjoint(m, d, y, qDeriv, dbar_qvel);
}

int mjd_smoothAccCtrlAdjoint(const mjModel* m, const mjData* d,
                             const mjtNum* bar_qacc_smooth,
                             mjtNum* y, mjtNum* dbar_ctrl) {
  if (!m || !d || (m->nv && (!bar_qacc_smooth || !y)) || (m->nu && !dbar_ctrl)) {
    return mjADJ_INVALID_ARGUMENT;
  }
  int model_status = mjd_checkAdjointModel(m);
  if (model_status != mjADJ_OK || m->opt.integrator != mjINT_EULER || d->efm_active) {
    return mjADJ_UNSUPPORTED;
  }
  size_t tangent_bytes = (size_t)m->nv * sizeof(mjtNum);
  size_t ctrl_bytes = (size_t)m->nu * sizeof(mjtNum);
  if (ranges_overlap(bar_qacc_smooth, tangent_bytes, y, tangent_bytes) ||
      ranges_overlap(bar_qacc_smooth, tangent_bytes, dbar_ctrl, ctrl_bytes) ||
      ranges_overlap(y, tangent_bytes, dbar_ctrl, ctrl_bytes)) {
    return mjADJ_INVALID_ARGUMENT;
  }
  memcpy(y, bar_qacc_smooth, tangent_bytes);
  mj_solveLD(y, d->qLD, d->qLDiagInv, m->nv, 1,
             m->M_rownnz, m->M_rowadr, m->M_colind, NULL);
  return mjd_actuatorCtrlAdjoint(m, d, y, dbar_ctrl);
}

static void body_inertia_mul(const mjModel* m, const mjData* d, int body,
                             const mjtNum vector[3], mjtNum result[3]) {
  const mjtNum* rotation = d->ximat + 9*body;
  mjtNum local[3];
  mju_mulMatTVec3(local, rotation, vector);
  for (int row = 0; row < 3; ++row) {
    local[row] *= m->body_inertia[3*body+row];
  }
  mju_mulMatVec3(result, rotation, local);
}

int mjd_rneBiasPosAdjoint(const mjModel* m, mjData* d,
                          const mjtNum* bar_qfrc_bias, mjtNum* dbar_qpos) {
  if (!m || !d || (m->nv && (!bar_qfrc_bias || !dbar_qpos))) {
    return mjADJ_INVALID_ARGUMENT;
  }
  if (!scalar_tree_kind(m)) return mjADJ_UNSUPPORTED;
  int free_root = scalar_tree_kind(m) == 2;
  size_t tangent_bytes = (size_t)m->nv * sizeof(mjtNum);
  if (ranges_overlap(bar_qfrc_bias, tangent_bytes, dbar_qpos, tangent_bytes)) {
    return mjADJ_INVALID_ARGUMENT;
  }
  mju_zero(dbar_qpos, m->nv);
  mj_markStack(d);
  mjtNum* storage = mjSTACKALLOC(d, 51*m->nbody, mjtNum);
  mju_zero(storage, 51*m->nbody);
  mjtNum* omega = storage;
  mjtNum* velocity = storage+3*m->nbody;
  mjtNum* alpha = storage+6*m->nbody;
  mjtNum* acc = storage+9*m->nbody;
  mjtNum* axes = storage+12*m->nbody;
  mjtNum* force = storage+15*m->nbody;
  mjtNum* moment = storage+18*m->nbody;
  mjtNum* delta_pos = storage+21*m->nbody;
  mjtNum* delta_omega = storage+24*m->nbody;
  mjtNum* delta_velocity = storage+27*m->nbody;
  mjtNum* delta_alpha = storage+30*m->nbody;
  mjtNum* delta_acc = storage+33*m->nbody;
  mjtNum* inertia_alpha = storage+36*m->nbody;
  mjtNum* inertia_omega = storage+39*m->nbody;
  mjtNum* com = storage+42*m->nbody;
  mjtNum* delta_force = storage+45*m->nbody;
  mjtNum* delta_moment = storage+48*m->nbody;
  if (!mjDISABLED(mjDSBL_GRAVITY)) {
    mju_scl3(acc, m->opt.gravity, -1);
  }

  for (int body = 1; body < m->nbody; ++body) {
    int parent = m->body_parentid[body];
    mjtNum offset[3], tmp[3], tmp1[3];
    mju_sub3(offset, d->xpos+3*body, d->xpos+3*parent);
    mju_copy(omega+3*body, omega+3*parent, 3);
    mju_copy(alpha+3*body, alpha+3*parent, 3);
    mju_cross(velocity+3*body, omega+3*parent, offset);
    mju_addTo3(velocity+3*body, velocity+3*parent);
    mju_cross(acc+3*body, alpha+3*parent, offset);
    mju_addTo3(acc+3*body, acc+3*parent);
    mju_cross(tmp, omega+3*parent, offset);
    mju_cross(tmp1, omega+3*parent, tmp);
    mju_addTo3(acc+3*body, tmp1);
    if (free_root && body == 1) {
      mju_mulMatVec3(omega+3*body, d->xmat+9*body, d->qvel+3);
      mju_copy(velocity+3*body, d->qvel, 3);
    }
    if (m->body_jntnum[body] && !(free_root && body == 1)) {
      int joint = m->body_jntadr[body];
      int dof = m->jnt_dofadr[joint];
      mju_mulMatVec3(axes+3*body, d->xmat+9*body, m->jnt_axis+3*joint);
      mju_scl3(tmp, axes+3*body, d->qvel[dof]);
      if (m->jnt_type[joint] == mjJNT_HINGE) {
        mju_addTo3(omega+3*body, tmp);
        mju_cross(tmp1, omega+3*parent, tmp);
        mju_addTo3(alpha+3*body, tmp1);
      } else {
        mju_addTo3(velocity+3*body, tmp);
        mju_cross(tmp1, omega+3*parent, tmp);
        mju_scl3(tmp1, tmp1, 2);
        mju_addTo3(acc+3*body, tmp1);
      }
    }
    mju_mulMatVec3(com+3*body, d->xmat+9*body, m->body_ipos+3*body);
    mju_cross(tmp, alpha+3*body, com+3*body);
    mju_cross(tmp1, omega+3*body, com+3*body);
    mjtNum centrifugal[3];
    mju_cross(centrifugal, omega+3*body, tmp1);
    for (int k = 0; k < 3; ++k) {
      force[3*body+k] = m->body_mass[body]*(acc[3*body+k]+tmp[k]+centrifugal[k]);
    }
    body_inertia_mul(m, d, body, alpha+3*body, inertia_alpha+3*body);
    body_inertia_mul(m, d, body, omega+3*body, inertia_omega+3*body);
    mju_cross(moment+3*body, omega+3*body, inertia_omega+3*body);
    mju_addTo3(moment+3*body, inertia_alpha+3*body);
    mju_cross(tmp, com+3*body, force+3*body);
    mju_addTo3(moment+3*body, tmp);
  }
  mjtNum* local_force = force;

  for (int joint = 0; joint < m->njnt; ++joint) {
    int joint_body = m->jnt_bodyid[joint];
    int free_joint = m->jnt_type[joint] == mjJNT_FREE;
    int channels = free_joint ? 6 : 1;
    for (int channel = 0; channel < channels; ++channel) {
      int dof = m->jnt_dofadr[joint]+channel;
      int rotating = m->jnt_type[joint] == mjJNT_HINGE ||
                     (free_joint && channel >= 3);
      mjtNum free_axis[3] = {0, 0, 0};
      const mjtNum* perturb_axis = axes+3*joint_body;
      if (free_joint) {
        if (channel < 3) {
          free_axis[channel] = 1;
        } else {
          mjtNum local_axis[3] = {0, 0, 0};
          local_axis[channel-3] = 1;
          mju_mulMatVec3(free_axis, d->xmat+9*joint_body, local_axis);
        }
        perturb_axis = free_axis;
      }
      mju_zero(delta_pos, 15*m->nbody);
    mju_zero(delta_force, 6*m->nbody);
    for (int body = joint_body; body < m->nbody; ++body) {
      int ancestor = body;
      while (ancestor != joint_body && ancestor > 0) {
        ancestor = m->body_parentid[ancestor];
      }
      if (ancestor != joint_body) continue;
      if (rotating) {
        mjtNum offset[3];
        mju_sub3(offset, d->xpos+3*body, d->xpos+3*joint_body);
        mju_cross(delta_pos+3*body, perturb_axis, offset);
      } else {
        mju_copy(delta_pos+3*body, perturb_axis, 3);
      }
    }

    for (int body = 1; body < m->nbody; ++body) {
      int parent = m->body_parentid[body];
      int ancestor = body;
      while (ancestor != joint_body && ancestor > 0) {
        ancestor = m->body_parentid[ancestor];
      }
      if (ancestor != joint_body) continue;
      int body_joint = m->body_jntnum[body] ? m->body_jntadr[body] : -1;
      if (free_root && body == 1) body_joint = -1;
      int body_dof = body_joint >= 0 ? m->jnt_dofadr[body_joint] : -1;
      int body_rotating = body_joint >= 0 && m->jnt_type[body_joint] == mjJNT_HINGE;
      mjtNum delta_axis[3] = {0, 0, 0};
      if (rotating && body_joint >= 0) {
        mju_cross(delta_axis, perturb_axis, axes+3*body);
      }
      mjtNum offset[3], delta_offset[3], tmp[3], tmp1[3], tmp2[3], tmp3[3];
      mju_sub3(offset, d->xpos+3*body, d->xpos+3*parent);
      mju_sub3(delta_offset, delta_pos+3*body, delta_pos+3*parent);
      mju_copy(delta_omega+3*body, delta_omega+3*parent, 3);
      mju_copy(delta_alpha+3*body, delta_alpha+3*parent, 3);
      mju_cross(delta_velocity+3*body, delta_omega+3*parent, offset);
      mju_cross(tmp, omega+3*parent, delta_offset);
      mju_addTo3(delta_velocity+3*body, tmp);
      mju_addTo3(delta_velocity+3*body, delta_velocity+3*parent);
      mju_cross(delta_acc+3*body, delta_alpha+3*parent, offset);
      mju_cross(tmp, alpha+3*parent, delta_offset);
      mju_addTo3(delta_acc+3*body, tmp);
      mju_addTo3(delta_acc+3*body, delta_acc+3*parent);
      mju_cross(tmp, omega+3*parent, offset);
      mju_cross(tmp1, delta_omega+3*parent, tmp);
      mju_cross(tmp2, delta_omega+3*parent, offset);
      mju_cross(tmp3, omega+3*parent, tmp2);
      mju_addTo3(delta_acc+3*body, tmp1);
      mju_addTo3(delta_acc+3*body, tmp3);
      mju_cross(tmp, omega+3*parent, delta_offset);
      mju_cross(tmp1, omega+3*parent, tmp);
      mju_addTo3(delta_acc+3*body, tmp1);
      if (body_joint >= 0) {
        mju_scl3(tmp, delta_axis, d->qvel[body_dof]);
        mju_scl3(tmp1, axes+3*body, d->qvel[body_dof]);
        if (body_rotating) {
          mju_addTo3(delta_omega+3*body, tmp);
          mju_cross(tmp2, delta_omega+3*parent, tmp1);
          mju_addTo3(delta_alpha+3*body, tmp2);
          mju_cross(tmp2, omega+3*parent, tmp);
          mju_addTo3(delta_alpha+3*body, tmp2);
        } else {
          mju_addTo3(delta_velocity+3*body, tmp);
          mju_cross(tmp2, delta_omega+3*parent, tmp1);
          mju_addTo3(delta_acc+3*body, tmp2);
          mju_addTo3(delta_acc+3*body, tmp2);
          mju_cross(tmp2, omega+3*parent, tmp);
          mju_addTo3(delta_acc+3*body, tmp2);
          mju_addTo3(delta_acc+3*body, tmp2);
        }
      }
      if (free_root && body == 1 && rotating) {
        mju_cross(delta_omega+3*body, perturb_axis, omega+3*body);
      }

      mjtNum delta_com[3] = {0, 0, 0};
      if (rotating) mju_cross(delta_com, perturb_axis, com+3*body);
      mjtNum* local_delta_force = delta_force+3*body;
      mjtNum* local_delta_moment = delta_moment+3*body;
      mju_cross(tmp, delta_alpha+3*body, com+3*body);
      mju_cross(tmp1, alpha+3*body, delta_com);
      mju_add3(local_delta_force, tmp, tmp1);
      mju_cross(tmp, delta_omega+3*body, com+3*body);
      mju_cross(tmp1, omega+3*body, com+3*body);
      mju_cross(tmp2, delta_omega+3*body, tmp1);
      mju_addTo3(local_delta_force, tmp2);
      mju_cross(tmp2, omega+3*body, tmp);
      mju_addTo3(local_delta_force, tmp2);
      mju_cross(tmp, omega+3*body, delta_com);
      mju_cross(tmp2, omega+3*body, tmp);
      mju_addTo3(local_delta_force, tmp2);
      for (int k = 0; k < 3; ++k) {
        local_delta_force[k] = m->body_mass[body]*(delta_acc[3*body+k]+local_delta_force[k]);
      }
      mjtNum delta_inertia_alpha[3], delta_inertia_omega[3];
      body_inertia_mul(m, d, body, delta_alpha+3*body, delta_inertia_alpha);
      body_inertia_mul(m, d, body, delta_omega+3*body, delta_inertia_omega);
      if (rotating) {
        mju_cross(tmp, perturb_axis, alpha+3*body);
        body_inertia_mul(m, d, body, tmp, tmp1);
        mju_sub3(delta_inertia_alpha, delta_inertia_alpha, tmp1);
        mju_cross(tmp, perturb_axis, inertia_alpha+3*body);
        mju_addTo3(delta_inertia_alpha, tmp);
        mju_cross(tmp, perturb_axis, omega+3*body);
        body_inertia_mul(m, d, body, tmp, tmp1);
        mju_sub3(delta_inertia_omega, delta_inertia_omega, tmp1);
        mju_cross(tmp, perturb_axis, inertia_omega+3*body);
        mju_addTo3(delta_inertia_omega, tmp);
      }
      mju_copy(local_delta_moment, delta_inertia_alpha, 3);
      mju_cross(tmp, delta_omega+3*body, inertia_omega+3*body);
      mju_addTo3(local_delta_moment, tmp);
      mju_cross(tmp, omega+3*body, delta_inertia_omega);
      mju_addTo3(local_delta_moment, tmp);
      mju_cross(tmp, delta_com, local_force+3*body);
      mju_addTo3(local_delta_moment, tmp);
      mju_cross(tmp, com+3*body, local_delta_force);
      mju_addTo3(local_delta_moment, tmp);
    }
    if (free_root) {
      for (int row = 0; row < 6; ++row) {
        mjtNum row_axis[3] = {0, 0, 0};
        if (row < 3) {
          row_axis[row] = 1;
        } else {
          mjtNum local_axis[3] = {0, 0, 0};
          local_axis[row-3] = 1;
          mju_mulMatVec3(row_axis, d->xmat+9, local_axis);
        }
        mjtNum delta_axis[3] = {0, 0, 0};
        if (rotating && row >= 3 && joint_body == 1) {
          mju_cross(delta_axis, perturb_axis, row_axis);
        }
        for (int body = 1; body < m->nbody; ++body) {
          mjtNum value[3], delta_value[3];
          if (row < 3) {
            mju_copy(value, force+3*body, 3);
            mju_copy(delta_value, delta_force+3*body, 3);
          } else {
            mjtNum offset[3], delta_offset[3], tmp[3];
            mju_sub3(offset, d->xpos+3*body, d->xpos+3);
            mju_sub3(delta_offset, delta_pos+3*body, delta_pos+3);
            mju_cross(tmp, offset, force+3*body);
            mju_add3(value, moment+3*body, tmp);
            mju_cross(tmp, delta_offset, force+3*body);
            mju_add3(delta_value, delta_moment+3*body, tmp);
            mju_cross(tmp, offset, delta_force+3*body);
            mju_addTo3(delta_value, tmp);
          }
          dbar_qpos[dof] += bar_qfrc_bias[row]*(
              mju_dot3(delta_axis, value) + mju_dot3(row_axis, delta_value));
        }
      }
    }
    for (int row_body = 1; row_body < m->nbody; ++row_body) {
      if (!m->body_jntnum[row_body] || (free_root && row_body == 1)) continue;
      int row_joint = m->body_jntadr[row_body];
      int row_dof = m->jnt_dofadr[row_joint];
      int row_rotating = m->jnt_type[row_joint] == mjJNT_HINGE;
      int ancestor = row_body;
      while (ancestor != joint_body && ancestor > 0) {
        ancestor = m->body_parentid[ancestor];
      }
      mjtNum delta_axis[3] = {0, 0, 0};
      if (rotating && ancestor == joint_body) {
        mju_cross(delta_axis, perturb_axis, axes+3*row_body);
      }
      for (int body = row_body; body < m->nbody; ++body) {
        ancestor = body;
        while (ancestor != row_body && ancestor > 0) {
          ancestor = m->body_parentid[ancestor];
        }
        if (ancestor != row_body) continue;
        mjtNum value[3], delta_value[3];
        if (row_rotating) {
          mjtNum offset[3], delta_offset[3], tmp[3];
          mju_sub3(offset, d->xpos+3*body, d->xpos+3*row_body);
          mju_sub3(delta_offset, delta_pos+3*body, delta_pos+3*row_body);
          mju_cross(tmp, offset, force+3*body);
          mju_add3(value, moment+3*body, tmp);
          mju_cross(tmp, delta_offset, force+3*body);
          mju_add3(delta_value, delta_moment+3*body, tmp);
          mju_cross(tmp, offset, delta_force+3*body);
          mju_addTo3(delta_value, tmp);
        } else {
          mju_copy(value, force+3*body, 3);
          mju_copy(delta_value, delta_force+3*body, 3);
        }
        mjtNum contribution = bar_qfrc_bias[row_dof]*(
            mju_dot3(delta_axis, value) + mju_dot3(axes+3*row_body, delta_value));
        dbar_qpos[dof] += contribution;
      }
    }
  }
  }
  mj_freeStack(d);
  return mjADJ_OK;
}

static int plane_sphere_pos_adjoint(const mjModel* m, const mjData* d,
                                    const mjtNum* y, const mjtNum* lambda, const mjtNum* mu,
                                    const mjtNum* hinv, const mjtNum* qacc_adv,
                                    mjtNum impedance_sign, mjtNum* dbar_qpos) {
  if (!m || !d || (m->nv && (!y || !lambda || !mu || !qacc_adv || !dbar_qpos))) {
    return mjADJ_INVALID_ARGUMENT;
  }
  int dim = d->ncon == 1 ? d->contact[0].dim : 0;
  if (d->ncon != 1 ||
      !((dim == 1 && d->nefc == 1) ||
        (dim >= 3 && m->opt.cone == mjCONE_PYRAMIDAL && d->nefc == 2*(dim-1)) ||
        (dim >= 3 && dim <= 6 && m->opt.cone == mjCONE_ELLIPTIC && d->nefc == dim))) {
    return mjADJ_UNSUPPORTED;
  }
  int cone_contact = m->opt.cone == mjCONE_ELLIPTIC &&
                     d->efc_state[0] == mjCNSTRSTATE_CONE;
  if (cone_contact && !hinv) return mjADJ_INVALID_ARGUMENT;
  const mjContact* con = d->contact;
  int sphere = con->geom1;
  int plane = con->geom2;
  if (m->geom_type[plane] != mjGEOM_PLANE) {
    sphere = con->geom2;
    plane = con->geom1;
  }
  int moving_type = m->geom_type[sphere];
  if ((moving_type != mjGEOM_SPHERE && moving_type != mjGEOM_CAPSULE &&
       moving_type != mjGEOM_BOX && moving_type != mjGEOM_CYLINDER &&
       moving_type != mjGEOM_ELLIPSOID) ||
      m->geom_type[plane] != mjGEOM_PLANE || m->geom_bodyid[plane] != 0) {
    return mjADJ_UNSUPPORTED;
  }
  int sphere_body = m->geom_bodyid[sphere];
  int root = m->body_rootid[sphere_body];
  const mjtNum* com = d->subtree_com+3*root;
  const mjtNum* normal = con->frame;
  // plane-sphere: center; plane-capsule: endpoint; plane-box: corner;
  // plane-cylinder: rim point; plane-ellipsoid: support point (radius 0)
  mjtNum moving_r = (moving_type == mjGEOM_BOX || moving_type == mjGEOM_CYLINDER ||
                     moving_type == mjGEOM_ELLIPSOID) ?
                    0 : m->geom_size[3*sphere];
  mjtNum center[3];
  for (int k = 0; k < 3; ++k) {
    center[k] = con->pos[k]+normal[k]*(.5*con->dist+moving_r);
  }
  // plane-cylinder: the rim witness is the lowest point of the rim circle, which
  // slides around (and along) the rim as the cylinder rotates; vec = rcyl*w/|w|,
  // w = prj*axis - n with the axis flipped so prj = n.axis <= 0
  mjtNum cyl_axis[3] = {0, 0, 0}, cyl_w[3], cyl_wlen = 0, cyl_r = 0, cyl_prj = 0;
  if (moving_type == mjGEOM_CYLINDER) {
    const mjtNum* cmat = d->geom_xmat+9*sphere;
    cyl_axis[0] = cmat[2];
    cyl_axis[1] = cmat[5];
    cyl_axis[2] = cmat[8];
    cyl_prj = mju_dot3(normal, cyl_axis);
    if (cyl_prj > 0) {
      cyl_prj = -cyl_prj;
      mju_scl3(cyl_axis, cyl_axis, -1);
    }
    for (int k = 0; k < 3; ++k) {
      cyl_w[k] = cyl_prj*cyl_axis[k]-normal[k];
    }
    cyl_wlen = mju_norm3(cyl_w);
    if (cyl_wlen < 1e-6) {
      return mjADJ_NON_SMOOTH;
    }
    cyl_r = m->geom_size[3*sphere];
  }
  // plane-ellipsoid: the witness is the ellipsoid support point in direction -n:
  // s = p2 + R*D^2*u/|D*u|, u = R^T*(-n), D = diag(size); smooth for any u != 0
  mjtNum el_u[3], el_sloc[3], el_du[3], el_dun = 0, el_size[3];
  if (moving_type == mjGEOM_ELLIPSOID) {
    const mjtNum* emat = d->geom_xmat+9*sphere;
    mjtNum negn[3];
    mju_scl3(negn, normal, -1);
    mju_mulMatTVec3(el_u, emat, negn);
    mjtNum dun2 = 0;
    for (int k = 0; k < 3; ++k) {
      el_size[k] = m->geom_size[3*sphere+k];
      el_du[k] = el_size[k]*el_u[k];
      dun2 += el_du[k]*el_du[k];
    }
    el_dun = mju_sqrt(dun2);
    if (el_dun < 1e-9) {
      return mjADJ_NON_SMOOTH;
    }
    for (int k = 0; k < 3; ++k) {
      el_sloc[k] = el_size[k]*el_du[k]/el_dun;
    }
    mjtNum sw[3];
    mju_mulMatVec3(sw, emat, el_sloc);
    mjtNum sp[3];
    mju_add3(sp, sw, d->geom_xpos+3*sphere);
    if (mju_dist3(sp, center) > 1e-9*(1+mju_norm3(center))) {
      return mjADJ_NON_SMOOTH;
    }
  }
  mjtNum cone_H[36];
  mjtNum cone_q[6];
  if (cone_contact) {
    int hessian_status = plane_sphere_cone_hessian(m, d, cone_H);
    if (hessian_status != mjADJ_OK) return hessian_status;
    for (int row = 0; row < d->nefc; ++row) {
      cone_q[row] = mju_dot(cone_H+d->nefc*row, hinv, d->nefc);
    }
  }
  for (int joint = 0; joint < m->njnt; ++joint) {
    int joint_body = m->jnt_bodyid[joint];
    int free_joint = m->jnt_type[joint] == mjJNT_FREE;
    int channels = free_joint ? 6 : 1;
    for (int channel = 0; channel < channels; ++channel) {
      int dof = m->jnt_dofadr[joint]+channel;
      int rotating = m->jnt_type[joint] == mjJNT_HINGE ||
                     (free_joint && channel >= 3);
      mjtNum axis[3] = {0, 0, 0};
      if (free_joint && channel < 3) {
        axis[channel] = 1;
      } else if (free_joint) {
        mju_copy(axis, d->cdof+6*dof, 3);
      } else {
        mju_copy(axis, d->xaxis+3*joint, 3);
      }
      const mjtNum* anchor = d->xanchor+3*joint;
      mjtNum delta_point[3] = {0, 0, 0};
      mjtNum delta_com[3] = {0, 0, 0};
      int ancestor = sphere_body;
      while (ancestor != joint_body && ancestor > 0) {
        ancestor = m->body_parentid[ancestor];
      }
      int moves_contact = ancestor == joint_body;
      if (moves_contact) {
        if (rotating) {
          mjtNum offset[3];
          mju_sub3(offset, center, anchor);
          mju_cross(delta_point, axis, offset);
          if (moving_type == mjGEOM_CYLINDER) {
            // non-material rim witness: add dvec - axis x vec
            mjtNum da[3], axn[3];
            mju_cross(da, axis, cyl_axis);
            mju_cross(axn, axis, normal);
            mjtNum dprj = mju_dot3(normal, da);
            for (int k = 0; k < 3; ++k) {
              mjtNum term = (dprj*cyl_axis[k]+axn[k])/cyl_wlen+
                  cyl_w[k]*cyl_prj*dprj/(cyl_wlen*cyl_wlen*cyl_wlen);
              delta_point[k] += cyl_r*term;
            }
          }
          if (moving_type == mjGEOM_ELLIPSOID) {
            // support point is not material: add R * d(s_local)
            // du = R^T (axis x n); ds_loc = (D^2 du)/|Du| - s_loc (Du . D du)/|Du|^2
            mjtNum axn[3], du[3], ddu[3];
            mju_cross(axn, axis, normal);
            const mjtNum* emat = d->geom_xmat+9*sphere;
            mju_mulMatTVec3(du, emat, axn);
            mjtNum ddun = 0;
            for (int k = 0; k < 3; ++k) {
              ddu[k] = el_size[k]*du[k];
              ddun += el_du[k]*ddu[k];
            }
            mjtNum dsloc[3];
            for (int k = 0; k < 3; ++k) {
              dsloc[k] = el_size[k]*ddu[k]/el_dun-
                  el_sloc[k]*ddun/(el_dun*el_dun);
            }
            mjtNum dscorr[3];
            mju_mulMatVec3(dscorr, emat, dsloc);
            mju_addTo3(delta_point, dscorr);
          }
        } else {
          mju_copy(delta_point, axis, 3);
        }
      }
      for (int body = 1; body < m->nbody; ++body) {
        if (m->body_rootid[body] != root) continue;
        ancestor = body;
        while (ancestor != joint_body && ancestor > 0) {
          ancestor = m->body_parentid[ancestor];
        }
        if (ancestor != joint_body) continue;
        mjtNum delta_ipos[3];
        if (rotating) {
          mjtNum offset[3];
          mju_sub3(offset, d->xipos+3*body, anchor);
          mju_cross(delta_ipos, axis, offset);
        } else {
          mju_copy(delta_ipos, axis, 3);
        }
        mju_addToScl3(delta_com, delta_ipos,
                      m->body_mass[body]/m->body_subtreemass[root]);
      }
      mjtNum delta_dist = mju_dot3(normal, delta_point);
      mjtNum delta_contact[3];
      mju_copy(delta_contact, delta_point, 3);
      mju_addToScl3(delta_contact, normal, -.5*delta_dist);
      mjtNum delta_lever[3];
      mju_sub3(delta_lever, delta_contact, delta_com);
      for (int row = 0; row < m->nv; ++row) {
        int row_body = m->dof_bodyid[row];
        int row_joint = m->dof_jntid[row];
        mjtJoint row_type = (mjtJoint)m->jnt_type[row_joint];
        int row_rotating = row_type == mjJNT_HINGE ||
                           (row_type == mjJNT_FREE && row >= 3);
        ancestor = sphere_body;
        while (ancestor != row_body && ancestor > 0) {
          ancestor = m->body_parentid[ancestor];
        }
        if (ancestor != row_body) continue;
        const mjtNum* cdof = d->cdof+6*row;
        mjtNum delta_axis[3] = {0, 0, 0};
        mjtNum delta_lin[3] = {0, 0, 0};
        ancestor = row_body;
        while (ancestor != joint_body && ancestor > 0) {
          ancestor = m->body_parentid[ancestor];
        }
        if (ancestor == joint_body && rotating && !(row_type == mjJNT_FREE && row < 3)) {
          mjtNum row_axis[3];
          if (row_rotating) {
            mju_copy(row_axis, cdof, 3);
          } else {
            mju_copy(row_axis, d->xaxis+3*row_joint, 3);
          }
          mju_cross(delta_axis, axis, row_axis);
        }
        if (row_rotating) {
          mjtNum row_lever[3], delta_anchor[3] = {0, 0, 0};
          mju_sub3(row_lever, com, d->xanchor+3*row_joint);
          if (ancestor == joint_body) {
            if (rotating) {
              mjtNum offset[3];
              mju_sub3(offset, d->xanchor+3*row_joint, anchor);
              mju_cross(delta_anchor, axis, offset);
            } else {
              mju_copy(delta_anchor, axis, 3);
            }
          }
          mjtNum delta_row_lever[3], tmp[3];
          mju_sub3(delta_row_lever, delta_com, delta_anchor);
          mju_cross(delta_lin, delta_axis, row_lever);
          mju_cross(tmp, cdof, delta_row_lever);
          mju_addTo3(delta_lin, tmp);
        } else if (row_type != mjJNT_FREE) {
          mju_copy(delta_lin, delta_axis, 3);
        }
        mjtNum lever[3], tmp[3], cross_delta[3], delta_jac_vec[3], jac_lin[3];
        mju_sub3(lever, con->pos, com);
        mju_cross(tmp, cdof, delta_lever);
        mju_cross(cross_delta, delta_axis, lever);
        mju_add3(delta_jac_vec, delta_lin, tmp);
        mju_addTo3(delta_jac_vec, cross_delta);
        mju_cross(jac_lin, cdof, lever);
        mju_addTo3(jac_lin, cdof+3);
        mjtNum delta_tangent[3] = {0, 0, 0};
        if (moving_type == mjGEOM_CAPSULE && moves_contact && rotating) {
          const mjtNum* mat = d->geom_xmat+9*sphere;
          mjtNum capsule_axis[3] = {mat[2], mat[5], mat[8]}, raw_delta[3];
          mju_cross(raw_delta, axis, capsule_axis);
          mjtNum normal_delta = mju_dot3(normal, raw_delta);
          mjtNum tangent_delta = mju_dot3(con->frame+3, raw_delta);
          mjtNum projection = mju_dot3(normal, capsule_axis);
          mjtNum length = mju_sqrt(mju_max(mjMINVAL, 1-projection*projection));
          for (int k = 0; k < 3; ++k) {
            delta_tangent[k] = (raw_delta[k]-normal[k]*normal_delta-
                                con->frame[3+k]*tangent_delta)/length;
          }
        }
        for (int constraint = 0; constraint < d->nefc; ++constraint) {
          mjtNum delta_jac = mju_dot3(normal, delta_jac_vec);
          mjtNum delta_dir[3] = {0, 0, 0};
          if (m->opt.cone == mjCONE_ELLIPTIC) {
            if (constraint < 3) {
              delta_jac = mju_dot3(con->frame+3*constraint, delta_jac_vec);
              if (constraint == 1) {
                mju_copy(delta_dir, delta_tangent, 3);
              } else if (constraint == 2) {
                mju_cross(delta_dir, normal, delta_tangent);
              }
              delta_jac += mju_dot3(delta_dir, jac_lin);
            } else {
              int direction = constraint == 3 ? 0 : constraint-3;
              delta_jac = mju_dot3(con->frame+3*direction, delta_axis);
              if (constraint == 4) {
                mju_copy(delta_dir, delta_tangent, 3);
              } else if (constraint == 5) {
                mju_cross(delta_dir, normal, delta_tangent);
              }
              delta_jac += mju_dot3(delta_dir, cdof);
            }
          } else if (dim > 1) {
            int tangent = constraint/2+1;
            mjtNum sign = constraint%2 ? -1 : 1;
            if (tangent < 3) {
              delta_jac += sign*con->friction[tangent-1]*
                           mju_dot3(con->frame+3*tangent, delta_jac_vec);
              if (tangent == 1) {
                mju_copy(delta_dir, delta_tangent, 3);
              } else {
                mju_cross(delta_dir, normal, delta_tangent);
              }
              delta_jac += sign*con->friction[tangent-1]*mju_dot3(delta_dir, jac_lin);
            } else {
              int direction = tangent-3;
              delta_jac += sign*con->friction[tangent-1]*
                           mju_dot3(con->frame+3*direction, delta_axis);
              if (direction == 1) {
                mju_copy(delta_dir, delta_tangent, 3);
              } else if (direction == 2) {
                mju_cross(delta_dir, normal, delta_tangent);
              }
              delta_jac += sign*con->friction[tangent-1]*mju_dot3(delta_dir, cdof);
            }
          }
          mjtNum acceleration_bar = cone_contact ? -cone_q[constraint] : mu[constraint];
          dbar_qpos[dof] += delta_jac*(y[row]*lambda[constraint] +
              acceleration_bar*(qacc_adv[row]+d->efc_KBIP[4*constraint+1]*d->qvel[row]));
        }
      }
      for (int constraint = 0; constraint < d->nefc; ++constraint) {
        mjtNum K = d->efc_KBIP[4*constraint];
        mjtNum imp = d->efc_KBIP[4*constraint+2];
        mjtNum impP = d->efc_KBIP[4*constraint+3];
        mjtNum dR_ddist = 0;
        if (imp > 0 && imp < 1 && d->efc_R[constraint] > mjMINVAL) {
          dR_ddist = -d->efc_R[constraint]*impP/(imp*(1-imp));
        }
        mjtNum position_gain = K*(imp+impP*(d->efc_pos[constraint]-d->efc_margin[constraint]));
        dbar_qpos[dof] += (cone_contact ? -cone_q[constraint] : mu[constraint])*
                          position_gain*delta_dist;
        if (cone_contact) {
          mjtNum scale = -lambda[constraint]*impP/(imp*(1-imp))*delta_dist;
          dbar_qpos[dof] += impedance_sign*hinv[constraint]*scale;
        } else {
          dbar_qpos[dof] += mu[constraint]*lambda[constraint]*dR_ddist*delta_dist;
        }
      }
    }
  }
  return mjADJ_OK;
}

int mjd_planeSpherePosAdjoint(const mjModel* m, const mjData* d,
                              const mjtNum* y, const mjtNum* lambda, const mjtNum* mu,
                              const mjtNum* hinv, const mjtNum* qacc_adv,
                              mjtNum* dbar_qpos) {
  return plane_sphere_pos_adjoint(m, d, y, lambda, mu, hinv, qacc_adv, 1, dbar_qpos);
}

int mjd_planeSphereStepPosAdjoint(const mjModel* m, const mjData* d,
                                  const mjtNum* y, const mjtNum* lambda, const mjtNum* mu,
                                  const mjtNum* hinv, const mjtNum* qacc_adv,
                                  mjtNum* dbar_qpos) {
  return plane_sphere_pos_adjoint(m, d, y, lambda, mu, hinv, qacc_adv, -1, dbar_qpos);
}

// VJP of one capsule-capsule / sphere-capsule / sphere-sphere contact (non-parallel
// capsule axes, interior closest points). Spheres are degenerate segments (h=0).
// Contact geometry mirrors mjraw_CapsuleCapsule / mjraw_SphereCapsule /
// mjraw_SphereSphere: witness points w1/w2 on the two scaled axes (sphere: center),
// normal from w1 to w2, pos = midpoint of the two surface points.
static int capsule_capsule_pos_adjoint(const mjModel* m, const mjData* d,
                                       const mjtNum* y, const mjtNum* lambda, const mjtNum* mu,
                                       const mjtNum* hinv, const mjtNum* qacc_adv,
                                       mjtNum impedance_sign, mjtNum* dbar_qpos) {
  if (!m || !d || (m->nv && (!y || !lambda || !mu || !qacc_adv || !dbar_qpos))) {
    return mjADJ_INVALID_ARGUMENT;
  }
  int dim = d->ncon == 1 ? d->contact[0].dim : 0;
  if (d->ncon != 1 || d->ne || d->nf ||
      !((dim == 1 && d->nefc == 1) ||
        (dim >= 3 && m->opt.cone == mjCONE_PYRAMIDAL && d->nefc == 2*(dim-1)) ||
        (dim >= 3 && dim <= 6 && m->opt.cone == mjCONE_ELLIPTIC && d->nefc == dim))) {
    return mjADJ_UNSUPPORTED;
  }
  const mjContact* con = d->contact;
  int cap1 = con->geom1;
  int cap2 = con->geom2;
  int type1 = m->geom_type[cap1], type2 = m->geom_type[cap2];
  if (!((type1 == mjGEOM_CAPSULE || type1 == mjGEOM_SPHERE) &&
        (type2 == mjGEOM_CAPSULE || type2 == mjGEOM_SPHERE))) {
    return mjADJ_UNSUPPORTED;
  }
  int body1 = m->geom_bodyid[cap1];
  int body2 = m->geom_bodyid[cap2];
  mjtNum r1 = m->geom_size[3*cap1];
  mjtNum r2 = m->geom_size[3*cap2];
  mjtNum h1 = type1 == mjGEOM_CAPSULE ? m->geom_size[3*cap1+1] : 0;
  mjtNum h2 = type2 == mjGEOM_CAPSULE ? m->geom_size[3*cap2+1] : 0;
  const mjtNum* p1 = d->geom_xpos+3*cap1;
  const mjtNum* p2 = d->geom_xpos+3*cap2;
  const mjtNum* mat1 = d->geom_xmat+9*cap1;
  const mjtNum* mat2 = d->geom_xmat+9*cap2;
  mjtNum a1[3] = {mat1[2], mat1[5], mat1[8]};
  mjtNum a2[3] = {mat2[2], mat2[5], mat2[8]};
  mjtNum s1[3], s2[3];
  mju_scl3(s1, a1, h1);
  mju_scl3(s2, a2, h2);
  const mjtNum* nrm = con->frame;
  mjtNum L = con->dist + r1 + r2;
  if (L < mjMINVAL)     return mjADJ_NON_SMOOTH;
  // reconstruct witness points from the contact record
  mjtNum w1[3], w2[3], g12[3];
  for (int k = 0; k < 3; ++k) {
    w1[k] = con->pos[k]-nrm[k]*(r1+.5*con->dist);
    w2[k] = con->pos[k]+nrm[k]*(r2+.5*con->dist);
    g12[k] = w2[k]-w1[k];
  }
  // closest-point system; spheres are degenerate segments with no unknown
  mjtNum ma = mju_dot3(s1, s1), pp = mju_dot3(s1, s2), mc = mju_dot3(s2, s2);
  mjtNum dif[3];
  mju_sub3(dif, p1, p2);
  mjtNum u = -mju_dot3(s1, dif), v = mju_dot3(s2, dif);
  mjtNum x1 = 0, x2 = 0;
  if (ma > mjMINVAL && mc > mjMINVAL) {
    mjtNum scale = mju_max(ma, mc);
    mjtNum det = ma*mc-pp*pp;
    if (det < 1e-12*scale*scale)     return mjADJ_NON_SMOOTH;
    x1 = (mc*u+pp*v)/det;
    x2 = (ma*v+pp*u)/det;
  } else if (mc > mjMINVAL) {
    // sphere-capsule: only the capsule closest point is unknown
    x2 = v/mc;
    if (mju_abs(x2) > 1-1e-6)     return mjADJ_NON_SMOOTH;
  } else if (ma > mjMINVAL) {
    x1 = u/ma;
    if (mju_abs(x1) > 1-1e-6)     return mjADJ_NON_SMOOTH;
  }
  mjtNum chk1[3], chk2[3];
  mju_scl3(chk1, s1, x1);
  mju_addTo3(chk1, p1);
  mju_scl3(chk2, s2, x2);
  mju_addTo3(chk2, p2);
  if (mju_dist3(chk1, w1) > 1e-9*(1+mju_norm3(w1)) ||
      mju_dist3(chk2, w2) > 1e-9*(1+mju_norm3(w2))) {
        return mjADJ_NON_SMOOTH;
  }
  if (ma > mjMINVAL && mju_abs(x1) > 1-1e-6)     return mjADJ_NON_SMOOTH;
  // tangent frame comes from mju_makeFrame with a zero preferred tangent; keep the
  // branch (|n_y| vs 0.5) away from the decision boundary
  if (mju_abs(mju_abs(nrm[1])-.5) < 1e-3)     return mjADJ_NON_SMOOTH;
  int cone_contact = m->opt.cone == mjCONE_ELLIPTIC &&
                     d->efc_state[0] == mjCNSTRSTATE_CONE;
  if (cone_contact && !hinv) return mjADJ_INVALID_ARGUMENT;
  mjtNum cone_H[36];
  mjtNum cone_q[6];
  if (cone_contact) {
    int hessian_status = plane_sphere_cone_hessian(m, d, cone_H);
    if (hessian_status != mjADJ_OK) return hessian_status;
    for (int row = 0; row < d->nefc; ++row) {
      cone_q[row] = mju_dot(cone_H+d->nefc*row, hinv, d->nefc);
    }
  }
  for (int joint = 0; joint < m->njnt; ++joint) {
    int joint_body = m->jnt_bodyid[joint];
    int free_joint = m->jnt_type[joint] == mjJNT_FREE;
    int channels = free_joint ? 6 : 1;
    for (int channel = 0; channel < channels; ++channel) {
      int dof = m->jnt_dofadr[joint]+channel;
      int rotating = m->jnt_type[joint] == mjJNT_HINGE ||
                     (free_joint && channel >= 3);
      mjtNum axis[3] = {0, 0, 0};
      if (free_joint && channel < 3) {
        axis[channel] = 1;
      } else if (free_joint) {
        mju_copy(axis, d->cdof+6*dof, 3);
      } else {
        mju_copy(axis, d->xaxis+3*joint, 3);
      }
      const mjtNum* anchor = d->xanchor+3*joint;
      // per-side pose derivatives
      mjtNum dp1[3] = {0, 0, 0}, dp2[3] = {0, 0, 0};
      mjtNum da1[3] = {0, 0, 0}, da2[3] = {0, 0, 0};
      int moves1 = 0, moves2 = 0;
      int ancestor = body1;
      while (ancestor != joint_body && ancestor > 0) {
        ancestor = m->body_parentid[ancestor];
      }
      moves1 = ancestor == joint_body;
      ancestor = body2;
      while (ancestor != joint_body && ancestor > 0) {
        ancestor = m->body_parentid[ancestor];
      }
      moves2 = ancestor == joint_body;
      if (moves1) {
        if (rotating) {
          mjtNum offset[3];
          mju_sub3(offset, p1, anchor);
          mju_cross(dp1, axis, offset);
          mju_cross(da1, axis, a1);
        } else {
          mju_copy(dp1, axis, 3);
        }
      }
      if (moves2) {
        if (rotating) {
          mjtNum offset[3];
          mju_sub3(offset, p2, anchor);
          mju_cross(dp2, axis, offset);
          mju_cross(da2, axis, a2);
        } else {
          mju_copy(dp2, axis, 3);
        }
      }
      // derivative of the closest-point parameters
      mjtNum ds1[3], ds2[3], dg0[3];
      mju_scl3(ds1, da1, h1);
      mju_scl3(ds2, da2, h2);
      for (int k = 0; k < 3; ++k) {
        dg0[k] = dp2[k]+ds2[k]*x2-dp1[k]-ds1[k]*x1;
      }
      mjtNum rhs1 = mju_dot3(s1, dg0)+mju_dot3(ds1, g12);
      mjtNum rhs2 = -mju_dot3(s2, dg0)-mju_dot3(ds2, g12);
      mjtNum dx1 = 0, dx2 = 0;
      if (ma > mjMINVAL && mc > mjMINVAL) {
        mjtNum detp = ma*mc-pp*pp;
        dx1 = (mc*rhs1+pp*rhs2)/detp;
        dx2 = (pp*rhs1+ma*rhs2)/detp;
      } else if (mc > mjMINVAL) {
        dx2 = rhs2/mc;
      } else if (ma > mjMINVAL) {
        dx1 = rhs1/ma;
      }
      mjtNum dg[3], dn[3];
      for (int k = 0; k < 3; ++k) {
        dg[k] = dg0[k]+s2[k]*dx2-s1[k]*dx1;
      }
      mjtNum dL = mju_dot3(nrm, dg);
      for (int k = 0; k < 3; ++k) {
        dn[k] = (dg[k]-nrm[k]*dL)/L;
      }
      // makeFrame default tangent: t0 = ey if |n_y|<0.5 else ez; t = P t0/|P t0|
      mjtNum t0[3] = {0, 1, 0};
      if (mju_abs(nrm[1]) >= .5) {
        t0[0] = 0;
        t0[1] = 0;
        t0[2] = 1;
      }
      mjtNum c = mju_dot3(nrm, t0);
      mjtNum uvec[3], dtan[3];
      for (int k = 0; k < 3; ++k) {
        uvec[k] = t0[k]-nrm[k]*c;
      }
      mjtNum us = mju_norm3(uvec);
      if (us < mjMINVAL)     return mjADJ_NON_SMOOTH;
      mjtNum dc = mju_dot3(dn, t0);
      mjtNum du[3];
      for (int k = 0; k < 3; ++k) {
        du[k] = -dn[k]*c-nrm[k]*dc;
      }
      mjtNum dus = mju_dot3(uvec, du)/us;
      for (int k = 0; k < 3; ++k) {
        dtan[k] = du[k]/us-uvec[k]*dus/(us*us);
      }
      const mjtNum* tan = con->frame+3;
      mjtNum dbit[3], tmp_bit[3];
      mju_cross(tmp_bit, dn, tan);
      mju_cross(dbit, nrm, dtan);
      mju_addTo3(dbit, tmp_bit);
      mjtNum dw1[3], dpos[3];
      for (int k = 0; k < 3; ++k) {
        dw1[k] = dp1[k]+ds1[k]*x1+s1[k]*dx1;
        dpos[k] = dw1[k]+dn[k]*(r1+.5*con->dist)+nrm[k]*.5*dL;
      }
      mjtNum ddist = dL;
      for (int row = 0; row < m->nv; ++row) {
        int row_body = m->dof_bodyid[row];
        int row_joint = m->dof_jntid[row];
        mjtJoint row_type = (mjtJoint)m->jnt_type[row_joint];
        int row_rotating = row_type == mjJNT_HINGE ||
                           (row_type == mjJNT_FREE && row >= 3);
        int in1 = 0, in2 = 0;
        ancestor = body1;
        while (ancestor != row_body && ancestor > 0) {
          ancestor = m->body_parentid[ancestor];
        }
        in1 = ancestor == row_body;
        ancestor = body2;
        while (ancestor != row_body && ancestor > 0) {
          ancestor = m->body_parentid[ancestor];
        }
        in2 = ancestor == row_body;
        if (!in1 && !in2) continue;
        int moved = 0;
        ancestor = row_body;
        while (ancestor != joint_body && ancestor > 0) {
          ancestor = m->body_parentid[ancestor];
        }
        moved = ancestor == joint_body;
        const mjtNum* cdof = d->cdof+6*row;
        mjtNum delta_axis[3] = {0, 0, 0};
        mjtNum delta_lin[3] = {0, 0, 0};
        if (moved && rotating && !(row_type == mjJNT_FREE && row < 3)) {
          mjtNum row_axis[3];
          if (row_rotating) {
            mju_copy(row_axis, cdof, 3);
          } else {
            mju_copy(row_axis, d->xaxis+3*row_joint, 3);
          }
          mju_cross(delta_axis, axis, row_axis);
        }
        if (row_rotating) {
          mjtNum dcom[3] = {0, 0, 0};
          int row_root = m->body_rootid[row_body];
          for (int body = 1; body < m->nbody; ++body) {
            if (m->body_rootid[body] != row_root) continue;
            ancestor = body;
            while (ancestor != joint_body && ancestor > 0) {
              ancestor = m->body_parentid[ancestor];
            }
            if (ancestor != joint_body) continue;
            mjtNum dip[3];
            if (rotating) {
              mjtNum offset[3];
              mju_sub3(offset, d->xipos+3*body, anchor);
              mju_cross(dip, axis, offset);
            } else {
              mju_copy(dip, axis, 3);
            }
            mju_addToScl3(dcom, dip,
                          m->body_mass[body]/m->body_subtreemass[row_root]);
          }
          mjtNum danchor[3] = {0, 0, 0};
          if (moved) {
            if (rotating) {
              mjtNum offset[3];
              mju_sub3(offset, d->xanchor+3*row_joint, anchor);
              mju_cross(danchor, axis, offset);
            } else {
              mju_copy(danchor, axis, 3);
            }
          }
          const mjtNum* com = d->subtree_com+3*row_root;
          mjtNum row_lever[3], dlever[3], tmp[3];
          mju_sub3(row_lever, com, d->xanchor+3*row_joint);
          mju_sub3(dlever, dcom, danchor);
          mju_cross(delta_lin, delta_axis, row_lever);
          mju_cross(tmp, cdof, dlever);
          mju_addTo3(delta_lin, tmp);
        } else if (row_type != mjJNT_FREE) {
          mju_copy(delta_lin, delta_axis, 3);
        }
        mjtNum jac1[3] = {0, 0, 0}, jac2[3] = {0, 0, 0};
        mjtNum djac1[3] = {0, 0, 0}, djac2[3] = {0, 0, 0};
        if (in1) {
          const mjtNum* com = d->subtree_com+3*m->body_rootid[body1];
          mjtNum dcom1[3] = {0, 0, 0};
          int root1 = m->body_rootid[body1];
          for (int body = 1; body < m->nbody; ++body) {
            if (m->body_rootid[body] != root1) continue;
            ancestor = body;
            while (ancestor != joint_body && ancestor > 0) {
              ancestor = m->body_parentid[ancestor];
            }
            if (ancestor != joint_body) continue;
            mjtNum dip[3];
            if (rotating) {
              mjtNum offset[3];
              mju_sub3(offset, d->xipos+3*body, anchor);
              mju_cross(dip, axis, offset);
            } else {
              mju_copy(dip, axis, 3);
            }
            mju_addToScl3(dcom1, dip,
                          m->body_mass[body]/m->body_subtreemass[root1]);
          }
          mjtNum lever[3], dlev[3], tmp[3];
          mju_sub3(lever, con->pos, com);
          mju_sub3(dlev, dpos, dcom1);
          mju_cross(jac1, cdof, lever);
          mju_addTo3(jac1, cdof+3);
          mju_cross(djac1, delta_axis, lever);
          mju_cross(tmp, cdof, dlev);
          mju_addTo3(djac1, tmp);
          mju_addTo3(djac1, delta_lin);
        }
        if (in2) {
          const mjtNum* com = d->subtree_com+3*m->body_rootid[body2];
          mjtNum dcom2[3] = {0, 0, 0};
          int root2 = m->body_rootid[body2];
          for (int body = 1; body < m->nbody; ++body) {
            if (m->body_rootid[body] != root2) continue;
            ancestor = body;
            while (ancestor != joint_body && ancestor > 0) {
              ancestor = m->body_parentid[ancestor];
            }
            if (ancestor != joint_body) continue;
            mjtNum dip[3];
            if (rotating) {
              mjtNum offset[3];
              mju_sub3(offset, d->xipos+3*body, anchor);
              mju_cross(dip, axis, offset);
            } else {
              mju_copy(dip, axis, 3);
            }
            mju_addToScl3(dcom2, dip,
                          m->body_mass[body]/m->body_subtreemass[root2]);
          }
          mjtNum lever[3], dlev[3], tmp[3];
          mju_sub3(lever, con->pos, com);
          mju_sub3(dlev, dpos, dcom2);
          mju_cross(jac2, cdof, lever);
          mju_addTo3(jac2, cdof+3);
          mju_cross(djac2, delta_axis, lever);
          mju_cross(tmp, cdof, dlev);
          mju_addTo3(djac2, tmp);
          mju_addTo3(djac2, delta_lin);
        }
        mjtNum jacD[3], djacD[3], angD[3], dangD[3];
        for (int k = 0; k < 3; ++k) {
          // stock contact Jacobian is pos2 - pos1 (mj_jacDifPair)
          jacD[k] = jac2[k]-jac1[k];
          djacD[k] = djac2[k]-djac1[k];
          angD[k] = (in2 ? cdof[k] : 0)-(in1 ? cdof[k] : 0);
          dangD[k] = (in2 ? delta_axis[k] : 0)-(in1 ? delta_axis[k] : 0);
        }
        for (int constraint = 0; constraint < d->nefc; ++constraint) {
          // pyramidal contact rows are n + sign*mu*tangent combinations, so every
          // pyramidal pair rows are n + sign*fr*row_k combinations, so every row
          // carries the (varying) normal direction; elliptic torsional rows are
          // purely rotational
          mjtNum delta_jac = mju_dot3(dn, jacD)+mju_dot3(nrm, djacD);
          if (m->opt.cone == mjCONE_ELLIPTIC) {
            if (constraint > 0) {
              int direction = constraint < 3 ? constraint : constraint-3;
              const mjtNum* dir = con->frame+3*direction;
              mjtNum ddir[3];
              if (direction == 0) {
                mju_copy(ddir, dn, 3);
              } else if (direction == 1) {
                mju_copy(ddir, dtan, 3);
              } else {
                mju_copy(ddir, dbit, 3);
              }
              if (constraint < 3) {
                delta_jac = mju_dot3(ddir, jacD)+mju_dot3(dir, djacD);
              } else {
                delta_jac = mju_dot3(ddir, angD)+mju_dot3(dir, dangD);
              }
            }
          } else if (dim > 1) {
            int tangent = constraint/2+1;
            mjtNum sign = constraint%2 ? -1 : 1;
            mjtNum friction = con->friction[tangent-1];
            if (tangent < 3) {
              const mjtNum* td = con->frame+3*tangent;
              mjtNum dtd[3];
              if (tangent == 1) {
                mju_copy(dtd, dtan, 3);
              } else {
                mju_copy(dtd, dbit, 3);
              }
              delta_jac += sign*friction*(mju_dot3(dtd, jacD)+mju_dot3(td, djacD));
            } else {
              int direction = tangent-3;
              const mjtNum* dir = con->frame+3*direction;
              mjtNum ddir[3];
              if (direction == 0) {
                mju_copy(ddir, dn, 3);
              } else if (direction == 1) {
                mju_copy(ddir, dtan, 3);
              } else {
                mju_copy(ddir, dbit, 3);
              }
              delta_jac += sign*friction*(mju_dot3(ddir, angD)+mju_dot3(dir, dangD));
            }
          }
          mjtNum acceleration_bar = cone_contact ? -cone_q[constraint] : mu[constraint];
          dbar_qpos[dof] += delta_jac*(y[row]*lambda[constraint]+
              acceleration_bar*(qacc_adv[row]+d->efc_KBIP[4*constraint+1]*d->qvel[row]));
        }
      }
      for (int constraint = 0; constraint < d->nefc; ++constraint) {
        mjtNum K = d->efc_KBIP[4*constraint];
        mjtNum imp = d->efc_KBIP[4*constraint+2];
        mjtNum impP = d->efc_KBIP[4*constraint+3];
        mjtNum dR_ddist = 0;
        if (imp > 0 && imp < 1 && d->efc_R[constraint] > mjMINVAL) {
          dR_ddist = -d->efc_R[constraint]*impP/(imp*(1-imp));
        }
        mjtNum position_gain = K*(imp+impP*(d->efc_pos[constraint]-d->efc_margin[constraint]));
        dbar_qpos[dof] += (cone_contact ? -cone_q[constraint] : mu[constraint])*
                          position_gain*ddist;
        if (cone_contact) {
          mjtNum scale = -lambda[constraint]*impP/(imp*(1-imp))*ddist;
          dbar_qpos[dof] += impedance_sign*hinv[constraint]*scale;
        } else {
          dbar_qpos[dof] += mu[constraint]*lambda[constraint]*dR_ddist*ddist;
        }
      }
    }
  }
  return mjADJ_OK;
}

int mjd_capsuleCapsulePosAdjoint(const mjModel* m, const mjData* d,
                                 const mjtNum* y, const mjtNum* lambda, const mjtNum* mu,
                                 const mjtNum* hinv, const mjtNum* qacc_adv,
                                 mjtNum* dbar_qpos) {
  return capsule_capsule_pos_adjoint(m, d, y, lambda, mu, hinv, qacc_adv, 1, dbar_qpos);
}

int mjd_capsuleCapsuleStepPosAdjoint(const mjModel* m, const mjData* d,
                                     const mjtNum* y, const mjtNum* lambda,
                                     const mjtNum* mu, const mjtNum* hinv,
                                     const mjtNum* qacc_adv, mjtNum* dbar_qpos) {
  return capsule_capsule_pos_adjoint(m, d, y, lambda, mu, hinv, qacc_adv, -1, dbar_qpos);
}

// VJP of one sphere-box / capsule-box contact whose box witness is a face interior
// point (exactly one clamped box-local coordinate). Mirrors mjraw_SphereBox as called
// by mjraw_CapsuleBox (single contact, no second capsule point). The capsule axis
// point q = p1 + s1*x1 satisfies s1 . (q - w2) = 0; the box witness is
// w2 = R*P*c_loc + size_j*n + p2 with P = I - e_j e_j^T (clamped coord j held).
static int sphere_box_face_pos_adjoint(const mjModel* m, const mjData* d,
                                       const mjtNum* y, const mjtNum* lambda,
                                       const mjtNum* mu, const mjtNum* hinv,
                                       const mjtNum* qacc_adv,
                                       mjtNum impedance_sign, mjtNum* dbar_qpos) {
  if (!m || !d || (m->nv && (!y || !lambda || !mu || !qacc_adv || !dbar_qpos))) {
    return mjADJ_INVALID_ARGUMENT;
  }
  int dim = d->ncon == 1 ? d->contact[0].dim : 0;
  if (d->ncon != 1 || d->ne || d->nf ||
      !((dim == 1 && d->nefc == 1) ||
        (dim >= 3 && m->opt.cone == mjCONE_PYRAMIDAL && d->nefc == 2*(dim-1)) ||
        (dim >= 3 && dim <= 6 && m->opt.cone == mjCONE_ELLIPTIC && d->nefc == dim))) {
    return mjADJ_UNSUPPORTED;
  }
  const mjContact* con = d->contact;
  int cap1 = con->geom1;
  int box = con->geom2;
  int type1 = m->geom_type[cap1];
  if (!((type1 == mjGEOM_SPHERE || type1 == mjGEOM_CAPSULE) &&
        m->geom_type[box] == mjGEOM_BOX)) {
    return mjADJ_UNSUPPORTED;
  }
  int body1 = m->geom_bodyid[cap1];
  int body2 = m->geom_bodyid[box];
  mjtNum r1 = m->geom_size[3*cap1];
  mjtNum h1 = type1 == mjGEOM_CAPSULE ? m->geom_size[3*cap1+1] : 0;
  const mjtNum* p1 = d->geom_xpos+3*cap1;
  const mjtNum* p2 = d->geom_xpos+3*box;
  const mjtNum* mat1 = d->geom_xmat+9*cap1;
  const mjtNum* mat2 = d->geom_xmat+9*box;
  const mjtNum* size2 = m->geom_size+3*box;
  mjtNum a1[3] = {mat1[2], mat1[5], mat1[8]};
  mjtNum s1[3];
  mju_scl3(s1, a1, h1);
  const mjtNum* nrm = con->frame;
  mjtNum L = con->dist + r1;
  if (L < mjMINVAL)     return mjADJ_NON_SMOOTH;
  // reconstruct capsule-axis point and box witness from the contact record
  // (pos is the midpoint of w2 and the sphere surface point q + n*r1)
  mjtNum q[3], w2[3], g12[3];
  for (int k = 0; k < 3; ++k) {
    q[k] = con->pos[k]-nrm[k]*(r1+.5*con->dist);
    w2[k] = con->pos[k]+nrm[k]*.5*con->dist;
    g12[k] = w2[k]-q[k];
  }
  // verify the witness is a box face interior: exactly one clamped local coordinate
  mjtNum inv[9], cloc[3], rel[3];
  mju_transpose(inv, mat2, 3, 3);
  mju_sub3(rel, w2, p2);
  mju_mulMatVec3(cloc, inv, rel);
  int clamps = 0, jface = -1;
  for (int k = 0; k < 3; ++k) {
    if (mju_abs(mju_abs(cloc[k])-size2[k]) < 1e-9) {
      clamps++;
      jface = k;
    } else if (mju_abs(cloc[k]) >= size2[k]-1e-4) {
          return mjADJ_NON_SMOOTH;
    }
  }
  if (clamps != 1)     return mjADJ_NON_SMOOTH;
  mjtNum x1 = 0;
  int x1_free = 0;
  if (h1 > 0) {
    mjtNum dif1[3];
    mju_sub3(dif1, q, p1);
    mjtNum ma = mju_dot3(s1, s1);
    if (ma < mjMINVAL)     return mjADJ_NON_SMOOTH;
    x1 = mju_dot3(s1, dif1)/ma;
    if (mju_abs(x1) <= 1-1e-6) {
      // interior axis point: perpendicular-foot condition s1 . (w2 - q) = 0
      x1_free = 1;
      if (mju_abs(mju_dot3(s1, g12)) > 1e-9*(1+mju_norm3(g12))) {
            return mjADJ_NON_SMOOTH;
      }
      if (mju_abs(mju_dot3(s1, nrm)) < 1e-6*mju_sqrt(ma))     return mjADJ_NON_SMOOTH;
    } else {
      // segment boundary: the contact is sphere-box at the capsule end; the
      // parameter is frozen (dx1 = 0) as long as the endpoint stays the argmin
      x1 = x1 < 0 ? -1 : 1;
      if (mju_dot3(s1, g12)*x1 < -1e-9*(1+mju_norm3(g12)))     return mjADJ_NON_SMOOTH;
    }
  }
  if (mju_abs(mju_abs(nrm[1])-.5) < 1e-3)     return mjADJ_NON_SMOOTH;
  int cone_contact = m->opt.cone == mjCONE_ELLIPTIC &&
                     d->efc_state[0] == mjCNSTRSTATE_CONE;
  if (cone_contact && !hinv) return mjADJ_INVALID_ARGUMENT;
  mjtNum cone_H[36];
  mjtNum cone_q[6];
  if (cone_contact) {
    int hessian_status = plane_sphere_cone_hessian(m, d, cone_H);
    if (hessian_status != mjADJ_OK) return hessian_status;
    for (int row = 0; row < d->nefc; ++row) {
      cone_q[row] = mju_dot(cone_H+d->nefc*row, hinv, d->nefc);
    }
  }
  mjtNum rcol[3] = {mat2[jface], mat2[3+jface], mat2[6+jface]};
  (void)rcol;
  for (int joint = 0; joint < m->njnt; ++joint) {
    int joint_body = m->jnt_bodyid[joint];
    int free_joint = m->jnt_type[joint] == mjJNT_FREE;
    int channels = free_joint ? 6 : 1;
    for (int channel = 0; channel < channels; ++channel) {
      int dof = m->jnt_dofadr[joint]+channel;
      int rotating = m->jnt_type[joint] == mjJNT_HINGE ||
                     (free_joint && channel >= 3);
      mjtNum axis[3] = {0, 0, 0};
      if (free_joint && channel < 3) {
        axis[channel] = 1;
      } else if (free_joint) {
        mju_copy(axis, d->cdof+6*dof, 3);
      } else {
        mju_copy(axis, d->xaxis+3*joint, 3);
      }
      const mjtNum* anchor = d->xanchor+3*joint;
      // per-side pose derivatives
      mjtNum dp1[3] = {0, 0, 0}, dp2[3] = {0, 0, 0};
      mjtNum da1[3] = {0, 0, 0}, dtheta[3] = {0, 0, 0};
      int moves1 = 0, moves2 = 0;
      int ancestor = body1;
      while (ancestor != joint_body && ancestor > 0) {
        ancestor = m->body_parentid[ancestor];
      }
      moves1 = ancestor == joint_body;
      ancestor = body2;
      while (ancestor != joint_body && ancestor > 0) {
        ancestor = m->body_parentid[ancestor];
      }
      moves2 = ancestor == joint_body;
      if (moves1) {
        if (rotating) {
          mjtNum offset[3];
          mju_sub3(offset, p1, anchor);
          mju_cross(dp1, axis, offset);
          mju_cross(da1, axis, a1);
        } else {
          mju_copy(dp1, axis, 3);
        }
      }
      if (moves2) {
        if (rotating) {
          mjtNum offset[3];
          mju_sub3(offset, p2, anchor);
          mju_cross(dp2, axis, offset);
          mju_copy(dtheta, axis, 3);
        } else {
          mju_copy(dp2, axis, 3);
        }
      }
      mjtNum ds1[3];
      mju_scl3(ds1, da1, h1);
      mjtNum dq0[3], dcloc[3], dw2_0[3], relqp[3];
      for (int k = 0; k < 3; ++k) {
        dq0[k] = dp1[k]+ds1[k]*x1;
        relqp[k] = q[k]-p2[k];
      }
      // dcloc = R^T (dq0 - dp2 - dtheta x (q - p2))
      mjtNum tmp3a[3], tmp3b[3];
      mju_cross(tmp3a, dtheta, relqp);
      for (int k = 0; k < 3; ++k) {
        tmp3b[k] = dq0[k]-dp2[k]-tmp3a[k];
      }
      mju_mulMatTVec3(dcloc, mat2, tmp3b);
      // dw2_0 = dtheta x (R cloc) + R P dcloc + dp2 ; P keeps non-clamped coords
      mjtNum rcloc[3], pdcloc[3];
      mju_mulMatVec3(rcloc, mat2, cloc);
      for (int k = 0; k < 3; ++k) {
        pdcloc[k] = k == jface ? 0 : dcloc[k];
      }
      mjtNum rpdcloc[3];
      mju_mulMatVec3(rpdcloc, mat2, pdcloc);
      mju_cross(tmp3a, dtheta, rcloc);
      for (int k = 0; k < 3; ++k) {
        dw2_0[k] = tmp3a[k]+rpdcloc[k]+dp2[k];
      }
      // capsule-axis parameter derivative from s1 . (w2 - q) = 0 (interior only):
      // dx1 = (s1.dq0 - s1.dw2_0) / (s1.n)^2 ; J = R P R^T = I - n n^T
      mjtNum dx1 = 0;
      if (x1_free) {
        mjtNum denom = mju_dot3(s1, nrm);
        denom *= denom;
        if (denom < 1e-12)     return mjADJ_NON_SMOOTH;
        mjtNum rhs = mju_dot3(s1, dq0)-mju_dot3(s1, dw2_0);
        dx1 = rhs/denom;
      }
      // dg = dw2_0 - dq0 + (J - I) s1 dx1 = dw2_0 - dq0 - (s1.n) n dx1
      mjtNum dg[3], dn[3];
      {
        mjtNum s1n = mju_dot3(s1, nrm);
        for (int k = 0; k < 3; ++k) {
          dg[k] = dw2_0[k]-dq0[k]-s1n*nrm[k]*dx1;
        }
      }
      mjtNum dL = mju_dot3(nrm, dg);
      for (int k = 0; k < 3; ++k) {
        dn[k] = (dg[k]-nrm[k]*dL)/L;
      }
      // makeFrame default tangent
      mjtNum t0[3] = {0, 1, 0};
      if (mju_abs(nrm[1]) >= .5) {
        t0[0] = 0;
        t0[1] = 0;
        t0[2] = 1;
      }
      mjtNum c = mju_dot3(nrm, t0);
      mjtNum uvec[3], dtan[3];
      for (int k = 0; k < 3; ++k) {
        uvec[k] = t0[k]-nrm[k]*c;
      }
      mjtNum us = mju_norm3(uvec);
      if (us < mjMINVAL)     return mjADJ_NON_SMOOTH;
      mjtNum dc = mju_dot3(dn, t0);
      mjtNum du[3];
      for (int k = 0; k < 3; ++k) {
        du[k] = -dn[k]*c-nrm[k]*dc;
      }
      mjtNum dus = mju_dot3(uvec, du)/us;
      for (int k = 0; k < 3; ++k) {
        dtan[k] = du[k]/us-uvec[k]*dus/(us*us);
      }
      const mjtNum* tan = con->frame+3;
      mjtNum dbit[3], tmp_bit[3];
      mju_cross(tmp_bit, dn, tan);
      mju_cross(dbit, nrm, dtan);
      mju_addTo3(dbit, tmp_bit);
      mjtNum dw1[3], dpos[3];
      for (int k = 0; k < 3; ++k) {
        dw1[k] = dq0[k]+s1[k]*dx1;
        dpos[k] = dw1[k]+dn[k]*(r1+.5*con->dist)+nrm[k]*.5*dL;
      }
      mjtNum ddist = dL;
      for (int row = 0; row < m->nv; ++row) {
        int row_body = m->dof_bodyid[row];
        int row_joint = m->dof_jntid[row];
        mjtJoint row_type = (mjtJoint)m->jnt_type[row_joint];
        int row_rotating = row_type == mjJNT_HINGE ||
                           (row_type == mjJNT_FREE && row >= 3);
        int in1 = 0, in2 = 0;
        ancestor = body1;
        while (ancestor != row_body && ancestor > 0) {
          ancestor = m->body_parentid[ancestor];
        }
        in1 = ancestor == row_body;
        ancestor = body2;
        while (ancestor != row_body && ancestor > 0) {
          ancestor = m->body_parentid[ancestor];
        }
        in2 = ancestor == row_body;
        if (!in1 && !in2) continue;
        int moved = 0;
        ancestor = row_body;
        while (ancestor != joint_body && ancestor > 0) {
          ancestor = m->body_parentid[ancestor];
        }
        moved = ancestor == joint_body;
        const mjtNum* cdof = d->cdof+6*row;
        mjtNum delta_axis[3] = {0, 0, 0};
        mjtNum delta_lin[3] = {0, 0, 0};
        if (moved && rotating && !(row_type == mjJNT_FREE && row < 3)) {
          mjtNum row_axis[3];
          if (row_rotating) {
            mju_copy(row_axis, cdof, 3);
          } else {
            mju_copy(row_axis, d->xaxis+3*row_joint, 3);
          }
          mju_cross(delta_axis, axis, row_axis);
        }
        if (row_rotating) {
          mjtNum dcom[3] = {0, 0, 0};
          int row_root = m->body_rootid[row_body];
          for (int body = 1; body < m->nbody; ++body) {
            if (m->body_rootid[body] != row_root) continue;
            ancestor = body;
            while (ancestor != joint_body && ancestor > 0) {
              ancestor = m->body_parentid[ancestor];
            }
            if (ancestor != joint_body) continue;
            mjtNum dip[3];
            if (rotating) {
              mjtNum offset[3];
              mju_sub3(offset, d->xipos+3*body, anchor);
              mju_cross(dip, axis, offset);
            } else {
              mju_copy(dip, axis, 3);
            }
            mju_addToScl3(dcom, dip,
                          m->body_mass[body]/m->body_subtreemass[row_root]);
          }
          mjtNum danchor[3] = {0, 0, 0};
          if (moved) {
            if (rotating) {
              mjtNum offset[3];
              mju_sub3(offset, d->xanchor+3*row_joint, anchor);
              mju_cross(danchor, axis, offset);
            } else {
              mju_copy(danchor, axis, 3);
            }
          }
          const mjtNum* com = d->subtree_com+3*row_root;
          mjtNum row_lever[3], dlever[3], tmpv[3];
          mju_sub3(row_lever, com, d->xanchor+3*row_joint);
          mju_sub3(dlever, dcom, danchor);
          mju_cross(delta_lin, delta_axis, row_lever);
          mju_cross(tmpv, cdof, dlever);
          mju_addTo3(delta_lin, tmpv);
        } else if (row_type != mjJNT_FREE) {
          mju_copy(delta_lin, delta_axis, 3);
        }
        mjtNum jac1[3] = {0, 0, 0}, jac2[3] = {0, 0, 0};
        mjtNum djac1[3] = {0, 0, 0}, djac2[3] = {0, 0, 0};
        if (in1) {
          const mjtNum* com = d->subtree_com+3*m->body_rootid[body1];
          mjtNum dcom1[3] = {0, 0, 0};
          int root1 = m->body_rootid[body1];
          for (int body = 1; body < m->nbody; ++body) {
            if (m->body_rootid[body] != root1) continue;
            ancestor = body;
            while (ancestor != joint_body && ancestor > 0) {
              ancestor = m->body_parentid[ancestor];
            }
            if (ancestor != joint_body) continue;
            mjtNum dip[3];
            if (rotating) {
              mjtNum offset[3];
              mju_sub3(offset, d->xipos+3*body, anchor);
              mju_cross(dip, axis, offset);
            } else {
              mju_copy(dip, axis, 3);
            }
            mju_addToScl3(dcom1, dip,
                          m->body_mass[body]/m->body_subtreemass[root1]);
          }
          mjtNum lever[3], dlev[3], tmpv[3];
          mju_sub3(lever, con->pos, com);
          mju_sub3(dlev, dpos, dcom1);
          mju_cross(jac1, cdof, lever);
          mju_addTo3(jac1, cdof+3);
          mju_cross(djac1, delta_axis, lever);
          mju_cross(tmpv, cdof, dlev);
          mju_addTo3(djac1, tmpv);
          mju_addTo3(djac1, delta_lin);
        }
        if (in2) {
          const mjtNum* com = d->subtree_com+3*m->body_rootid[body2];
          mjtNum dcom2[3] = {0, 0, 0};
          int root2 = m->body_rootid[body2];
          for (int body = 1; body < m->nbody; ++body) {
            if (m->body_rootid[body] != root2) continue;
            ancestor = body;
            while (ancestor != joint_body && ancestor > 0) {
              ancestor = m->body_parentid[ancestor];
            }
            if (ancestor != joint_body) continue;
            mjtNum dip[3];
            if (rotating) {
              mjtNum offset[3];
              mju_sub3(offset, d->xipos+3*body, anchor);
              mju_cross(dip, axis, offset);
            } else {
              mju_copy(dip, axis, 3);
            }
            mju_addToScl3(dcom2, dip,
                          m->body_mass[body]/m->body_subtreemass[root2]);
          }
          mjtNum lever[3], dlev[3], tmpv[3];
          mju_sub3(lever, con->pos, com);
          mju_sub3(dlev, dpos, dcom2);
          mju_cross(jac2, cdof, lever);
          mju_addTo3(jac2, cdof+3);
          mju_cross(djac2, delta_axis, lever);
          mju_cross(tmpv, cdof, dlev);
          mju_addTo3(djac2, tmpv);
          mju_addTo3(djac2, delta_lin);
        }
        mjtNum jacD[3], djacD[3], angD[3], dangD[3];
        for (int k = 0; k < 3; ++k) {
          jacD[k] = jac2[k]-jac1[k];
          djacD[k] = djac2[k]-djac1[k];
          angD[k] = (in2 ? cdof[k] : 0)-(in1 ? cdof[k] : 0);
          dangD[k] = (in2 ? delta_axis[k] : 0)-(in1 ? delta_axis[k] : 0);
        }
        for (int constraint = 0; constraint < d->nefc; ++constraint) {
          // pyramidal pair rows are n + sign*fr*row_k combinations, so every row
          // carries the (varying) normal direction; elliptic torsional rows are
          // purely rotational
          mjtNum delta_jac = mju_dot3(dn, jacD)+mju_dot3(nrm, djacD);
          if (m->opt.cone == mjCONE_ELLIPTIC) {
            if (constraint > 0) {
              int direction = constraint < 3 ? constraint : constraint-3;
              const mjtNum* dir = con->frame+3*direction;
              mjtNum ddir[3];
              if (direction == 0) {
                mju_copy(ddir, dn, 3);
              } else if (direction == 1) {
                mju_copy(ddir, dtan, 3);
              } else {
                mju_copy(ddir, dbit, 3);
              }
              if (constraint < 3) {
                delta_jac = mju_dot3(ddir, jacD)+mju_dot3(dir, djacD);
              } else {
                delta_jac = mju_dot3(ddir, angD)+mju_dot3(dir, dangD);
              }
            }
          } else if (dim > 1) {
            int tangent = constraint/2+1;
            mjtNum sign = constraint%2 ? -1 : 1;
            mjtNum friction = con->friction[tangent-1];
            if (tangent < 3) {
              const mjtNum* td = con->frame+3*tangent;
              mjtNum dtd[3];
              if (tangent == 1) {
                mju_copy(dtd, dtan, 3);
              } else {
                mju_copy(dtd, dbit, 3);
              }
              delta_jac += sign*friction*(mju_dot3(dtd, jacD)+mju_dot3(td, djacD));
            } else {
              int direction = tangent-3;
              const mjtNum* dir = con->frame+3*direction;
              mjtNum ddir[3];
              if (direction == 0) {
                mju_copy(ddir, dn, 3);
              } else if (direction == 1) {
                mju_copy(ddir, dtan, 3);
              } else {
                mju_copy(ddir, dbit, 3);
              }
              delta_jac += sign*friction*(mju_dot3(ddir, angD)+mju_dot3(dir, dangD));
            }
          }
          mjtNum acceleration_bar = cone_contact ? -cone_q[constraint] : mu[constraint];
          dbar_qpos[dof] += delta_jac*(y[row]*lambda[constraint]+
              acceleration_bar*(qacc_adv[row]+d->efc_KBIP[4*constraint+1]*d->qvel[row]));
        }
      }
      for (int constraint = 0; constraint < d->nefc; ++constraint) {
        mjtNum K = d->efc_KBIP[4*constraint];
        mjtNum imp = d->efc_KBIP[4*constraint+2];
        mjtNum impP = d->efc_KBIP[4*constraint+3];
        mjtNum dR_ddist = 0;
        if (imp > 0 && imp < 1 && d->efc_R[constraint] > mjMINVAL) {
          dR_ddist = -d->efc_R[constraint]*impP/(imp*(1-imp));
        }
        mjtNum position_gain = K*(imp+impP*(d->efc_pos[constraint]-d->efc_margin[constraint]));
        dbar_qpos[dof] += (cone_contact ? -cone_q[constraint] : mu[constraint])*
                          position_gain*ddist;
        if (cone_contact) {
          mjtNum scale = -lambda[constraint]*impP/(imp*(1-imp))*ddist;
          dbar_qpos[dof] += impedance_sign*hinv[constraint]*scale;
        } else {
          dbar_qpos[dof] += mu[constraint]*lambda[constraint]*dR_ddist*ddist;
        }
      }
    }
  }
  return mjADJ_OK;
}

// VJP of one sphere-cylinder contact (single contact). Mirrors mjc_SphereCylinder:
// side branch (witness = axis point at the sphere-center projection, radius rcyl),
// cap branch (|x| >= h: plane-sphere against the cap, n = -sign(x)*axis), corner
// branch (witness = rim corner, non-material under spin). w1 = sphere center with
// pos = w1 + n*(r1 + dist/2) in all branches.
static int sphere_cylinder_pos_adjoint(const mjModel* m, const mjData* d,
                                       const mjtNum* y, const mjtNum* lambda,
                                       const mjtNum* mu, const mjtNum* hinv,
                                       const mjtNum* qacc_adv,
                                       mjtNum impedance_sign, mjtNum* dbar_qpos) {
  if (!m || !d || (m->nv && (!y || !lambda || !mu || !qacc_adv || !dbar_qpos))) {
    return mjADJ_INVALID_ARGUMENT;
  }
  int dim = d->ncon == 1 ? d->contact[0].dim : 0;
  if (d->ncon != 1 || d->ne || d->nf ||
      !((dim == 1 && d->nefc == 1) ||
        (dim >= 3 && m->opt.cone == mjCONE_PYRAMIDAL && d->nefc == 2*(dim-1)) ||
        (dim >= 3 && dim <= 6 && m->opt.cone == mjCONE_ELLIPTIC && d->nefc == dim))) {
    return mjADJ_UNSUPPORTED;
  }
  const mjContact* con = d->contact;
  int sph = con->geom1;
  int cyl = con->geom2;
  if (m->geom_type[sph] != mjGEOM_SPHERE || m->geom_type[cyl] != mjGEOM_CYLINDER) {
    return mjADJ_UNSUPPORTED;
  }
  int body1 = m->geom_bodyid[sph];
  int body2 = m->geom_bodyid[cyl];
  mjtNum r1 = m->geom_size[3*sph];
  mjtNum rc = m->geom_size[3*cyl];
  mjtNum h = m->geom_size[3*cyl+1];
  const mjtNum* p1 = d->geom_xpos+3*sph;
  const mjtNum* p2 = d->geom_xpos+3*cyl;
  const mjtNum* mat2 = d->geom_xmat+9*cyl;
  mjtNum a2[3] = {mat2[2], mat2[5], mat2[8]};
  const mjtNum* nrm = con->frame;
  // branch classification from the stock geometry
  mjtNum rel[3];
  mju_sub3(rel, p1, p2);
  mjtNum x = mju_dot3(a2, rel);
  mjtNum pproj[3];
  for (int k = 0; k < 3; ++k) {
    pproj[k] = rel[k]-a2[k]*x;
  }
  mjtNum ppsqr = mju_dot3(pproj, pproj);
  int branch;  // 0 side, 1 cap, 2 corner
  if (mju_abs(x) <= h-1e-4 && ppsqr >= (rc+1e-4)*(rc+1e-4)) {
    branch = 0;
  } else if (mju_abs(x) >= h+1e-4 && ppsqr <= (rc-1e-4)*(rc-1e-4)) {
    branch = 1;
  } else if (mju_abs(x) >= h+1e-4 && ppsqr >= (rc+1e-4)*(rc+1e-4)) {
    branch = 2;
  } else {
        return mjADJ_NON_SMOOTH;
  }
  mjtNum sigma = x >= 0 ? 1 : -1;
  mjtNum r2eff = branch == 0 ? rc : 0;
  mjtNum L = con->dist+r1+r2eff;
  if (L < mjMINVAL)     return mjADJ_NON_SMOOTH;
  // reconstruct w1 (sphere center) and the branch witness w2
  mjtNum w1[3], w2[3];
  for (int k = 0; k < 3; ++k) {
    w1[k] = con->pos[k]-nrm[k]*(r1+.5*con->dist);
    w2[k] = con->pos[k]+nrm[k]*(.5*con->dist+r2eff);
  }
  {
    mjtNum chk[3];
    if (branch == 0) {
      for (int k = 0; k < 3; ++k) {
        chk[k] = p2[k]+a2[k]*x;
      }
    } else if (branch == 1) {
      for (int k = 0; k < 3; ++k) {
        chk[k] = p2[k]+a2[k]*sigma*h+pproj[k];
      }
    } else {
      mjtNum plen = mju_sqrt(ppsqr);
      for (int k = 0; k < 3; ++k) {
        chk[k] = p2[k]+a2[k]*sigma*h+rc*pproj[k]/plen;
      }
    }
    if (mju_dist3(chk, w2) > 1e-9*(1+mju_norm3(w2)))     return mjADJ_NON_SMOOTH;
    if (mju_dist3(w1, p1) > 1e-9*(1+mju_norm3(p1)))     return mjADJ_NON_SMOOTH;
  }
  if (branch == 1 && mju_dist3(nrm, (mjtNum[]){-sigma*a2[0], -sigma*a2[1], -sigma*a2[2]}) > 1e-9) {
        return mjADJ_NON_SMOOTH;
  }
  if (mju_abs(mju_abs(nrm[1])-.5) < 1e-3)     return mjADJ_NON_SMOOTH;
  int cone_contact = m->opt.cone == mjCONE_ELLIPTIC &&
                     d->efc_state[0] == mjCNSTRSTATE_CONE;
  if (cone_contact && !hinv) return mjADJ_INVALID_ARGUMENT;
  mjtNum cone_H[36];
  mjtNum cone_q[6];
  if (cone_contact) {
    int hessian_status = plane_sphere_cone_hessian(m, d, cone_H);
    if (hessian_status != mjADJ_OK) return hessian_status;
    for (int row = 0; row < d->nefc; ++row) {
      cone_q[row] = mju_dot(cone_H+d->nefc*row, hinv, d->nefc);
    }
  }
  mjtNum pplen = mju_sqrt(ppsqr);
  if (branch == 2 && pplen < mjMINVAL)     return mjADJ_NON_SMOOTH;
  for (int joint = 0; joint < m->njnt; ++joint) {
    int joint_body = m->jnt_bodyid[joint];
    int free_joint = m->jnt_type[joint] == mjJNT_FREE;
    int channels = free_joint ? 6 : 1;
    for (int channel = 0; channel < channels; ++channel) {
      int dof = m->jnt_dofadr[joint]+channel;
      int rotating = m->jnt_type[joint] == mjJNT_HINGE ||
                     (free_joint && channel >= 3);
      mjtNum axis[3] = {0, 0, 0};
      if (free_joint && channel < 3) {
        axis[channel] = 1;
      } else if (free_joint) {
        mju_copy(axis, d->cdof+6*dof, 3);
      } else {
        mju_copy(axis, d->xaxis+3*joint, 3);
      }
      const mjtNum* anchor = d->xanchor+3*joint;
      mjtNum dp1[3] = {0, 0, 0}, dp2[3] = {0, 0, 0}, da2[3] = {0, 0, 0};
      int moves1 = 0, moves2 = 0;
      int ancestor = body1;
      while (ancestor != joint_body && ancestor > 0) {
        ancestor = m->body_parentid[ancestor];
      }
      moves1 = ancestor == joint_body;
      ancestor = body2;
      while (ancestor != joint_body && ancestor > 0) {
        ancestor = m->body_parentid[ancestor];
      }
      moves2 = ancestor == joint_body;
      if (moves1) {
        if (rotating) {
          mjtNum offset[3];
          mju_sub3(offset, p1, anchor);
          mju_cross(dp1, axis, offset);
        } else {
          mju_copy(dp1, axis, 3);
        }
      }
      if (moves2) {
        if (rotating) {
          mjtNum offset[3];
          mju_sub3(offset, p2, anchor);
          mju_cross(dp2, axis, offset);
          mju_cross(da2, axis, a2);
        } else {
          mju_copy(dp2, axis, 3);
        }
      }
      // dx: projection coordinate x = a2.(p1-p2) is explicit (no condition solve)
      mjtNum dx = mju_dot3(da2, rel)+mju_dot3(a2, dp1)-mju_dot3(a2, dp2);
      mjtNum dw1[3], dw2[3], dn[3], ddist;
      mju_copy(dw1, dp1, 3);
      if (branch == 0) {
        for (int k = 0; k < 3; ++k) {
          dw2[k] = dp2[k]+da2[k]*x+a2[k]*dx;
        }
        mjtNum dg[3];
        for (int k = 0; k < 3; ++k) {
          dg[k] = dw2[k]-dw1[k];
        }
        mjtNum dL = mju_dot3(nrm, dg);
        ddist = dL;
        for (int k = 0; k < 3; ++k) {
          dn[k] = (dg[k]-nrm[k]*dL)/L;
        }
      } else if (branch == 1) {
        ddist = sigma*dx;
        mjtNum dcap[3], dpp[3];
        for (int k = 0; k < 3; ++k) {
          dcap[k] = dp2[k]+sigma*h*da2[k];
          dpp[k] = dp1[k]-dp2[k]-da2[k]*x-a2[k]*dx;
        }
        for (int k = 0; k < 3; ++k) {
          dw2[k] = dcap[k]+dpp[k];
          dn[k] = -sigma*da2[k];
        }
      } else {
        mjtNum dpp[3], dphat[3];
        for (int k = 0; k < 3; ++k) {
          dpp[k] = dp1[k]-dp2[k]-da2[k]*x-a2[k]*dx;
        }
        mjtNum ppd = mju_dot3(pproj, dpp)/pplen;
        for (int k = 0; k < 3; ++k) {
          dphat[k] = (dpp[k]-pproj[k]*ppd/pplen)/pplen;
          dw2[k] = dp2[k]+sigma*h*da2[k]+rc*dphat[k];
        }
        mjtNum dg[3];
        for (int k = 0; k < 3; ++k) {
          dg[k] = dw2[k]-dw1[k];
        }
        mjtNum dL = mju_dot3(nrm, dg);
        ddist = dL;
        for (int k = 0; k < 3; ++k) {
          dn[k] = (dg[k]-nrm[k]*dL)/L;
        }
      }
      mjtNum dpos[3];
      for (int k = 0; k < 3; ++k) {
        dpos[k] = dw1[k]+dn[k]*(r1+.5*con->dist)+nrm[k]*.5*ddist;
      }
      // makeFrame default tangent
      mjtNum t0[3] = {0, 1, 0};
      if (mju_abs(nrm[1]) >= .5) {
        t0[0] = 0;
        t0[1] = 0;
        t0[2] = 1;
      }
      mjtNum c = mju_dot3(nrm, t0);
      mjtNum uvec[3], dtan[3];
      for (int k = 0; k < 3; ++k) {
        uvec[k] = t0[k]-nrm[k]*c;
      }
      mjtNum us = mju_norm3(uvec);
      if (us < mjMINVAL)     return mjADJ_NON_SMOOTH;
      mjtNum dc = mju_dot3(dn, t0);
      mjtNum du[3];
      for (int k = 0; k < 3; ++k) {
        du[k] = -dn[k]*c-nrm[k]*dc;
      }
      mjtNum dus = mju_dot3(uvec, du)/us;
      for (int k = 0; k < 3; ++k) {
        dtan[k] = du[k]/us-uvec[k]*dus/(us*us);
      }
      const mjtNum* tan = con->frame+3;
      mjtNum dbit[3], tmp_bit[3];
      mju_cross(tmp_bit, dn, tan);
      mju_cross(dbit, nrm, dtan);
      mju_addTo3(dbit, tmp_bit);
      for (int row = 0; row < m->nv; ++row) {
        int row_body = m->dof_bodyid[row];
        int row_joint = m->dof_jntid[row];
        mjtJoint row_type = (mjtJoint)m->jnt_type[row_joint];
        int row_rotating = row_type == mjJNT_HINGE ||
                           (row_type == mjJNT_FREE && row >= 3);
        int in1 = 0, in2 = 0;
        ancestor = body1;
        while (ancestor != row_body && ancestor > 0) {
          ancestor = m->body_parentid[ancestor];
        }
        in1 = ancestor == row_body;
        ancestor = body2;
        while (ancestor != row_body && ancestor > 0) {
          ancestor = m->body_parentid[ancestor];
        }
        in2 = ancestor == row_body;
        if (!in1 && !in2) continue;
        int moved = 0;
        ancestor = row_body;
        while (ancestor != joint_body && ancestor > 0) {
          ancestor = m->body_parentid[ancestor];
        }
        moved = ancestor == joint_body;
        const mjtNum* cdof = d->cdof+6*row;
        mjtNum delta_axis[3] = {0, 0, 0};
        mjtNum delta_lin[3] = {0, 0, 0};
        if (moved && rotating && !(row_type == mjJNT_FREE && row < 3)) {
          mjtNum row_axis[3];
          if (row_rotating) {
            mju_copy(row_axis, cdof, 3);
          } else {
            mju_copy(row_axis, d->xaxis+3*row_joint, 3);
          }
          mju_cross(delta_axis, axis, row_axis);
        }
        if (row_rotating) {
          mjtNum dcom[3] = {0, 0, 0};
          int row_root = m->body_rootid[row_body];
          for (int body = 1; body < m->nbody; ++body) {
            if (m->body_rootid[body] != row_root) continue;
            ancestor = body;
            while (ancestor != joint_body && ancestor > 0) {
              ancestor = m->body_parentid[ancestor];
            }
            if (ancestor != joint_body) continue;
            mjtNum dip[3];
            if (rotating) {
              mjtNum offset[3];
              mju_sub3(offset, d->xipos+3*body, anchor);
              mju_cross(dip, axis, offset);
            } else {
              mju_copy(dip, axis, 3);
            }
            mju_addToScl3(dcom, dip,
                          m->body_mass[body]/m->body_subtreemass[row_root]);
          }
          mjtNum danchor[3] = {0, 0, 0};
          if (moved) {
            if (rotating) {
              mjtNum offset[3];
              mju_sub3(offset, d->xanchor+3*row_joint, anchor);
              mju_cross(danchor, axis, offset);
            } else {
              mju_copy(danchor, axis, 3);
            }
          }
          const mjtNum* com = d->subtree_com+3*row_root;
          mjtNum row_lever[3], dlever[3], tmpv[3];
          mju_sub3(row_lever, com, d->xanchor+3*row_joint);
          mju_sub3(dlever, dcom, danchor);
          mju_cross(delta_lin, delta_axis, row_lever);
          mju_cross(tmpv, cdof, dlever);
          mju_addTo3(delta_lin, tmpv);
        } else if (row_type != mjJNT_FREE) {
          mju_copy(delta_lin, delta_axis, 3);
        }
        mjtNum jac1[3] = {0, 0, 0}, jac2[3] = {0, 0, 0};
        mjtNum djac1[3] = {0, 0, 0}, djac2[3] = {0, 0, 0};
        if (in1) {
          const mjtNum* com = d->subtree_com+3*m->body_rootid[body1];
          mjtNum dcom1[3] = {0, 0, 0};
          int root1 = m->body_rootid[body1];
          for (int body = 1; body < m->nbody; ++body) {
            if (m->body_rootid[body] != root1) continue;
            ancestor = body;
            while (ancestor != joint_body && ancestor > 0) {
              ancestor = m->body_parentid[ancestor];
            }
            if (ancestor != joint_body) continue;
            mjtNum dip[3];
            if (rotating) {
              mjtNum offset[3];
              mju_sub3(offset, d->xipos+3*body, anchor);
              mju_cross(dip, axis, offset);
            } else {
              mju_copy(dip, axis, 3);
            }
            mju_addToScl3(dcom1, dip,
                          m->body_mass[body]/m->body_subtreemass[root1]);
          }
          mjtNum lever[3], dlev[3], tmpv[3];
          mju_sub3(lever, con->pos, com);
          mju_sub3(dlev, dpos, dcom1);
          mju_cross(jac1, cdof, lever);
          mju_addTo3(jac1, cdof+3);
          mju_cross(djac1, delta_axis, lever);
          mju_cross(tmpv, cdof, dlev);
          mju_addTo3(djac1, tmpv);
          mju_addTo3(djac1, delta_lin);
        }
        if (in2) {
          const mjtNum* com = d->subtree_com+3*m->body_rootid[body2];
          mjtNum dcom2[3] = {0, 0, 0};
          int root2 = m->body_rootid[body2];
          for (int body = 1; body < m->nbody; ++body) {
            if (m->body_rootid[body] != root2) continue;
            ancestor = body;
            while (ancestor != joint_body && ancestor > 0) {
              ancestor = m->body_parentid[ancestor];
            }
            if (ancestor != joint_body) continue;
            mjtNum dip[3];
            if (rotating) {
              mjtNum offset[3];
              mju_sub3(offset, d->xipos+3*body, anchor);
              mju_cross(dip, axis, offset);
            } else {
              mju_copy(dip, axis, 3);
            }
            mju_addToScl3(dcom2, dip,
                          m->body_mass[body]/m->body_subtreemass[root2]);
          }
          mjtNum lever[3], dlev[3], tmpv[3];
          mju_sub3(lever, con->pos, com);
          mju_sub3(dlev, dpos, dcom2);
          mju_cross(jac2, cdof, lever);
          mju_addTo3(jac2, cdof+3);
          mju_cross(djac2, delta_axis, lever);
          mju_cross(tmpv, cdof, dlev);
          mju_addTo3(djac2, tmpv);
          mju_addTo3(djac2, delta_lin);
        }
        mjtNum jacD[3], djacD[3], angD[3], dangD[3];
        for (int k = 0; k < 3; ++k) {
          jacD[k] = jac2[k]-jac1[k];
          djacD[k] = djac2[k]-djac1[k];
          angD[k] = (in2 ? cdof[k] : 0)-(in1 ? cdof[k] : 0);
          dangD[k] = (in2 ? delta_axis[k] : 0)-(in1 ? delta_axis[k] : 0);
        }
        for (int constraint = 0; constraint < d->nefc; ++constraint) {
          mjtNum delta_jac = mju_dot3(dn, jacD)+mju_dot3(nrm, djacD);
          if (m->opt.cone == mjCONE_ELLIPTIC) {
            if (constraint > 0) {
              int direction = constraint < 3 ? constraint : constraint-3;
              const mjtNum* dir = con->frame+3*direction;
              mjtNum ddir[3];
              if (direction == 0) {
                mju_copy(ddir, dn, 3);
              } else if (direction == 1) {
                mju_copy(ddir, dtan, 3);
              } else {
                mju_copy(ddir, dbit, 3);
              }
              if (constraint < 3) {
                delta_jac = mju_dot3(ddir, jacD)+mju_dot3(dir, djacD);
              } else {
                delta_jac = mju_dot3(ddir, angD)+mju_dot3(dir, dangD);
              }
            }
          } else if (dim > 1) {
            int tangent = constraint/2+1;
            mjtNum sign = constraint%2 ? -1 : 1;
            mjtNum friction = con->friction[tangent-1];
            if (tangent < 3) {
              const mjtNum* td = con->frame+3*tangent;
              mjtNum dtd[3];
              if (tangent == 1) {
                mju_copy(dtd, dtan, 3);
              } else {
                mju_copy(dtd, dbit, 3);
              }
              delta_jac += sign*friction*(mju_dot3(dtd, jacD)+mju_dot3(td, djacD));
            } else {
              int direction = tangent-3;
              const mjtNum* dir = con->frame+3*direction;
              mjtNum ddir[3];
              if (direction == 0) {
                mju_copy(ddir, dn, 3);
              } else if (direction == 1) {
                mju_copy(ddir, dtan, 3);
              } else {
                mju_copy(ddir, dbit, 3);
              }
              delta_jac += sign*friction*(mju_dot3(ddir, angD)+mju_dot3(dir, dangD));
            }
          }
          mjtNum acceleration_bar = cone_contact ? -cone_q[constraint] : mu[constraint];
          dbar_qpos[dof] += delta_jac*(y[row]*lambda[constraint]+
              acceleration_bar*(qacc_adv[row]+d->efc_KBIP[4*constraint+1]*d->qvel[row]));
        }
      }
      for (int constraint = 0; constraint < d->nefc; ++constraint) {
        mjtNum K = d->efc_KBIP[4*constraint];
        mjtNum imp = d->efc_KBIP[4*constraint+2];
        mjtNum impP = d->efc_KBIP[4*constraint+3];
        mjtNum dR_ddist = 0;
        if (imp > 0 && imp < 1 && d->efc_R[constraint] > mjMINVAL) {
          dR_ddist = -d->efc_R[constraint]*impP/(imp*(1-imp));
        }
        mjtNum position_gain = K*(imp+impP*(d->efc_pos[constraint]-d->efc_margin[constraint]));
        dbar_qpos[dof] += (cone_contact ? -cone_q[constraint] : mu[constraint])*
                          position_gain*ddist;
        if (cone_contact) {
          mjtNum scale = -lambda[constraint]*impP/(imp*(1-imp))*ddist;
          dbar_qpos[dof] += impedance_sign*hinv[constraint]*scale;
        } else {
          dbar_qpos[dof] += mu[constraint]*lambda[constraint]*dR_ddist*ddist;
        }
      }
    }
  }
  return mjADJ_OK;
}

int mjd_massPosAdjoint(const mjModel* m, mjData* d,
                       const mjtNum* y, const mjtNum* w, mjtNum* dbar_qpos) {
  if (!m || !d || (m->nv && (!y || !w || !dbar_qpos))) {
    return mjADJ_INVALID_ARGUMENT;
  }
  if (!scalar_tree_kind(m)) return mjADJ_UNSUPPORTED;
  int free_root = scalar_tree_kind(m) == 2;
  size_t tangent_bytes = (size_t)m->nv * sizeof(mjtNum);
  if (ranges_overlap(y, tangent_bytes, dbar_qpos, tangent_bytes) ||
      ranges_overlap(w, tangent_bytes, dbar_qpos, tangent_bytes)) {
    return mjADJ_INVALID_ARGUMENT;
  }
  mju_zero(dbar_qpos, m->nv);
  mj_markStack(d);
  mjtNum* storage = mjSTACKALLOC(d, 33*m->nbody, mjtNum);
  mju_zero(storage, 33*m->nbody);
  mjtNum* angular_y = storage;
  mjtNum* linear_y = storage+3*m->nbody;
  mjtNum* angular_w = storage+6*m->nbody;
  mjtNum* linear_w = storage+9*m->nbody;
  mjtNum* delta_angular_all = storage+12*m->nbody;
  mjtNum* delta_linear_all = storage+15*m->nbody;
  mjtNum* delta_angular_w_all = storage+18*m->nbody;
  mjtNum* delta_linear_w_all = storage+21*m->nbody;
  mjtNum* force_all = storage+24*m->nbody;
  mjtNum* moment_all = storage+27*m->nbody;
  mjtNum* delta_pos = storage+30*m->nbody;
  for (int body = 1; body < m->nbody; ++body) {
    int parent = m->body_parentid[body];
    mjtNum offset[3];
    mju_sub3(offset, d->xpos+3*body, d->xpos+3*parent);
    mjtNum cross_parent[3];
    mju_cross(cross_parent, angular_y+3*parent, offset);
    mju_add3(linear_y+3*body, linear_y+3*parent, cross_parent);
    mju_copy(angular_y+3*body, angular_y+3*parent, 3);
    mju_cross(cross_parent, angular_w+3*parent, offset);
    mju_add3(linear_w+3*body, linear_w+3*parent, cross_parent);
    mju_copy(angular_w+3*body, angular_w+3*parent, 3);
    if (free_root && body == 1) {
      mju_mulMatVec3(angular_y+3*body, d->xmat+9*body, y+3);
      mju_copy(linear_y+3*body, y, 3);
      mju_mulMatVec3(angular_w+3*body, d->xmat+9*body, w+3);
      mju_copy(linear_w+3*body, w, 3);
    }
    if (m->body_jntnum[body] && !(free_root && body == 1)) {
      int joint = m->body_jntadr[body];
      int dof = m->jnt_dofadr[joint];
      mjtNum axis[3];
      mju_mulMatVec3(axis, d->xmat+9*body, m->jnt_axis+3*joint);
      if (m->jnt_type[joint] == mjJNT_HINGE) {
        mju_scl3(cross_parent, axis, y[dof]);
        mju_addTo3(angular_y+3*body, cross_parent);
        mju_scl3(cross_parent, axis, w[dof]);
        mju_addTo3(angular_w+3*body, cross_parent);
      } else {
        mju_scl3(cross_parent, axis, y[dof]);
        mju_addTo3(linear_y+3*body, cross_parent);
        mju_scl3(cross_parent, axis, w[dof]);
        mju_addTo3(linear_w+3*body, cross_parent);
      }
    }
  }

  for (int body = 1; body < m->nbody; ++body) {
    mjtNum com[3];
    mju_mulMatVec3(com, d->xmat+9*body, m->body_ipos+3*body);
    mjtNum angular_com[3];
    mju_cross(angular_com, angular_y+3*body, com);
    mjtNum mass = m->body_mass[body];
    for (int k = 0; k < 3; ++k) {
      force_all[3*body+k] = mass*(linear_y[3*body+k] + angular_com[k]);
    }
    mjtNum inertia[3];
    body_inertia_mul(m, d, body, angular_y+3*body, inertia);
    mjtNum com_force[3];
    mju_cross(com_force, com, force_all+3*body);
    mju_add3(moment_all+3*body, inertia, com_force);
  }

  for (int joint = 0; joint < m->njnt; ++joint) {
    int joint_body = m->jnt_bodyid[joint];
    int free_joint = m->jnt_type[joint] == mjJNT_FREE;
    int channels = free_joint ? 6 : 1;
    for (int channel = 0; channel < channels; ++channel) {
      int dof = m->jnt_dofadr[joint]+channel;
      int rotating = m->jnt_type[joint] == mjJNT_HINGE ||
                     (free_joint && channel >= 3);
      mjtNum axis[3] = {0, 0, 0};
      if (free_joint) {
        if (channel < 3) {
          axis[channel] = 1;
        } else {
          mjtNum local_axis[3] = {0, 0, 0};
          local_axis[channel-3] = 1;
          mju_mulMatVec3(axis, d->xmat+9*joint_body, local_axis);
        }
      } else {
        mju_mulMatVec3(axis, d->xmat+9*joint_body, m->jnt_axis+3*joint);
      }
      mju_zero(delta_angular_all, 12*m->nbody);
      mju_zero(delta_pos, 3*m->nbody);
      for (int body = joint_body; body < m->nbody; ++body) {
        int ancestor = body;
        while (ancestor != joint_body && ancestor > 0) {
          ancestor = m->body_parentid[ancestor];
        }
        if (ancestor != joint_body) continue;
        if (rotating) {
          mjtNum offset[3];
          mju_sub3(offset, d->xpos+3*body, d->xpos+3*joint_body);
          mju_cross(delta_pos+3*body, axis, offset);
        } else {
          mju_copy(delta_pos+3*body, axis, 3);
        }
      }
    for (int body = joint_body; body < m->nbody; ++body) {
      int ancestor = body;
      while (ancestor != joint_body && ancestor > 0) {
        ancestor = m->body_parentid[ancestor];
      }
      if (ancestor != joint_body) continue;
      mjtNum com[3];
      mju_mulMatVec3(com, d->xmat+9*body, m->body_ipos+3*body);
      mjtNum mass = m->body_mass[body];
      mjtNum* force = force_all+3*body;
      mjtNum* moment = moment_all+3*body;
      mjtNum inertia[3];
      body_inertia_mul(m, d, body, angular_y+3*body, inertia);

      mjtNum* delta_angular = delta_angular_all+3*body;
      mjtNum* delta_linear = delta_linear_all+3*body;
      mjtNum* delta_angular_w = delta_angular_w_all+3*body;
      mjtNum* delta_linear_w = delta_linear_w_all+3*body;
      mjtNum delta_com[3] = {0, 0, 0};
      int parent = m->body_parentid[body];
      int body_joint = m->body_jntnum[body] ? m->body_jntadr[body] : -1;
      if (free_root && body == 1) body_joint = -1;
      mjtNum delta_axis[3] = {0, 0, 0};
      if (rotating && body_joint >= 0) {
        mjtNum body_axis[3];
        mju_mulMatVec3(body_axis, d->xmat+9*body, m->jnt_axis+3*body_joint);
        mju_cross(delta_axis, axis, body_axis);
      }
      mjtNum offset[3], delta_offset[3], tmp[3];
      mju_sub3(offset, d->xpos+3*body, d->xpos+3*parent);
      mju_sub3(delta_offset, delta_pos+3*body, delta_pos+3*parent);
      mju_copy(delta_angular, delta_angular_all+3*parent, 3);
      mju_copy(delta_angular_w, delta_angular_w_all+3*parent, 3);
      mju_cross(delta_linear, delta_angular_all+3*parent, offset);
      mju_addTo3(delta_linear, delta_linear_all+3*parent);
      mju_cross(tmp, angular_y+3*parent, delta_offset);
      mju_addTo3(delta_linear, tmp);
      mju_cross(delta_linear_w, delta_angular_w_all+3*parent, offset);
      mju_addTo3(delta_linear_w, delta_linear_w_all+3*parent);
      mju_cross(tmp, angular_w+3*parent, delta_offset);
      mju_addTo3(delta_linear_w, tmp);
      if (body_joint >= 0) {
        int body_dof = m->jnt_dofadr[body_joint];
        if (m->jnt_type[body_joint] == mjJNT_HINGE) {
          mju_scl3(tmp, delta_axis, y[body_dof]);
          mju_addTo3(delta_angular, tmp);
          mju_scl3(tmp, delta_axis, w[body_dof]);
          mju_addTo3(delta_angular_w, tmp);
        } else {
          mju_scl3(tmp, delta_axis, y[body_dof]);
          mju_addTo3(delta_linear, tmp);
          mju_scl3(tmp, delta_axis, w[body_dof]);
          mju_addTo3(delta_linear_w, tmp);
        }
      }
      if (free_root && body == 1 && rotating) {
        mju_cross(delta_angular, axis, angular_y+3*body);
        mju_cross(delta_angular_w, axis, angular_w+3*body);
      }
      if (rotating) mju_cross(delta_com, axis, com);

      mjtNum delta_force[3];
      mjtNum angular_delta[3];
      mju_cross(angular_delta, delta_angular, com);
      mjtNum angular_com_delta[3];
      mju_cross(angular_com_delta, angular_y+3*body, delta_com);
      for (int k = 0; k < 3; ++k) {
        delta_force[k] = mass*(delta_linear[k] + angular_delta[k] + angular_com_delta[k]);
      }
      mjtNum delta_inertia[3] = {0, 0, 0};
      mjtNum inertia_delta[3];
      body_inertia_mul(m, d, body, delta_angular, inertia_delta);
      if (rotating) {
        mjtNum theta_inertia[3];
        mju_cross(theta_inertia, axis, inertia);
        mjtNum theta_angular[3];
        mju_cross(theta_angular, axis, angular_y+3*body);
        mjtNum inertia_theta[3];
        body_inertia_mul(m, d, body, theta_angular, inertia_theta);
        mju_sub3(delta_inertia, theta_inertia, inertia_theta);
      }
      mjtNum delta_moment[3];
      mjtNum com_delta_force[3];
      mjtNum delta_com_force[3];
      mju_cross(com_delta_force, com, delta_force);
      mju_cross(delta_com_force, delta_com, force);
      for (int k = 0; k < 3; ++k) {
        delta_moment[k] = delta_inertia[k] + inertia_delta[k] +
                          com_delta_force[k] + delta_com_force[k];
      }
      for (int k = 0; k < 3; ++k) {
        dbar_qpos[dof] += delta_force[k]*linear_w[3*body+k] + force[k]*delta_linear_w[k] +
                          delta_moment[k]*angular_w[3*body+k] + moment[k]*delta_angular_w[k];
      }
    }
  }
  }
  mj_freeStack(d);
  return mjADJ_OK;
}

int mjd_freeMassPosAdjoint(const mjModel* m, const mjData* d,
                           const mjtNum y[6], const mjtNum w[6], mjtNum dbar_qpos[6]) {
  if (!m || !d || !y || !w || !dbar_qpos) {
    return mjADJ_INVALID_ARGUMENT;
  }
  if (m->nbody != 2 || m->njnt != 1 || m->nv != 6 || m->jnt_type[0] != mjJNT_FREE) {
    return mjADJ_UNSUPPORTED;
  }
  if (ranges_overlap(y, 6*sizeof(mjtNum), dbar_qpos, 6*sizeof(mjtNum)) ||
      ranges_overlap(w, 6*sizeof(mjtNum), dbar_qpos, 6*sizeof(mjtNum))) {
    return mjADJ_INVALID_ARGUMENT;
  }
  const mjtNum* offset = m->body_ipos + 3;
  mjtNum skew[9] = {0, -offset[2], offset[1],
                    offset[2], 0, -offset[0],
                    -offset[1], offset[0], 0};
  mjtNum mass = m->body_mass[1];
  for (int i = 0; i < 3; ++i) {
    dbar_qpos[i] = 0;
  }
  for (int axis = 0; axis < 3; ++axis) {
    mjtNum delta[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    if (axis == 0) {
      delta[5] = -1;
      delta[7] = 1;
    } else if (axis == 1) {
      delta[2] = 1;
      delta[6] = -1;
    } else {
      delta[1] = -1;
      delta[3] = 1;
    }
    mjtNum dB[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        for (int k = 0; k < 3; ++k) {
          for (int l = 0; l < 3; ++l) {
            dB[3*row+col] -= mass*d->xmat[9+3*row+k]*delta[3*k+l]*skew[3*l+col];
          }
        }
      }
    }
    mjtNum gradient = 0;
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        gradient += y[row]*dB[3*row+col]*w[3+col] + y[3+row]*dB[3*col+row]*w[col];
      }
    }
    dbar_qpos[3+axis] = gradient;
  }
  return mjADJ_OK;
}

int mjd_freeBiasPosAdjoint(const mjModel* m, const mjData* d,
                           const mjtNum bar_qfrc_bias[6], mjtNum dbar_qpos[6]) {
  if (!m || !d || !bar_qfrc_bias || !dbar_qpos) {
    return mjADJ_INVALID_ARGUMENT;
  }
  if (!mj_isFreeBody(m, 1) || m->nbody != 2 || m->njnt != 1) {
    return mjADJ_UNSUPPORTED;
  }
  if (ranges_overlap(bar_qfrc_bias, 6*sizeof(mjtNum), dbar_qpos, 6*sizeof(mjtNum))) {
    return mjADJ_INVALID_ARGUMENT;
  }
  const mjtNum* R = d->xmat + 9;
  const mjtNum* offset = m->body_ipos + 3;
  const mjtNum* angular = d->qvel + 3;
  mjtNum omega[3] = {0, 0, 0};
  mjtNum s[3] = {0, 0, 0};
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      omega[row] += R[3*row+col]*angular[col];
      s[row] += R[3*row+col]*offset[col];
    }
  }
  mjtNum A[3];
  mju_cross(A, omega, s);
  mjtNum force[3];
  mjtNum mass = m->body_mass[1];
  mju_cross(force, omega, A);
  for (int i = 0; i < 3; ++i) {
    force[i] = mass*force[i] - (mjDISABLED(mjDSBL_GRAVITY) ? 0 : mass*m->opt.gravity[i]);
  }
  for (int i = 0; i < 3; ++i) {
    dbar_qpos[i] = 0;
  }
  for (int axis = 0; axis < 3; ++axis) {
    mjtNum delta[3] = {0, 0, 0};
    delta[axis] = 1;
    mjtNum cross_delta[3];
    mjtNum omega_delta[3] = {0, 0, 0};
    mjtNum s_delta[3] = {0, 0, 0};
    mju_cross(cross_delta, delta, angular);
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        omega_delta[row] += R[3*row+col]*cross_delta[col];
      }
    }
    mju_cross(cross_delta, delta, offset);
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        s_delta[row] += R[3*row+col]*cross_delta[col];
      }
    }
    mjtNum A_delta[3];
    mjtNum tmp[3];
    mju_cross(A_delta, omega_delta, s);
    mju_cross(tmp, omega, s_delta);
    for (int i = 0; i < 3; ++i) A_delta[i] += tmp[i];
    mjtNum force_delta[3];
    mju_cross(force_delta, omega_delta, A);
    mju_cross(tmp, omega, A_delta);
    for (int i = 0; i < 3; ++i) force_delta[i] = mass*(force_delta[i] + tmp[i]);
    mjtNum delta_mat[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    if (axis == 0) {
      delta_mat[5] = -1;
      delta_mat[7] = 1;
    } else if (axis == 1) {
      delta_mat[2] = 1;
      delta_mat[6] = -1;
    } else {
      delta_mat[1] = -1;
      delta_mat[3] = 1;
    }
    mjtNum rotation_delta[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        for (int k = 0; k < 3; ++k) {
          rotation_delta[3*row+col] += R[3*row+k]*delta_mat[3*k+col];
        }
      }
    }
    mjtNum local_delta[3] = {0, 0, 0};
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        local_delta[row] += rotation_delta[3*col+row]*force[col] + R[3*col+row]*force_delta[col];
      }
    }
    mjtNum torque_delta[3];
    mju_cross(torque_delta, offset, local_delta);
    mjtNum gradient = 0;
    for (int i = 0; i < 3; ++i) {
      gradient += bar_qfrc_bias[i]*force_delta[i] + bar_qfrc_bias[3+i]*torque_delta[i];
    }
    dbar_qpos[3+axis] = gradient;
  }
  return mjADJ_OK;
}

int mjd_fixedSmoothAccPosAdjoint(const mjModel* m, mjData* d,
                                 const mjtNum* bar_qacc_smooth,
                                 mjtNum* y, mjtNum* dbar_qpos) {
  if (!m || !d || (m->nv && (!bar_qacc_smooth || !y || !dbar_qpos))) {
    return mjADJ_INVALID_ARGUMENT;
  }
  int model_status = mjd_checkAdjointModel(m);
  if (model_status != mjADJ_OK) return model_status;
  if ((m->opt.integrator != mjINT_EULER && m->opt.integrator != mjINT_IMPLICITFAST) ||
      d->efm_active) {
    return mjADJ_UNSUPPORTED;
  }
  int tree_kind = scalar_tree_kind(m);
  if (!tree_kind ||
      (tree_kind == 2 && m->nbody == 2 && m->opt.integrator == mjINT_IMPLICITFAST)) {
    return mjADJ_UNSUPPORTED;
  }
  size_t tangent_bytes = (size_t)m->nv * sizeof(mjtNum);
  if (ranges_overlap(bar_qacc_smooth, tangent_bytes, y, tangent_bytes) ||
      ranges_overlap(bar_qacc_smooth, tangent_bytes, dbar_qpos, tangent_bytes) ||
      ranges_overlap(y, tangent_bytes, dbar_qpos, tangent_bytes)) {
    return mjADJ_INVALID_ARGUMENT;
  }
  memcpy(y, bar_qacc_smooth, tangent_bytes);
  mj_solveLD(y, d->qLD, d->qLDiagInv, m->nv, 1,
             m->M_rownnz, m->M_rowadr, m->M_colind, NULL);
  mj_markStack(d);
  mjtNum* residual = mjSTACKALLOC(d, 6*m->nv, mjtNum);
  mjtNum* mass_residual = residual;
  mjtNum* bias_residual = residual+m->nv;
  mjtNum* passive_residual = residual+2*m->nv;
  mjtNum* passive_qvel = residual+3*m->nv;
  mjtNum* actuator_residual = residual+4*m->nv;
  mjtNum* actuator_qvel = residual+5*m->nv;
  int status = mjd_massPosAdjoint(m, d, y, d->qacc_smooth, mass_residual);
  if (status == mjADJ_OK) {
    status = mjd_rneBiasPosAdjoint(m, d, y, bias_residual);
  }
  if (status == mjADJ_OK) {
    status = mjd_passiveStateAdjoint(m, d, y, passive_residual, passive_qvel);
  }
  if (status == mjADJ_OK) {
    status = mjd_actuatorStateAdjoint(m, d, y, actuator_residual, actuator_qvel);
  }
  if (status == mjADJ_OK) {
    for (int i = 0; i < m->nv; ++i) {
      dbar_qpos[i] = passive_residual[i] + actuator_residual[i] -
                     bias_residual[i] - mass_residual[i];
    }
  }
  mj_freeStack(d);
  return status;
}

int mjd_freeSmoothAccPosAdjoint(const mjModel* m, const mjData* d,
                                const mjtNum bar_qacc_smooth[6],
                                mjtNum y[6], mjtNum dbar_qpos[6]) {
  if (!m || !d || !bar_qacc_smooth || !y || !dbar_qpos) {
    return mjADJ_INVALID_ARGUMENT;
  }
  if (!mj_isFreeBody(m, 1) || m->nbody != 2 || m->njnt != 1 ||
      m->opt.integrator != mjINT_EULER || d->efm_active) {
    return mjADJ_UNSUPPORTED;
  }
  if (ranges_overlap(bar_qacc_smooth, 6*sizeof(mjtNum), y, 6*sizeof(mjtNum)) ||
      ranges_overlap(bar_qacc_smooth, 6*sizeof(mjtNum), dbar_qpos, 6*sizeof(mjtNum)) ||
      ranges_overlap(y, 6*sizeof(mjtNum), dbar_qpos, 6*sizeof(mjtNum))) {
    return mjADJ_INVALID_ARGUMENT;
  }
  memcpy(y, bar_qacc_smooth, 6*sizeof(mjtNum));
  mj_solveLD(y, d->qLD, d->qLDiagInv, 6, 1,
             m->M_rownnz, m->M_rowadr, m->M_colind, NULL);
  mjtNum mass_residual[6];
  int status = mjd_freeMassPosAdjoint(m, d, y, d->qacc_smooth, mass_residual);
  if (status != mjADJ_OK) return status;
  mjtNum bias_residual[6];
  status = mjd_freeBiasPosAdjoint(m, d, y, bias_residual);
  if (status != mjADJ_OK) return status;
  mjtNum passive_residual[6];
  mjtNum passive_qvel[6];
  status = mjd_passiveStateAdjoint(m, d, y, passive_residual, passive_qvel);
  if (status != mjADJ_OK) return status;
  for (int i = 0; i < 6; ++i) {
    dbar_qpos[i] = passive_residual[i] - bias_residual[i] - mass_residual[i];
  }
  return mjADJ_OK;
}
