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

#include "src/engine/engine_derivative_adjoint.h"

#include <random>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mujoco/mujoco.h>
#include "src/engine/engine_core_smooth.h"
#include "src/engine/engine_derivative.h"
#include "src/engine/engine_util_blas.h"
#include "test/fixture.h"

namespace mujoco {
namespace {

using ::testing::NotNull;

class AdjointImplicitTest : public MujocoTest {};

TEST_F(AdjointImplicitTest, TransposeSolveMatchesDenseFactorization) {
  static constexpr char xml[] = R"(
  <mujoco><option solver="Newton" integrator="implicitfast" timestep=".01">
    <flag island="disable"/></option>
    <worldbody><body><joint name="j0" type="hinge" damping=".3"/>
      <geom type="capsule" size=".03" fromto="0 0 0 0 0 .4" mass="1"/>
      <body pos="0 0 .4"><joint name="j1" type="hinge" damping=".2"/>
        <geom type="capsule" size=".02" fromto="0 0 0 .3 0 0" mass=".5"/>
        <body pos=".3 0 0"><joint name="j2" type="slide" damping=".1"/>
          <geom size=".04" mass=".2"/>
        </body>
      </body>
    </body></worldbody>
    <actuator><position joint="j1" kp="2" kv=".4"/></actuator>
  </mujoco>)";
  char error[1024] = "";
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  data->qpos[0] = .3;
  data->qpos[1] = -.4;
  data->qpos[2] = .2;
  data->qvel[0] = .5;
  data->qvel[1] = -.7;
  data->qvel[2] = .9;
  mj_forward(model.get(), data.get());
  mju_zero(data->qDeriv, model->nD);
  mjd_smooth_vel(model.get(), data.get(), 0);
  std::vector<mjtNum> qH(model->nC);
  for (int i = 0; i < model->nC; ++i) {
    qH[i] = data->M[i] - model->opt.timestep*data->qDeriv[model->mapD2M[i]];
  }
  std::vector<mjtNum> dense(model->nv*model->nv, 0);
  for (int row = 0; row < model->nv; ++row) {
    for (int k = 0; k < model->M_rownnz[row]; ++k) {
      int adr = model->M_rowadr[row] + k;
      int col = model->M_colind[adr];
      dense[model->nv*row + col] = qH[adr];
      dense[model->nv*col + row] = qH[adr];
    }
  }
  std::vector<mjtNum> dense_factor = dense;
  ASSERT_NE(mju_cholFactor(dense_factor.data(), model->nv, 0), 0);
  mju_copy(data->qH, qH.data(), model->nC);
  mj_factorI(data->qH, data->qHDiagInv, model->nv,
             model->M_rownnz, model->M_rowadr, model->M_colind, NULL);

  std::mt19937_64 rng(31);
  std::normal_distribution<double> normal;
  for (int sample = 0; sample < 4; ++sample) {
    std::vector<mjtNum> bar(model->nv);
    for (mjtNum& value : bar) value = normal(rng);
    std::vector<mjtNum> actual = bar;
    ASSERT_EQ(mjd_implicitFastSolveAdjoint(model.get(), data.get(), actual.data(), actual.data()),
              mjADJ_OK);
    std::vector<mjtNum> expected(model->nv);
    mju_cholSolve(expected.data(), dense_factor.data(), bar.data(), model->nv);
    for (int i = 0; i < model->nv; ++i) {
      EXPECT_NEAR(actual[i], expected[i], 2e-10) << sample << ":" << i;
    }
    std::vector<mjtNum> separate(model->nv);
    ASSERT_EQ(mjd_implicitFastSolveAdjoint(model.get(), data.get(), bar.data(), separate.data()),
              mjADJ_OK);
    EXPECT_EQ(separate, actual);
  }
}

TEST_F(AdjointImplicitTest, FreeBodyAndWrongIntegratorRejected) {
  static constexpr char xml[] = R"(
  <mujoco><option solver="Newton" integrator="implicitfast"><flag island="disable"/></option>
    <worldbody><body><freejoint/><geom size=".1"/></body></worldbody>
  </mujoco>)";
  char error[1024] = "";
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  std::vector<mjtNum> bar(6, 1);
  std::vector<mjtNum> actual(6, 0);
  EXPECT_EQ(mjd_implicitFastSolveAdjoint(model.get(), data.get(), bar.data(), actual.data()),
            mjADJ_UNSUPPORTED);
  model->opt.integrator = mjINT_EULER;
  EXPECT_EQ(mjd_implicitFastSolveAdjoint(model.get(), data.get(), bar.data(), actual.data()),
            mjADJ_UNSUPPORTED);
}

}  // namespace
}  // namespace mujoco
