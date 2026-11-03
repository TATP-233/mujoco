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

#include <cstring>
#include <random>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mujoco/mujoco.h>
#include "test/fixture.h"

namespace mujoco {
namespace {

using ::testing::NotNull;

class AdjointSmoothTest : public MujocoTest {};

TEST_F(AdjointSmoothTest, CombinedForcesJvpVjp) {
  static constexpr char xml[] = R"(
  <mujoco><option solver="Newton" integrator="implicitfast" timestep=".01">
    <flag island="disable"/></option>
    <worldbody><body><joint name="j0" type="hinge" damping=".3"/>
      <geom type="capsule" size=".03" fromto="0 0 0 0 0 .4" mass="1"/>
      <body pos="0 0 .4"><joint name="j1" type="hinge" damping=".2"/>
        <geom type="capsule" size=".02" fromto="0 0 0 .3 0 0" mass=".5"/>
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
  data->qvel[0] = .5;
  data->qvel[1] = -.7;
  mj_forward(model.get(), data.get());
  std::vector<mjtNum> persistent(data->qDeriv, data->qDeriv + model->nD);
  std::vector<mjtNum> scratch(model->nD);
  std::mt19937_64 rng(37);
  std::normal_distribution<double> normal;
  for (int sample = 0; sample < 4; ++sample) {
    std::vector<mjtNum> bar = {normal(rng), normal(rng)};
    std::vector<mjtNum> direction = {normal(rng), normal(rng)};
    std::vector<mjtNum> gradient(2, 0);
    ASSERT_EQ(mjd_smoothVelAdjoint(model.get(), data.get(), bar.data(), scratch.data(),
                                   gradient.data()), mjADJ_OK);
    MjDataPtr plus = MakeData(model);
    MjDataPtr minus = MakeData(model);
    mj_copyData(plus.get(), model.get(), data.get());
    mj_copyData(minus.get(), model.get(), data.get());
    constexpr mjtNum eps = 1e-6;
    for (int i = 0; i < 2; ++i) {
      plus->qvel[i] += eps*direction[i];
      minus->qvel[i] -= eps*direction[i];
    }
    mj_forward(model.get(), plus.get());
    mj_forward(model.get(), minus.get());
    mjtNum jvp_vjp = 0;
    mjtNum vjp_jvp = 0;
    for (int i = 0; i < 2; ++i) {
      jvp_vjp += bar[i]*(plus->qfrc_smooth[i] - minus->qfrc_smooth[i])/(2*eps);
      vjp_jvp += gradient[i]*direction[i];
    }
    EXPECT_NEAR(vjp_jvp, jvp_vjp, 2e-8) << sample;
    EXPECT_EQ(std::memcmp(persistent.data(), data->qDeriv, model->nD*sizeof(mjtNum)), 0);
  }
}

TEST_F(AdjointSmoothTest, EulerSmoothAccelerationJvpVjp) {
  static constexpr char xml[] = R"(
  <mujoco><option solver="Newton" integrator="Euler" timestep=".01">
    <flag island="disable"/></option>
    <worldbody><body><joint name="j0" type="hinge" damping=".3"/>
      <geom type="capsule" size=".03" fromto="0 0 0 0 0 .4" mass="1"/>
      <body pos="0 0 .4"><joint name="j1" type="hinge" damping=".2"/>
        <geom type="capsule" size=".02" fromto="0 0 0 .3 0 0" mass=".5"/>
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
  data->qvel[0] = .5;
  data->qvel[1] = -.7;
  mj_forward(model.get(), data.get());
  std::vector<mjtNum> scratch(model->nD);
  std::vector<mjtNum> y(model->nv);
  std::mt19937_64 rng(41);
  std::normal_distribution<double> normal;
  for (int sample = 0; sample < 4; ++sample) {
    std::vector<mjtNum> bar = {normal(rng), normal(rng)};
    std::vector<mjtNum> direction = {normal(rng), normal(rng)};
    std::vector<mjtNum> gradient(2, 0);
    ASSERT_EQ(mjd_smoothAccVelAdjoint(model.get(), data.get(), bar.data(), scratch.data(),
                                      y.data(), gradient.data()), mjADJ_OK);
    MjDataPtr plus = MakeData(model);
    MjDataPtr minus = MakeData(model);
    mj_copyData(plus.get(), model.get(), data.get());
    mj_copyData(minus.get(), model.get(), data.get());
    constexpr mjtNum eps = 1e-6;
    for (int i = 0; i < 2; ++i) {
      plus->qvel[i] += eps*direction[i];
      minus->qvel[i] -= eps*direction[i];
    }
    mj_forward(model.get(), plus.get());
    mj_forward(model.get(), minus.get());
    mjtNum jvp_vjp = 0;
    mjtNum vjp_jvp = 0;
    for (int i = 0; i < 2; ++i) {
      jvp_vjp += bar[i]*(plus->qacc_smooth[i] - minus->qacc_smooth[i])/(2*eps);
      vjp_jvp += gradient[i]*direction[i];
    }
    mjtNum scale = jvp_vjp < 0 ? -jvp_vjp : jvp_vjp;
    mjtNum error = vjp_jvp - jvp_vjp;
    error = error < 0 ? -error : error;
    EXPECT_LT(error/(scale > 1 ? scale : 1), 1e-9) << sample;
  }
  model->opt.integrator = mjINT_IMPLICITFAST;
  std::vector<mjtNum> bar = {1, 1};
  std::vector<mjtNum> gradient(2, 0);
  EXPECT_EQ(mjd_smoothAccVelAdjoint(model.get(), data.get(), bar.data(), scratch.data(),
                                    y.data(), gradient.data()), mjADJ_UNSUPPORTED);
}

TEST_F(AdjointSmoothTest, EulerSmoothAccelerationCtrlAndClamp) {
  static constexpr char xml[] = R"(
  <mujoco><option solver="Newton" integrator="Euler" timestep=".01">
    <flag island="disable"/></option>
    <worldbody><body><joint name="j0" type="hinge" damping=".3"/>
      <geom type="capsule" size=".03" fromto="0 0 0 0 0 .4" mass="1"/>
      <body pos="0 0 .4"><joint name="j1" type="hinge" damping=".2"/>
        <geom type="capsule" size=".02" fromto="0 0 0 .3 0 0" mass=".5"/>
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
  data->qvel[0] = .5;
  data->qvel[1] = -.7;
  mj_forward(model.get(), data.get());
  std::vector<mjtNum> y(model->nv);
  std::mt19937_64 rng(43);
  std::normal_distribution<double> normal;
  for (int sample = 0; sample < 4; ++sample) {
    std::vector<mjtNum> bar = {normal(rng), normal(rng)};
    mjtNum direction = normal(rng);
    mjtNum gradient = 0;
    ASSERT_EQ(mjd_smoothAccCtrlAdjoint(model.get(), data.get(), bar.data(), y.data(), &gradient),
              mjADJ_OK);
    MjDataPtr plus = MakeData(model);
    MjDataPtr minus = MakeData(model);
    mj_copyData(plus.get(), model.get(), data.get());
    mj_copyData(minus.get(), model.get(), data.get());
    constexpr mjtNum eps = 1e-6;
    plus->ctrl[0] += eps*direction;
    minus->ctrl[0] -= eps*direction;
    mj_forward(model.get(), plus.get());
    mj_forward(model.get(), minus.get());
    mjtNum jvp_vjp = 0;
    for (int i = 0; i < 2; ++i) {
      jvp_vjp += bar[i]*(plus->qacc_smooth[i] - minus->qacc_smooth[i])/(2*eps);
    }
    mjtNum scale = jvp_vjp < 0 ? -jvp_vjp : jvp_vjp;
    mjtNum error = gradient*direction - jvp_vjp;
    error = error < 0 ? -error : error;
    EXPECT_LT(error/(scale > 1 ? scale : 1), 1e-9) << sample;
  }
  model->actuator_ctrllimited[0] = 1;
  model->actuator_ctrlrange[0] = -.05;
  model->actuator_ctrlrange[1] = .05;
  data->ctrl[0] = .1;
  mj_forward(model.get(), data.get());
  std::vector<mjtNum> bar = {1, 1};
  mjtNum gradient = 1;
  ASSERT_EQ(mjd_smoothAccCtrlAdjoint(model.get(), data.get(), bar.data(), y.data(), &gradient),
            mjADJ_OK);
  EXPECT_EQ(gradient, 0);
}

}  // namespace
}  // namespace mujoco
