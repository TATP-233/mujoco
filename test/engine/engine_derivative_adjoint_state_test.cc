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

#include <cmath>
#include <random>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mujoco/mujoco.h>
#include "test/fixture.h"

namespace mujoco {
namespace {

using ::testing::NotNull;

class AdjointStateTest : public MujocoTest {};

TEST_F(AdjointStateTest, AffineBiasGearAndJvpVjp) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option solver="Newton"><flag island="disable"/></option>
    <worldbody>
      <body><joint name="j0" type="slide"/><geom size=".1" mass="1"/></body>
      <body pos="1 0 0"><joint name="j1" type="hinge"/><geom size=".1" mass="1"/></body>
    </worldbody>
    <actuator>
      <position joint="j0" kp="2" kv=".3" gear="3"/>
      <position joint="j0" kp=".5" kv=".1" gear="-.5"/>
      <velocity joint="j1" kv="2" gear="-2"/>
    </actuator>
  </mujoco>)";
  char error[1024] = "";
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  data->qpos[0] = .1;
  data->qvel[0] = -.2;
  data->qvel[1] = .4;
  mj_forward(model.get(), data.get());
  std::vector<mjtNum> unit_bar = {1, 1};
  std::vector<mjtNum> grad_qpos(2, 0);
  std::vector<mjtNum> grad_qvel(2, 0);
  ASSERT_EQ(mjd_actuatorStateAdjoint(model.get(), data.get(), unit_bar.data(),
                                     grad_qpos.data(), grad_qvel.data()), mjADJ_OK);
  EXPECT_NEAR(grad_qpos[0], -18.125, 1e-12);
  EXPECT_NEAR(grad_qpos[1], 0, 1e-12);
  EXPECT_NEAR(grad_qvel[0], -2.725, 1e-12);
  EXPECT_NEAR(grad_qvel[1], -8, 1e-12);

  std::mt19937_64 rng(11);
  std::normal_distribution<double> normal;
  for (int sample = 0; sample < 4; ++sample) {
    std::vector<mjtNum> bar = {normal(rng), normal(rng)};
    std::vector<mjtNum> dir_qpos = {normal(rng), normal(rng)};
    std::vector<mjtNum> dir_qvel = {normal(rng), normal(rng)};
    ASSERT_EQ(mjd_actuatorStateAdjoint(model.get(), data.get(), bar.data(),
                                       grad_qpos.data(), grad_qvel.data()), mjADJ_OK);
    MjDataPtr plus = MakeData(model);
    MjDataPtr minus = MakeData(model);
    mj_copyData(plus.get(), model.get(), data.get());
    mj_copyData(minus.get(), model.get(), data.get());
    constexpr mjtNum eps = 1e-6;
    for (int i = 0; i < 2; ++i) {
      plus->qpos[i] += eps*dir_qpos[i];
      minus->qpos[i] -= eps*dir_qpos[i];
      plus->qvel[i] += eps*dir_qvel[i];
      minus->qvel[i] -= eps*dir_qvel[i];
    }
    mj_forward(model.get(), plus.get());
    mj_forward(model.get(), minus.get());
    mjtNum jvp_vjp = 0;
    mjtNum vjp_jvp = 0;
    for (int i = 0; i < 2; ++i) {
      jvp_vjp += bar[i]*(plus->qfrc_actuator[i] - minus->qfrc_actuator[i])/(2*eps);
      vjp_jvp += grad_qpos[i]*dir_qpos[i] + grad_qvel[i]*dir_qvel[i];
    }
    EXPECT_NEAR(vjp_jvp, jvp_vjp, 2e-8);
  }
}

TEST_F(AdjointStateTest, JointClampMasksQposAndQvel) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option solver="Newton"><flag island="disable"/></option>
    <worldbody><body>
      <joint name="j" type="slide" actuatorfrclimited="true" actuatorfrcrange="-.5 .5"/>
      <geom size=".1" mass="1"/>
    </body></worldbody>
    <actuator><position joint="j" kp="10" kv="1" gear="2"/></actuator>
  </mujoco>)";
  char error[1024] = "";
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  const mjtNum positions[] = {0, -.5, .5};
  const mjtNum expected[] = {-40, 0, 0};
  for (int sample = 0; sample < 3; ++sample) {
    MjDataPtr data = MakeData(model);
    data->qpos[0] = positions[sample];
    data->ctrl[0] = .02;
    mj_forward(model.get(), data.get());
    mjtNum bar = 1;
    mjtNum grad_qpos = 1;
    mjtNum grad_qvel = 1;
    ASSERT_EQ(mjd_actuatorStateAdjoint(model.get(), data.get(), &bar, &grad_qpos, &grad_qvel),
              mjADJ_OK);
    EXPECT_NEAR(grad_qpos, expected[sample], 1e-12);
    EXPECT_NEAR(grad_qvel, sample == 0 ? -4 : 0, 1e-12);
    mjtNum losses[2];
    for (int side = 0; side < 2; ++side) {
      MjDataPtr perturbed = MakeData(model);
      mj_copyData(perturbed.get(), model.get(), data.get());
      perturbed->qpos[0] += side == 0 ? 1e-6 : -1e-6;
      mj_forward(model.get(), perturbed.get());
      losses[side] = perturbed->qfrc_actuator[0];
    }
    EXPECT_NEAR(grad_qpos, (losses[0] - losses[1])/2e-6, 1e-7);
    for (int side = 0; side < 2; ++side) {
      MjDataPtr perturbed = MakeData(model);
      mj_copyData(perturbed.get(), model.get(), data.get());
      perturbed->qvel[0] += side == 0 ? 1e-6 : -1e-6;
      mj_forward(model.get(), perturbed.get());
      losses[side] = perturbed->qfrc_actuator[0];
    }
    EXPECT_NEAR(grad_qvel, (losses[0] - losses[1])/2e-6, 1e-7);
  }
}

TEST_F(AdjointStateTest, RejectsAliasedTangentBuffers) {
  static constexpr char xml[] = R"(
  <mujoco><option solver="Newton"><flag island="disable"/></option>
    <worldbody><body><joint name="j" type="slide"/><geom size=".1"/></body></worldbody>
    <actuator><position joint="j" kp="1"/></actuator>
  </mujoco>)";
  char error[1024] = "";
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());
  mjtNum value = 0;
  EXPECT_EQ(mjd_actuatorStateAdjoint(model.get(), data.get(), &value, &value, &value),
            mjADJ_INVALID_ARGUMENT);
}

}  // namespace
}  // namespace mujoco
