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
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mujoco/mujoco.h>
#include "test/engine/engine_derivative_adjoint_test_util.h"
#include "test/fixture.h"

namespace mujoco {
namespace {

using ::testing::NotNull;

class AdjointCtrlTest : public MujocoTest {};

double QfrcLoss(const mjModel* m, const mjData* d, mjtNum bar) {
  return bar*d->qfrc_actuator[0];
}

TEST_F(AdjointCtrlTest, ClampMatrixMatchesStockForward) {
  struct ClampCase {
    const char* name;
    const char* joint;
    const char* actuator;
  };
  const ClampCase clamp_cases[] = {
      {"ctrlrange", "", "ctrllimited=\"true\" ctrlrange=\"-.05 .05\""},
      {"forcerange", "", "forcelimited=\"true\" forcerange=\"-.5 .5\""},
      {"actuatorfrcrange", "actuatorfrclimited=\"true\" actuatorfrcrange=\"-.5 .5\"", ""},
  };
  struct Zone { const char* name; mjtNum ctrl; bool clamped; };
  const Zone zones[] = {{"lower", -1, true}, {"interior", 0, false}, {"upper", 1, true}};
  constexpr mjtNum bar = .7;
  int completed = 0;
  for (const ClampCase& clamp : clamp_cases) {
    std::string xml = std::string(R"(
    <mujoco>
      <option timestep=".01" gravity="0 0 0" solver="Newton">
        <flag island="disable" warmstart="disable"/>
      </option>
      <worldbody><body><joint name="j" type="slide" axis="1 0 0" )") +
      clamp.joint + R"(/>
        <geom type="sphere" size=".05" mass="1"/>
      </body></worldbody>
      <actuator><motor joint="j" gear="10" )" + clamp.actuator + R"(/></actuator>
    </mujoco>)";
    char error[1024] = "";
    MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
    ASSERT_THAT(model.get(), NotNull()) << error;
    for (const Zone& zone : zones) {
      SCOPED_TRACE(std::string(clamp.name) + "/" + zone.name);
      MjDataPtr data = MakeData(model);
      data->ctrl[0] = zone.ctrl;
      mj_forward(model.get(), data.get());
      mjtNum gradient = -1;
      ASSERT_EQ(mjd_actuatorCtrlAdjoint(model.get(), data.get(), &bar, &gradient), mjADJ_OK);
      mjtNum losses[2];
      for (int side = 0; side < 2; ++side) {
        MjDataPtr perturbed = MakeData(model);
        mj_copyData(perturbed.get(), model.get(), data.get());
        perturbed->ctrl[0] += side == 0 ? 1e-6 : -1e-6;
        mj_forward(model.get(), perturbed.get());
        losses[side] = QfrcLoss(model.get(), perturbed.get(), bar);
      }
      mjtNum fd = (losses[0] - losses[1])/2e-6;
      EXPECT_NEAR(gradient, fd, 1e-8);
      EXPECT_NEAR(gradient, zone.clamped ? 0 : 7, 1e-12);
      if (zone.clamped && std::string(clamp.name) == "actuatorfrcrange") {
        adjoint_test::Comparison rejected =
            adjoint_test::CompareVectors({7}, {static_cast<double>(fd)});
        EXPECT_GT(rejected.max_absolute, 1) << "unmasked joint clamp must be rejected";
      }
      ++completed;
    }
  }
  EXPECT_EQ(completed, 9);
}

TEST_F(AdjointCtrlTest, MultiActuatorJvpVjpDotProduct) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option solver="Newton"><flag island="disable"/></option>
    <worldbody>
      <body><joint name="j0" type="slide"/><geom size=".1" mass="1"/></body>
      <body pos="1 0 0"><joint name="j1" type="slide"/><geom size=".1" mass="1"/></body>
    </worldbody>
    <actuator>
      <motor joint="j0" gear="2"/>
      <motor joint="j1" gear="-3"/>
      <motor joint="j0" gear=".5"/>
    </actuator>
  </mujoco>)";
  char error[1024] = "";
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  data->ctrl[0] = .2;
  data->ctrl[1] = -.4;
  data->ctrl[2] = .1;
  mj_forward(model.get(), data.get());
  std::mt19937_64 rng(7);
  std::normal_distribution<double> normal;
  for (int sample = 0; sample < 4; ++sample) {
    std::vector<mjtNum> bar = {normal(rng), normal(rng)};
    std::vector<mjtNum> direction = {normal(rng), normal(rng), normal(rng)};
    std::vector<mjtNum> gradient(3, 0);
    ASSERT_EQ(mjd_actuatorCtrlAdjoint(model.get(), data.get(), bar.data(), gradient.data()),
              mjADJ_OK);
    MjDataPtr plus = MakeData(model);
    MjDataPtr minus = MakeData(model);
    mj_copyData(plus.get(), model.get(), data.get());
    mj_copyData(minus.get(), model.get(), data.get());
    constexpr mjtNum eps = 1e-6;
    for (int i = 0; i < 3; ++i) {
      plus->ctrl[i] += eps*direction[i];
      minus->ctrl[i] -= eps*direction[i];
    }
    mj_forward(model.get(), plus.get());
    mj_forward(model.get(), minus.get());
    mjtNum jvp_vjp = 0;
    mjtNum vjp_jvp = 0;
    for (int i = 0; i < 2; ++i) {
      jvp_vjp += bar[i]*(plus->qfrc_actuator[i] - minus->qfrc_actuator[i])/(2*eps);
    }
    for (int i = 0; i < 3; ++i) {
      vjp_jvp += gradient[i]*direction[i];
    }
    EXPECT_NEAR(vjp_jvp, jvp_vjp, 1e-8);
    EXPECT_GT(std::abs(vjp_jvp + .001 - jvp_vjp), 1e-4);
  }
}

TEST_F(AdjointCtrlTest, CtrlUpperBoundaryUsesRightDerivative) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option solver="Newton"><flag island="disable"/></option>
    <worldbody><body><joint name="j" type="slide"/><geom size=".1" mass="1"/></body></worldbody>
    <actuator><motor joint="j" gear="10" ctrllimited="true" ctrlrange="-.05 .05"/></actuator>
  </mujoco>)";
  char error[1024] = "";
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  data->ctrl[0] = .05;
  mj_forward(model.get(), data.get());
  mjtNum bar = 1;
  mjtNum gradient = -1;
  ASSERT_EQ(mjd_actuatorCtrlAdjoint(model.get(), data.get(), &bar, &gradient), mjADJ_OK);
  EXPECT_EQ(gradient, 0);
  MjDataPtr left = MakeData(model);
  MjDataPtr right = MakeData(model);
  mj_copyData(left.get(), model.get(), data.get());
  mj_copyData(right.get(), model.get(), data.get());
  left->ctrl[0] -= 1e-6;
  right->ctrl[0] += 1e-6;
  mj_forward(model.get(), left.get());
  mj_forward(model.get(), right.get());
  mjtNum base = QfrcLoss(model.get(), data.get(), bar);
  EXPECT_NEAR((base - QfrcLoss(model.get(), left.get(), bar))/1e-6, 10, 1e-8);
  EXPECT_NEAR((QfrcLoss(model.get(), right.get(), bar) - base)/1e-6, 0, 1e-12);
  EXPECT_NEAR((QfrcLoss(model.get(), right.get(), bar) -
               QfrcLoss(model.get(), left.get(), bar))/2e-6, 5, 1e-8);
}

TEST_F(AdjointCtrlTest, PositionActuatorJointClampRegression) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep=".01" gravity="0 0 0" solver="Newton"><flag island="disable"/></option>
    <worldbody>
      <body><joint name="j0" type="slide" actuatorfrclimited="true" actuatorfrcrange="-.5 .5"/>
        <geom size=".05" mass="1"/></body>
      <body pos="1 0 0"><joint name="j1" type="slide" actuatorfrclimited="true" actuatorfrcrange="-.5 .5"/>
        <geom size=".05" mass="1"/></body>
      <body pos="2 0 0"><joint name="j2" type="slide" actuatorfrclimited="true" actuatorfrcrange="-.5 .5"/>
        <geom size=".05" mass="1"/></body>
    </worldbody>
    <actuator>
      <position joint="j0" kp="10"/>
      <position joint="j1" kp="10"/>
      <position joint="j2" kp="10"/>
    </actuator>
  </mujoco>)";
  char error[1024] = "";
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  data->ctrl[0] = .02;
  data->ctrl[1] = .5;
  data->ctrl[2] = -.4;
  mj_forward(model.get(), data.get());
  EXPECT_NEAR(data->qfrc_actuator[0], .2, 1e-12);
  EXPECT_NEAR(data->qfrc_actuator[1], .5, 1e-12);
  EXPECT_NEAR(data->qfrc_actuator[2], -.5, 1e-12);
  std::vector<mjtNum> bar = {1, 1, 1};
  std::vector<mjtNum> gradient(3, -1);
  ASSERT_EQ(mjd_actuatorCtrlAdjoint(model.get(), data.get(), bar.data(), gradient.data()), mjADJ_OK);
  EXPECT_EQ(gradient, std::vector<mjtNum>({10, 0, 0}));
  for (int column = 0; column < 3; ++column) {
    mjtNum losses[2];
    for (int side = 0; side < 2; ++side) {
      MjDataPtr perturbed = MakeData(model);
      mj_copyData(perturbed.get(), model.get(), data.get());
      perturbed->ctrl[column] += side == 0 ? 1e-6 : -1e-6;
      mj_forward(model.get(), perturbed.get());
      losses[side] = perturbed->qfrc_actuator[0] + perturbed->qfrc_actuator[1] +
                     perturbed->qfrc_actuator[2];
    }
    EXPECT_NEAR(gradient[column], (losses[0] - losses[1])/2e-6, 1e-8) << column;
  }
  adjoint_test::Comparison rejected =
      adjoint_test::CompareVectors({10, 10, 10}, {10, 0, 0});
  EXPECT_GT(rejected.max_absolute, 1) << "unmasked affine-bias clamp must be rejected";
}

}  // namespace
}  // namespace mujoco
