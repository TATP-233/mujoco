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

#include "test/engine/engine_derivative_adjoint_test_util.h"

#include <cmath>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mujoco/mujoco.h>
#include "src/engine/engine_util_blas.h"
#include "test/fixture.h"

namespace mujoco {
namespace {

using ::testing::NotNull;
using adjoint_test::CompareVectors;
using adjoint_test::InputBlock;
using adjoint_test::StepLoss;
using adjoint_test::StockStepCentralTangent;

class AdjointValidationTest : public MujocoTest {};

TEST_F(AdjointValidationTest, StockForwardNoiseFloorAllColumns) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep=".002" solver="Newton" iterations="10" integrator="Euler">
      <flag island="disable" warmstart="disable"/>
    </option>
    <worldbody>
      <geom type="plane" size="2 2 .1" friction=".8 .005 .0001"/>
      <body pos="0 0 .0995">
        <joint name="x" type="slide" axis="1 0 0"/>
        <joint name="z" type="slide" axis="0 0 1"/>
        <geom type="sphere" size=".1" mass="1" friction=".8 .005 .0001"/>
      </body>
    </worldbody>
    <actuator><motor joint="x" gear="2"/></actuator>
  </mujoco>)";
  char error[1024] = "";
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  data->qvel[0] = .5;
  data->ctrl[0] = .1;
  const std::vector<mjtNum> bar_qpos = {.3, -.2};
  const std::vector<mjtNum> bar_qvel = {.4, .7};
  const mjtNum epsilons[] = {1e-4, 1e-5, 1e-6};
  struct Column { InputBlock block; int index; const char* name; };
  const Column columns[] = {
      {InputBlock::kQPos, 0, "qpos_x"}, {InputBlock::kQPos, 1, "qpos_z"},
      {InputBlock::kQVel, 0, "qvel_x"}, {InputBlock::kQVel, 1, "qvel_z"},
      {InputBlock::kCtrl, 0, "ctrl_x"},
  };
  int usable = 0;
  for (const Column& column : columns) {
    SCOPED_TRACE(column.name);
    std::vector<double> losses;
    for (mjtNum eps : epsilons) {
      double value = StepLoss(StockStepCentralTangent(model.get(), data.get(),
                                                       column.block, column.index, eps),
                              bar_qpos, bar_qvel);
      ASSERT_TRUE(std::isfinite(value));
      losses.push_back(value);
      ++usable;
    }
    double signature = std::max(std::abs(losses[0]),
                                std::max(std::abs(losses[1]), std::abs(losses[2])));
    EXPECT_GT(signature, 1e-7);
    double convergence = std::abs(losses[1] - losses[2]) /
                         std::max(1e-10, std::abs(losses[2]));
    RecordProperty("medium_vs_fine_relative", convergence);
    RecordProperty("fine_loss", losses[2]);
    EXPECT_LT(convergence, .05);
  }
  EXPECT_EQ(usable, 15);
}

TEST_F(AdjointValidationTest, PerturbedGradientMustFail) {
  const std::vector<double> expected = {.2, -1, 3, .01};
  adjoint_test::Comparison matched = CompareVectors(expected, expected);
  ASSERT_EQ(matched.nonfinite, 0);
  ASSERT_EQ(matched.max_absolute, 0);
  std::vector<double> corrupted = expected;
  corrupted[2] += .1 * std::max(1.0, std::abs(corrupted[2]));
  adjoint_test::Comparison rejected = CompareVectors(corrupted, expected);
  EXPECT_GT(rejected.max_absolute, .1);
  EXPECT_GT(rejected.max_relative, .05);
  EXPECT_LT(rejected.cosine, 1);
}

TEST_F(AdjointValidationTest, PyramidalFirstEdgeSatisfiedWitness) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep=".002" cone="pyramidal" solver="Newton" iterations="100"
            ls_iterations="50" tolerance="0">
      <flag warmstart="disable" island="disable"/>
    </option>
    <worldbody>
      <geom type="plane" size="1 1 .1" friction=".8 .005 .0001"/>
      <body>
        <joint type="slide" axis="1 0 0"/>
        <joint type="slide" axis="0 1 0"/>
        <joint type="slide" axis="0 0 1"/>
        <joint type="hinge" axis="1 0 0"/>
        <joint type="hinge" axis="0 1 0"/>
        <joint type="hinge" axis="0 0 1"/>
        <geom type="box" size=".1 .1 .1" mass="1" friction=".8 .005 .0001"/>
      </body>
    </worldbody>
  </mujoco>)";
  char error[1024] = "";
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  for (double velocity : {.5, -.5}) {
    SCOPED_TRACE(velocity);
    MjDataPtr data = MakeData(model);
    data->qpos[2] = .0995;
    data->qpos[3] = .01;
    data->qpos[4] = -.01;
    data->qvel[0] = velocity;
    mj_step(model.get(), data.get());
    ASSERT_GT(data->ncon, 0);
    int witnesses = 0;
    std::string state_report;
    for (int contact_id = 0; contact_id < data->ncon; ++contact_id) {
      const mjContact& contact = data->contact[contact_id];
      ASSERT_EQ(contact.dim, 3);
      ASSERT_GE(contact.efc_address, 0);
      int first = contact.efc_address;
      ASSERT_LE(first + 4, data->nefc);
      bool first_active = data->efc_state[first] != mjCNSTRSTATE_SATISFIED;
      bool any_active = false;
      double force = 0;
      for (int edge = 0; edge < 4; ++edge) {
        any_active |= data->efc_state[first + edge] != mjCNSTRSTATE_SATISFIED;
        force += std::abs(data->efc_force[first + edge]);
        state_report += std::to_string(data->efc_state[first + edge]);
      }
      state_report += ";";
      if (!first_active && any_active) {
        ++witnesses;
        EXPECT_GT(force, 0);
      }
    }
    EXPECT_GT(witnesses, 0) << "missing first-edge-satisfied active contact; states: "
                            << state_report;
  }
}

TEST_F(AdjointValidationTest, ClampStockForwardMatrix) {
  struct ClampCase {
    const char* name;
    const char* joint;
    const char* actuator;
    mjtNum saturated_qfrc;
  };
  const ClampCase clamp_cases[] = {
      {"ctrlrange", "", "ctrllimited=\"true\" ctrlrange=\"-.05 .05\"", .5},
      {"forcerange", "", "forcelimited=\"true\" forcerange=\"-.5 .5\"", 5},
      {"actuatorfrcrange", "actuatorfrclimited=\"true\" actuatorfrcrange=\"-.5 .5\"", "", .5},
  };
  struct Zone { const char* name; mjtNum ctrl; bool clamped; };
  const Zone zones[] = {{"lower", -1, true}, {"interior", 0, false}, {"upper", 1, true}};
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
      data->qvel[0] = .1;
      data->ctrl[0] = zone.ctrl;
      mj_forward(model.get(), data.get());
      if (zone.clamped) {
        EXPECT_NEAR(std::abs(data->qfrc_actuator[0]), clamp.saturated_qfrc, 1e-12);
      }
      double losses[2];
      for (int side = 0; side < 2; ++side) {
        MjDataPtr perturbed = MakeData(model);
        mj_copyData(perturbed.get(), model.get(), data.get());
        perturbed->ctrl[0] += side == 0 ? 1e-6 : -1e-6;
        mj_step(model.get(), perturbed.get());
        losses[side] = perturbed->qpos[0] + perturbed->qvel[0];
      }
      double derivative = (losses[0] - losses[1]) / 2e-6;
      RecordProperty("ctrl_derivative", derivative);
      if (zone.clamped) {
        EXPECT_NEAR(derivative, 0, 1e-10);
      } else {
        EXPECT_NEAR(derivative, .101, 1e-8);
      }
      ++completed;
    }
  }
  EXPECT_EQ(completed, 9);
}

}  // namespace
}  // namespace mujoco
