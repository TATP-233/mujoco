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
#include "test/fixture.h"

namespace mujoco {
namespace {

using ::testing::NotNull;

class AdjointPassiveTest : public MujocoTest {
 protected:
  void SetUp() override {
    static constexpr char xml[] = R"(
    <mujoco>
      <option solver="Newton"><flag island="disable"/></option>
      <worldbody>
        <body><joint name="j0" type="slide" stiffness="2" springref=".1" damping=".3"/>
          <geom size=".1" mass="1"/></body>
        <body pos="1 0 0"><joint name="j1" type="hinge" stiffness="4" springref="-.2" damping=".5"/>
          <geom size=".1" mass="1"/></body>
      </worldbody>
      <actuator><general joint="j0" gainprm="1" damping=".7"/></actuator>
    </mujoco>)";
    char error[1024] = "";
    model_ = LoadModelFromString(xml, error, sizeof(error));
    ASSERT_THAT(model_.get(), NotNull()) << error;
    data_ = MakeData(model_);
    data_->qpos[0] = .4;
    data_->qpos[1] = -.3;
    data_->qvel[0] = .6;
    data_->qvel[1] = -.8;
    mj_forward(model_.get(), data_.get());
  }

  MjModelPtr model_;
  MjDataPtr data_;
};

TEST_F(AdjointPassiveTest, LinearSpringDamperAndJvpVjp) {
  std::vector<mjtNum> unit_bar = {1, 1};
  std::vector<mjtNum> grad_qpos(2, 0);
  std::vector<mjtNum> grad_qvel(2, 0);
  ASSERT_EQ(mjd_passiveStateAdjoint(model_.get(), data_.get(), unit_bar.data(),
                                    grad_qpos.data(), grad_qvel.data()), mjADJ_OK);
  EXPECT_EQ(grad_qpos, std::vector<mjtNum>({-2, -4}));
  EXPECT_EQ(grad_qvel, std::vector<mjtNum>({-1, -.5}));

  std::mt19937_64 rng(17);
  std::normal_distribution<double> normal;
  for (int sample = 0; sample < 4; ++sample) {
    std::vector<mjtNum> bar = {normal(rng), normal(rng)};
    std::vector<mjtNum> dir_qpos = {normal(rng), normal(rng)};
    std::vector<mjtNum> dir_qvel = {normal(rng), normal(rng)};
    ASSERT_EQ(mjd_passiveStateAdjoint(model_.get(), data_.get(), bar.data(),
                                      grad_qpos.data(), grad_qvel.data()), mjADJ_OK);
    MjDataPtr plus = MakeData(model_);
    MjDataPtr minus = MakeData(model_);
    mj_copyData(plus.get(), model_.get(), data_.get());
    mj_copyData(minus.get(), model_.get(), data_.get());
    constexpr mjtNum eps = 1e-6;
    for (int i = 0; i < 2; ++i) {
      plus->qpos[i] += eps*dir_qpos[i];
      minus->qpos[i] -= eps*dir_qpos[i];
      plus->qvel[i] += eps*dir_qvel[i];
      minus->qvel[i] -= eps*dir_qvel[i];
    }
    mj_forward(model_.get(), plus.get());
    mj_forward(model_.get(), minus.get());
    mjtNum jvp_vjp = 0;
    mjtNum vjp_jvp = 0;
    for (int i = 0; i < 2; ++i) {
      jvp_vjp += bar[i]*(plus->qfrc_passive[i] - minus->qfrc_passive[i])/(2*eps);
      vjp_jvp += grad_qpos[i]*dir_qpos[i] + grad_qvel[i]*dir_qvel[i];
    }
    EXPECT_NEAR(vjp_jvp, jvp_vjp, 2e-8);
  }
}

TEST_F(AdjointPassiveTest, DisableFlagsSelectForceBlocks) {
  std::vector<mjtNum> bar = {1, 1};
  std::vector<mjtNum> grad_qpos(2, 1);
  std::vector<mjtNum> grad_qvel(2, 1);
  model_->opt.disableflags |= mjDSBL_SPRING;
  ASSERT_EQ(mjd_passiveStateAdjoint(model_.get(), data_.get(), bar.data(),
                                    grad_qpos.data(), grad_qvel.data()), mjADJ_OK);
  EXPECT_EQ(grad_qpos, std::vector<mjtNum>({0, 0}));
  EXPECT_EQ(grad_qvel, std::vector<mjtNum>({-1, -.5}));
  model_->opt.disableflags = mjDSBL_ISLAND | mjDSBL_DAMPER;
  ASSERT_EQ(mjd_passiveStateAdjoint(model_.get(), data_.get(), bar.data(),
                                    grad_qpos.data(), grad_qvel.data()), mjADJ_OK);
  EXPECT_EQ(grad_qpos, std::vector<mjtNum>({-2, -4}));
  EXPECT_EQ(grad_qvel, std::vector<mjtNum>({0, 0}));
}

TEST_F(AdjointPassiveTest, UnsupportedTermsFailBeforeWriting) {
  std::vector<mjtNum> bar = {1, 1};
  std::vector<mjtNum> grad_qpos = {7, 8};
  std::vector<mjtNum> grad_qvel = {9, 10};
  model_->jnt_stiffnesspoly[0] = .2;
  EXPECT_EQ(mjd_passiveStateAdjoint(model_.get(), data_.get(), bar.data(),
                                    grad_qpos.data(), grad_qvel.data()), mjADJ_UNSUPPORTED);
  EXPECT_EQ(grad_qpos, std::vector<mjtNum>({7, 8}));
  EXPECT_EQ(grad_qvel, std::vector<mjtNum>({9, 10}));
  model_->jnt_stiffnesspoly[0] = 0;
  model_->dof_dampingpoly[0] = .3;
  EXPECT_EQ(mjd_passiveStateAdjoint(model_.get(), data_.get(), bar.data(),
                                    grad_qpos.data(), grad_qvel.data()), mjADJ_UNSUPPORTED);
  model_->dof_dampingpoly[0] = 0;
  model_->flg_gravcomp = 1;
  EXPECT_EQ(mjd_passiveStateAdjoint(model_.get(), data_.get(), bar.data(),
                                    grad_qpos.data(), grad_qvel.data()), mjADJ_UNSUPPORTED);
}

TEST_F(AdjointPassiveTest, FreeJointDampingIsRejected) {
  static constexpr char xml[] = R"(
  <mujoco><option solver="Newton"><flag island="disable"/></option>
    <worldbody><body><joint type="free" damping=".2"/><geom size=".1"/></body></worldbody>
  </mujoco>)";
  char error[1024] = "";
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());
  std::vector<mjtNum> bar(6, 1);
  std::vector<mjtNum> grad_qpos(6, 0);
  std::vector<mjtNum> grad_qvel(6, 0);
  EXPECT_EQ(mjd_passiveStateAdjoint(model.get(), data.get(), bar.data(),
                                    grad_qpos.data(), grad_qvel.data()), mjADJ_UNSUPPORTED);
}

}  // namespace
}  // namespace mujoco
