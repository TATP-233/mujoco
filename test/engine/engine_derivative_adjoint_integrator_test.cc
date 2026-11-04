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

class AdjointIntegratorTest : public MujocoTest {
 protected:
  void SetUp() override {
    static constexpr char xml[] = R"(
    <mujoco>
      <worldbody>
        <body pos=".1 -.2 .3" euler="20 -30 40"><freejoint/><geom size=".1"/></body>
        <body pos="1 0 0" euler="-15 25 35"><joint name="ball" type="ball"/><geom size=".1"/></body>
        <body pos="2 0 0"><joint name="hinge" type="hinge"/><geom size=".1"/></body>
        <body pos="3 0 0"><joint name="slide" type="slide"/><geom size=".1"/></body>
      </worldbody>
    </mujoco>)";
    char error[1024] = "";
    model_ = LoadModelFromString(xml, error, sizeof(error));
    ASSERT_THAT(model_.get(), NotNull()) << error;
    data_ = MakeData(model_);
    const mjtNum velocity[11] = {.4, -.3, .2, 5, -3, 2, .7, -1.1, 1.9, -.6, .8};
    mju_copy(data_->qvel, velocity, 11);
  }

  MjModelPtr model_;
  MjDataPtr data_;
};

TEST_F(AdjointIntegratorTest, MixedJointsJvpVjp) {
  constexpr mjtNum dt = .02;
  std::mt19937_64 rng(29);
  std::normal_distribution<double> normal;
  std::vector<mjtNum> grad_qpos(model_->nv);
  std::vector<mjtNum> grad_qvel(model_->nv);
  std::vector<mjtNum> qpos_plus(model_->nq);
  std::vector<mjtNum> qpos_minus(model_->nq);
  std::vector<mjtNum> qvel_plus(model_->nv);
  std::vector<mjtNum> qvel_minus(model_->nv);
  std::vector<mjtNum> tangent(model_->nv);
  for (int sample = 0; sample < 4; ++sample) {
    std::vector<mjtNum> bar(model_->nv);
    std::vector<mjtNum> dir_qpos(model_->nv);
    std::vector<mjtNum> dir_qvel(model_->nv);
    for (int i = 0; i < model_->nv; ++i) {
      bar[i] = normal(rng);
      dir_qpos[i] = normal(rng);
      dir_qvel[i] = normal(rng);
    }
    ASSERT_EQ(mjd_integratePosAdjoint(model_.get(), data_->qvel, dt, bar.data(),
                                      grad_qpos.data(), grad_qvel.data()), mjADJ_OK);
    mju_copy(qpos_plus.data(), data_->qpos, model_->nq);
    mju_copy(qpos_minus.data(), data_->qpos, model_->nq);
    constexpr mjtNum eps = 1e-6;
    mj_integratePos(model_.get(), qpos_plus.data(), dir_qpos.data(), eps);
    mj_integratePos(model_.get(), qpos_minus.data(), dir_qpos.data(), -eps);
    for (int i = 0; i < model_->nv; ++i) {
      qvel_plus[i] = data_->qvel[i] + eps*dir_qvel[i];
      qvel_minus[i] = data_->qvel[i] - eps*dir_qvel[i];
    }
    mj_integratePos(model_.get(), qpos_plus.data(), qvel_plus.data(), dt);
    mj_integratePos(model_.get(), qpos_minus.data(), qvel_minus.data(), dt);
    mj_differentiatePos(model_.get(), tangent.data(), 1, qpos_minus.data(), qpos_plus.data());
    mjtNum jvp_vjp = 0;
    mjtNum vjp_jvp = 0;
    for (int i = 0; i < model_->nv; ++i) {
      jvp_vjp += bar[i]*tangent[i]/(2*eps);
      vjp_jvp += grad_qpos[i]*dir_qpos[i] + grad_qvel[i]*dir_qvel[i];
    }
    EXPECT_NEAR(vjp_jvp, jvp_vjp, 5e-8) << sample;
  }
}

TEST_F(AdjointIntegratorTest, ZeroVelocityAndAliasing) {
  std::vector<mjtNum> zero(model_->nv, 0);
  std::vector<mjtNum> bar(model_->nv);
  for (int i = 0; i < model_->nv; ++i) bar[i] = .1*(i + 1);
  std::vector<mjtNum> grad_qpos(model_->nv, 0);
  std::vector<mjtNum> grad_qvel(model_->nv, 0);
  ASSERT_EQ(mjd_integratePosAdjoint(model_.get(), zero.data(), .02, bar.data(),
                                    grad_qpos.data(), grad_qvel.data()), mjADJ_OK);
  for (int i = 0; i < model_->nv; ++i) {
    EXPECT_NEAR(grad_qpos[i], bar[i], 1e-14);
    EXPECT_NEAR(grad_qvel[i], .02*bar[i], 1e-14);
  }
  EXPECT_EQ(mjd_integratePosAdjoint(model_.get(), zero.data(), .02, bar.data(),
                                    bar.data(), grad_qvel.data()), mjADJ_INVALID_ARGUMENT);
}

}  // namespace
}  // namespace mujoco
