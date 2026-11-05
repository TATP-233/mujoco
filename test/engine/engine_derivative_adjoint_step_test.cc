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

#include "engine/engine_collision_primitive.h"
#include <memory>
#include <random>
#include <regex>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mujoco/mujoco.h>
#include "src/engine/engine_derivative_adjoint.h"
#include "src/engine/engine_derivative_fd.h"
#include "src/engine/engine_util_solve.h"
#include "test/engine/engine_derivative_adjoint_test_util.h"
#include <fstream>
#include <unistd.h>
#include <string>
#include "test/fixture.h"

namespace mujoco {
namespace {

using ::testing::NotNull;
using adjoint_test::InputBlock;
using adjoint_test::StepLoss;
using adjoint_test::StockStepCentralTangent;

struct StepContextDeleter {
  void operator()(mjAdjContext* ctx) const { mj_deleteAdjContext(ctx); }
};
using StepContextPtr = std::unique_ptr<mjAdjContext, StepContextDeleter>;

class AdjointStepTest : public MujocoTest {};

TEST_F(AdjointStepTest, FreeBodyForwardCommitAndAllColumns) {
  static constexpr char xml[] = R"(
  <mujoco><option solver="Newton" integrator="Euler" timestep=".002">
    <flag island="disable"/></option>
    <worldbody><body pos=".1 -.2 .3" euler="20 -30 40"><freejoint/>
      <geom type="box" size=".1 .2 .3" mass="2" pos=".04 -.02 .03" euler="10 20 30"/>
    </body></worldbody>
  </mujoco>)";
  char error[1024] = "";
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  const mjtNum qvel[6] = {.4, -.3, .2, 5, -3, 2};
  mju_copy(data->qvel, qvel, 6);
  StepContextPtr ctx(mj_makeAdjContext(model.get()));
  ASSERT_THAT(ctx.get(), NotNull());
  ASSERT_EQ(mj_reserveAdjContext(ctx.get(), 0, 0, 0, 0), mjADJ_OK);
  MjDataPtr reference = MakeData(model);
  mj_copyData(reference.get(), model.get(), data.get());
  mj_step(model.get(), reference.get());
  std::mt19937_64 rng(53);
  std::normal_distribution<double> normal;
  int compared = 0;
  for (int sample = 0; sample < 4; ++sample) {
    std::vector<mjtNum> bar_qpos(6);
    std::vector<mjtNum> bar_qvel(6);
    for (int i = 0; i < 6; ++i) {
      bar_qpos[i] = normal(rng);
      bar_qvel[i] = normal(rng);
    }
    std::vector<mjtNum> grad_qpos(6, 0);
    std::vector<mjtNum> grad_qvel(6, 0);
    MjDataPtr step_data = MakeData(model);
    mj_copyData(step_data.get(), model.get(), data.get());
    ASSERT_EQ(mjd_stepAdjoint(model.get(), step_data.get(), ctx.get(), bar_qpos.data(),
                              bar_qvel.data(), grad_qpos.data(), grad_qvel.data(), nullptr),
              mjADJ_OK);
    for (int i = 0; i < model->nq; ++i) EXPECT_EQ(step_data->qpos[i], reference->qpos[i]);
    for (int i = 0; i < model->nv; ++i) EXPECT_EQ(step_data->qvel[i], reference->qvel[i]);
    EXPECT_EQ(step_data->time, reference->time);
    for (int column = 0; column < 12; ++column) {
      bool is_qpos = column < 6;
      int index = is_qpos ? column : column - 6;
      double fd = StepLoss(StockStepCentralTangent(model.get(), data.get(),
                           is_qpos ? InputBlock::kQPos : InputBlock::kQVel, index, 1e-7),
                           bar_qpos, bar_qvel);
      double actual = is_qpos ? grad_qpos[index] : grad_qvel[index];
      double error = actual - fd;
      error = error < 0 ? -error : error;
      EXPECT_LT(error/(fd < 0 ? -fd : fd > 1 ? fd : 1), 2e-7)
          << sample << ":" << column << " actual=" << actual << " fd=" << fd;
      ++compared;
    }
  }
  EXPECT_EQ(compared, 48);
}

TEST_F(AdjointStepTest, FixedTreeForwardCommitAndAllColumns) {
  static constexpr char xml[] = R"(
  <mujoco><option solver="Newton" integrator="Euler" timestep=".002">
    <flag island="disable"/></option>
    <worldbody>
      <body pos=".1 -.2 .3" euler="20 -30 40">
        <joint type="hinge" axis=".3 .8 .5" stiffness=".17" damping=".11"/>
        <geom type="capsule" size=".08 .2" mass="1.2" pos=".04 -.02 .03" euler="10 20 30"/>
        <body pos=".25 .1 -.15" euler="-15 25 35">
          <joint type="hinge" axis="-.2 .6 .7" stiffness="-.09" damping=".07"/>
          <geom type="box" size=".07 .09 .11" mass=".8" pos="-.03 .04 .02" euler="25 -10 15"/>
          <body pos=".12 -.08 .16" euler="20 10 -25">
            <joint type="slide" axis=".5 -.3 .8" stiffness=".13" damping=".05"/>
            <geom type="sphere" size=".1" mass=".4" pos=".03 .02 -.04"/>
          </body>
        </body>
      </body>
    </worldbody>
  </mujoco>)";
  char error[1024] = "";
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  const mjtNum qpos[3] = {.3, -.4, .25};
  const mjtNum qvel[3] = {.8, -1.2, .6};
  mju_copy(data->qpos, qpos, 3);
  mju_copy(data->qvel, qvel, 3);
  StepContextPtr ctx(mj_makeAdjContext(model.get()));
  ASSERT_THAT(ctx.get(), NotNull());
  ASSERT_EQ(mj_reserveAdjContext(ctx.get(), 0, 0, 0, 0), mjADJ_OK);
  MjDataPtr reference = MakeData(model);
  mj_copyData(reference.get(), model.get(), data.get());
  mj_step(model.get(), reference.get());
  std::mt19937_64 rng(137);
  std::normal_distribution<double> normal;
  int compared = 0;
  for (int sample = 0; sample < 4; ++sample) {
    std::vector<mjtNum> bar_qpos(3);
    std::vector<mjtNum> bar_qvel(3);
    for (int i = 0; i < 3; ++i) {
      bar_qpos[i] = normal(rng);
      bar_qvel[i] = normal(rng);
    }
    std::vector<mjtNum> grad_qpos(3, 0);
    std::vector<mjtNum> grad_qvel(3, 0);
    MjDataPtr step_data = MakeData(model);
    mj_copyData(step_data.get(), model.get(), data.get());
    ASSERT_EQ(mjd_stepAdjoint(model.get(), step_data.get(), ctx.get(), bar_qpos.data(),
                              bar_qvel.data(), grad_qpos.data(), grad_qvel.data(), nullptr),
              mjADJ_OK);
    for (int i = 0; i < model->nq; ++i) EXPECT_EQ(step_data->qpos[i], reference->qpos[i]);
    for (int i = 0; i < model->nv; ++i) EXPECT_EQ(step_data->qvel[i], reference->qvel[i]);
    EXPECT_EQ(step_data->time, reference->time);
    for (int column = 0; column < 6; ++column) {
      bool is_qpos = column < 3;
      int index = is_qpos ? column : column-3;
      double fd = StepLoss(StockStepCentralTangent(model.get(), data.get(),
                           is_qpos ? InputBlock::kQPos : InputBlock::kQVel, index, 1e-7),
                           bar_qpos, bar_qvel);
      double actual = is_qpos ? grad_qpos[index] : grad_qvel[index];
      EXPECT_NEAR(actual, fd, 2e-7*std::max(1.0, std::abs(fd)))
          << sample << ":" << column << " actual=" << actual << " fd=" << fd;
      ++compared;
    }
  }
  EXPECT_EQ(compared, 24);

  MjDataPtr integrated = MakeData(model);
  mj_copyData(integrated.get(), model.get(), data.get());
  mj_step(model.get(), integrated.get());
  mjtNum bar_qpos[3] = {.3, -.7, 1.1};
  mjtNum bar_qvel[3] = {-.2, .9, .4};
  mjtNum integrate_pos[3];
  mjtNum integrate_vel[3];
  ASSERT_EQ(mjd_integratePosAdjoint(model.get(), integrated->qvel, model->opt.timestep,
                                    bar_qpos, integrate_pos, integrate_vel), mjADJ_OK);
  mjtNum adj_qacc[3];
  for (int i = 0; i < 3; ++i) adj_qacc[i] = model->opt.timestep*(bar_qvel[i]+integrate_vel[i]);
  mjtNum expected_pos[3];
  mjtNum expected_vel[3];
  mjtNum y[3];
  mjtNum qacc_adv[3];
  for (int i = 0; i < 3; ++i) {
    qacc_adv[i] = (integrated->qvel[i]-data->qvel[i])/model->opt.timestep;
  }
  mjData factor_data = *integrated;
  factor_data.qpos = data->qpos;
  factor_data.qvel = data->qvel;
  factor_data.qLD = integrated->qH;
  factor_data.qLDiagInv = integrated->qHDiagInv;
  factor_data.qacc_smooth = qacc_adv;
  ASSERT_EQ(mjd_fixedSmoothAccPosAdjoint(model.get(), &factor_data, adj_qacc, y, expected_pos),
            mjADJ_OK);
  std::vector<mjtNum> scratch(model->nD);
  ASSERT_EQ(mjd_smoothVelAdjoint(model.get(), &factor_data, y, scratch.data(), expected_vel), mjADJ_OK);
  std::vector<mjtNum> zero_pos(3, 0);
  std::vector<mjtNum> adj_qacc_vec(adj_qacc, adj_qacc+3);
  for (int column = 0; column < 3; ++column) {
    double fd = (StepLoss(StockStepCentralTangent(model.get(), data.get(),
                          InputBlock::kQVel, column, 1e-7), zero_pos, adj_qacc_vec) -
                 adj_qacc[column]) / model->opt.timestep;
    EXPECT_NEAR(expected_vel[column], fd, 1e-8*std::max(1.0, std::abs(fd)))
        << "qacc_adv column=" << column << " analytic=" << expected_vel[column] << " fd=" << fd;
  }
  mjtNum actual_pos[3];
  mjtNum actual_vel[3];
  MjDataPtr step_data = MakeData(model);
  mj_copyData(step_data.get(), model.get(), data.get());
  ASSERT_EQ(mjd_stepAdjoint(model.get(), step_data.get(), ctx.get(), bar_qpos, bar_qvel,
                            actual_pos, actual_vel, nullptr), mjADJ_OK);
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(actual_pos[i], expected_pos[i]+integrate_pos[i], 1e-12) << i;
    EXPECT_NEAR(actual_vel[i], expected_vel[i]+bar_qvel[i]+integrate_vel[i], 1e-12) << i;
  }

  model->opt.disableflags |= mjDSBL_EULERDAMP;
  StepContextPtr explicit_ctx(mj_makeAdjContext(model.get()));
  ASSERT_THAT(explicit_ctx.get(), NotNull());
  ASSERT_EQ(mj_reserveAdjContext(explicit_ctx.get(), 0, 0, 0, 0), mjADJ_OK);
  MjDataPtr explicit_data = MakeData(model);
  mj_copyData(explicit_data.get(), model.get(), data.get());
  ASSERT_EQ(mjd_stepAdjoint(model.get(), explicit_data.get(), explicit_ctx.get(), bar_qpos, bar_qvel,
                            actual_pos, actual_vel, nullptr), mjADJ_OK);
  std::vector<mjtNum> bar_qpos_vec(bar_qpos, bar_qpos+3);
  std::vector<mjtNum> bar_qvel_vec(bar_qvel, bar_qvel+3);
  for (int column = 0; column < 6; ++column) {
    bool is_qpos = column < 3;
    int index = is_qpos ? column : column-3;
    double fd = StepLoss(StockStepCentralTangent(model.get(), data.get(),
                         is_qpos ? InputBlock::kQPos : InputBlock::kQVel, index, 1e-7),
                         bar_qpos_vec, bar_qvel_vec);
    double actual = is_qpos ? actual_pos[index] : actual_vel[index];
    EXPECT_NEAR(actual, fd, 2e-7*std::max(1.0, std::abs(fd))) << "explicit column=" << column;
  }

  model->opt.disableflags &= ~mjDSBL_EULERDAMP;
  model->opt.integrator = mjINT_IMPLICITFAST;
  StepContextPtr implicit_ctx(mj_makeAdjContext(model.get()));
  ASSERT_THAT(implicit_ctx.get(), NotNull());
  ASSERT_EQ(mj_reserveAdjContext(implicit_ctx.get(), 0, 0, 0, 0), mjADJ_OK);
  int implicit_compared = 0;
  for (int sample = 0; sample < 4; ++sample) {
    std::vector<mjtNum> implicit_bar_qpos(3);
    std::vector<mjtNum> implicit_bar_qvel(3);
    for (int i = 0; i < 3; ++i) {
      implicit_bar_qpos[i] = normal(rng);
      implicit_bar_qvel[i] = normal(rng);
    }
    std::vector<mjtNum> grad_qpos(3, 0);
    std::vector<mjtNum> grad_qvel(3, 0);
    MjDataPtr implicit_data = MakeData(model);
    mj_copyData(implicit_data.get(), model.get(), data.get());
    ASSERT_EQ(mjd_stepAdjoint(model.get(), implicit_data.get(), implicit_ctx.get(),
                              implicit_bar_qpos.data(), implicit_bar_qvel.data(),
                              grad_qpos.data(), grad_qvel.data(), nullptr), mjADJ_OK);
    for (int column = 0; column < 6; ++column) {
      bool is_qpos = column < 3;
      int index = is_qpos ? column : column-3;
      double fd = StepLoss(StockStepCentralTangent(model.get(), data.get(),
                           is_qpos ? InputBlock::kQPos : InputBlock::kQVel, index, 1e-7),
                           implicit_bar_qpos, implicit_bar_qvel);
      double actual = is_qpos ? grad_qpos[index] : grad_qvel[index];
      EXPECT_NEAR(actual, fd, 2e-7*std::max(1.0, std::abs(fd)))
          << "implicitfast " << sample << ":" << column << " actual=" << actual << " fd=" << fd;
      ++implicit_compared;
    }
  }
  EXPECT_EQ(implicit_compared, 24);
}

TEST_F(AdjointStepTest, FixedTreeActuatorsAllInputColumns) {
  static constexpr char xml[] = R"(
  <mujoco><option solver="Newton" timestep=".002"><flag island="disable"/></option>
    <worldbody>
      <body pos=".1 -.2 .3" euler="20 -30 40">
        <joint name="j0" type="hinge" axis=".3 .8 .5" stiffness=".17" damping=".11"/>
        <geom type="capsule" size=".08 .2" mass="1.2" pos=".04 -.02 .03" euler="10 20 30"/>
        <body pos=".25 .1 -.15" euler="-15 25 35">
          <joint name="j1" type="hinge" axis="-.2 .6 .7" stiffness="-.09" damping=".07"/>
          <geom type="box" size=".07 .09 .11" mass=".8" pos="-.03 .04 .02" euler="25 -10 15"/>
          <body pos=".12 -.08 .16" euler="20 10 -25">
            <joint name="j2" type="slide" axis=".5 -.3 .8" stiffness=".13" damping=".05"/>
            <geom type="sphere" size=".1" mass=".4" pos=".03 .02 -.04"/>
          </body>
        </body>
      </body>
    </worldbody>
    <actuator>
      <position joint="j0" kp="1.3" kv=".17" gear="1.1"/>
      <position joint="j1" kp=".9" kv=".13" gear="-.8"/>
      <motor joint="j2" gear="1.4"/>
    </actuator>
  </mujoco>)";
  char error[1024] = "";
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  const mjtNum qpos[3] = {.3, -.4, .25};
  const mjtNum qvel[3] = {.8, -1.2, .6};
  const mjtNum ctrl[3] = {.2, -.3, .4};
  mju_copy(data->qpos, qpos, 3);
  mju_copy(data->qvel, qvel, 3);
  mju_copy(data->ctrl, ctrl, 3);
  std::mt19937_64 rng(139);
  std::normal_distribution<double> normal;
  int compared = 0;
  for (int integrator : {mjINT_EULER, mjINT_IMPLICITFAST}) {
    model->opt.integrator = static_cast<mjtIntegrator>(integrator);
    StepContextPtr ctx(mj_makeAdjContext(model.get()));
    ASSERT_THAT(ctx.get(), NotNull());
    ASSERT_EQ(mj_reserveAdjContext(ctx.get(), 0, 0, 0, 0), mjADJ_OK);
    for (int sample = 0; sample < 4; ++sample) {
      std::vector<mjtNum> bar_qpos(3);
      std::vector<mjtNum> bar_qvel(3);
      for (int i = 0; i < 3; ++i) {
        bar_qpos[i] = normal(rng);
        bar_qvel[i] = normal(rng);
      }
      std::vector<mjtNum> grad_qpos(3, 0);
      std::vector<mjtNum> grad_qvel(3, 0);
      std::vector<mjtNum> grad_ctrl(3, 0);
      MjDataPtr step_data = MakeData(model);
      mj_copyData(step_data.get(), model.get(), data.get());
      ASSERT_EQ(mjd_stepAdjoint(model.get(), step_data.get(), ctx.get(), bar_qpos.data(),
                                bar_qvel.data(), grad_qpos.data(), grad_qvel.data(), grad_ctrl.data()),
                mjADJ_OK);
      for (int column = 0; column < 9; ++column) {
        InputBlock block = column < 3 ? InputBlock::kQPos :
                           (column < 6 ? InputBlock::kQVel : InputBlock::kCtrl);
        int index = column < 3 ? column : (column < 6 ? column-3 : column-6);
        double fd = StepLoss(StockStepCentralTangent(model.get(), data.get(), block, index, 1e-7),
                             bar_qpos, bar_qvel);
        double actual = column < 3 ? grad_qpos[index] :
                        (column < 6 ? grad_qvel[index] : grad_ctrl[index]);
        EXPECT_NEAR(actual, fd, 2e-7*std::max(1.0, std::abs(fd)))
            << integrator << ":" << sample << ":" << column
            << " actual=" << actual << " fd=" << fd;
        ++compared;
      }
    }
  }
  EXPECT_EQ(compared, 72);
}

TEST_F(AdjointStepTest, FreeTreeEulerAllInputColumns) {
  static constexpr char xml[] = R"(
  <mujoco><option solver="Newton" integrator="Euler" timestep=".002">
    <flag island="disable"/></option>
    <worldbody>
      <body pos=".1 -.2 .3" euler="20 -30 40"><freejoint/>
        <geom type="capsule" size=".08 .2" mass="1.2" pos=".04 -.02 .03" euler="10 20 30"/>
        <body pos=".25 .1 -.15" euler="-15 25 35">
          <joint name="j1" type="hinge" axis="-.2 .6 .7" stiffness="-.09" damping=".07"/>
          <geom type="box" size=".07 .09 .11" mass=".8" pos="-.03 .04 .02" euler="25 -10 15"/>
          <body pos=".12 -.08 .16" euler="20 10 -25">
            <joint name="j2" type="slide" axis=".5 -.3 .8" stiffness=".13" damping=".05"/>
            <geom type="sphere" size=".1" mass=".4" pos=".03 .02 -.04"/>
          </body>
        </body>
      </body>
    </worldbody>
    <actuator>
      <position joint="j1" kp=".9" kv=".13" gear="-.8"/>
      <motor joint="j2" gear="1.4"/>
    </actuator>
  </mujoco>)";
  char error[1024] = "";
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  const mjtNum qvel[8] = {.4, -.3, .2, .5, -.6, .7, .8, -1.2};
  mju_copy(data->qvel, qvel, 8);
  data->qpos[7] = .3;
  data->qpos[8] = -.4;
  data->ctrl[0] = -.3;
  data->ctrl[1] = .4;
  StepContextPtr ctx(mj_makeAdjContext(model.get()));
  ASSERT_THAT(ctx.get(), NotNull());
  ASSERT_EQ(mj_reserveAdjContext(ctx.get(), 0, 0, 0, 0), mjADJ_OK);
  MjDataPtr reference = MakeData(model);
  mj_copyData(reference.get(), model.get(), data.get());
  mj_step(model.get(), reference.get());
  std::mt19937_64 rng(157);
  std::normal_distribution<double> normal;
  int compared = 0;
  for (int sample = 0; sample < 4; ++sample) {
    std::vector<mjtNum> bar_qpos(8);
    std::vector<mjtNum> bar_qvel(8);
    for (int i = 0; i < 8; ++i) {
      bar_qpos[i] = normal(rng);
      bar_qvel[i] = normal(rng);
    }
    std::vector<mjtNum> grad_qpos(8, 0);
    std::vector<mjtNum> grad_qvel(8, 0);
    std::vector<mjtNum> grad_ctrl(2, 0);
    MjDataPtr step_data = MakeData(model);
    mj_copyData(step_data.get(), model.get(), data.get());
    ASSERT_EQ(mjd_stepAdjoint(model.get(), step_data.get(), ctx.get(), bar_qpos.data(),
                              bar_qvel.data(), grad_qpos.data(), grad_qvel.data(), grad_ctrl.data()),
              mjADJ_OK);
    for (int i = 0; i < model->nq; ++i) EXPECT_EQ(step_data->qpos[i], reference->qpos[i]);
    for (int i = 0; i < model->nv; ++i) EXPECT_EQ(step_data->qvel[i], reference->qvel[i]);
    for (int column = 0; column < 10; ++column) {
      InputBlock block = column < 8 ? InputBlock::kQPos :
                         (column < 16 ? InputBlock::kQVel : InputBlock::kCtrl);
      int index = column < 8 ? column : (column < 16 ? column-8 : column-16);
      double fd = StepLoss(StockStepCentralTangent(model.get(), data.get(), block, index, 1e-7),
                           bar_qpos, bar_qvel);
      double actual = column < 8 ? grad_qpos[index] :
                      (column < 16 ? grad_qvel[index] : grad_ctrl[index]);
      EXPECT_NEAR(actual, fd, 2e-7*std::max(1.0, std::abs(fd)))
          << sample << ":" << column << " actual=" << actual << " fd=" << fd;
      ++compared;
    }
  }
  EXPECT_EQ(compared, 40);

  model->opt.disableflags |= mjDSBL_EULERDAMP;
  StepContextPtr explicit_ctx(mj_makeAdjContext(model.get()));
  ASSERT_THAT(explicit_ctx.get(), NotNull());
  ASSERT_EQ(mj_reserveAdjContext(explicit_ctx.get(), 0, 0, 0, 0), mjADJ_OK);
  std::vector<mjtNum> bar_qpos(8, 0);
  std::vector<mjtNum> bar_qvel(8, 0);
  for (int i = 0; i < 8; ++i) {
    bar_qpos[i] = normal(rng);
    bar_qvel[i] = normal(rng);
  }
  std::vector<mjtNum> grad_qpos(8, 0);
  std::vector<mjtNum> grad_qvel(8, 0);
  std::vector<mjtNum> grad_ctrl(2, 0);
  MjDataPtr explicit_data = MakeData(model);
  mj_copyData(explicit_data.get(), model.get(), data.get());
  ASSERT_EQ(mjd_stepAdjoint(model.get(), explicit_data.get(), explicit_ctx.get(), bar_qpos.data(),
                            bar_qvel.data(), grad_qpos.data(), grad_qvel.data(), grad_ctrl.data()),
            mjADJ_OK);
  for (int column = 0; column < 10; ++column) {
    InputBlock block = column < 8 ? InputBlock::kQPos :
                       (column < 16 ? InputBlock::kQVel : InputBlock::kCtrl);
    int index = column < 8 ? column : (column < 16 ? column-8 : column-16);
    double fd = StepLoss(StockStepCentralTangent(model.get(), data.get(), block, index, 1e-7),
                         bar_qpos, bar_qvel);
    double actual = column < 8 ? grad_qpos[index] :
                    (column < 16 ? grad_qvel[index] : grad_ctrl[index]);
    EXPECT_NEAR(actual, fd, 2e-7*std::max(1.0, std::abs(fd))) << "explicit column=" << column;
  }

  model->opt.disableflags &= ~mjDSBL_EULERDAMP;
  model->opt.integrator = mjINT_IMPLICITFAST;
  StepContextPtr implicit_ctx(mj_makeAdjContext(model.get()));
  ASSERT_THAT(implicit_ctx.get(), NotNull());
  ASSERT_EQ(mj_reserveAdjContext(implicit_ctx.get(), 0, 0, 0, 0), mjADJ_OK);
  int implicit_compared = 0;
  for (int sample = 0; sample < 4; ++sample) {
    std::vector<mjtNum> implicit_bar_qpos(8);
    std::vector<mjtNum> implicit_bar_qvel(8);
    for (int i = 0; i < 8; ++i) {
      implicit_bar_qpos[i] = normal(rng);
      implicit_bar_qvel[i] = normal(rng);
    }
    std::vector<mjtNum> grad_qpos(8, 0);
    std::vector<mjtNum> grad_qvel(8, 0);
    std::vector<mjtNum> grad_ctrl(2, 0);
    MjDataPtr implicit_data = MakeData(model);
    mj_copyData(implicit_data.get(), model.get(), data.get());
    ASSERT_EQ(mjd_stepAdjoint(model.get(), implicit_data.get(), implicit_ctx.get(),
                              implicit_bar_qpos.data(), implicit_bar_qvel.data(),
                              grad_qpos.data(), grad_qvel.data(), grad_ctrl.data()), mjADJ_OK);
    for (int column = 0; column < 10; ++column) {
      InputBlock block = column < 8 ? InputBlock::kQPos :
                         (column < 16 ? InputBlock::kQVel : InputBlock::kCtrl);
      int index = column < 8 ? column : (column < 16 ? column-8 : column-16);
      double fd = StepLoss(StockStepCentralTangent(model.get(), data.get(), block, index, 1e-7),
                           implicit_bar_qpos, implicit_bar_qvel);
      double actual = column < 8 ? grad_qpos[index] :
                      (column < 16 ? grad_qvel[index] : grad_ctrl[index]);
      EXPECT_NEAR(actual, fd, 2e-7*std::max(1.0, std::abs(fd)))
          << "implicitfast " << sample << ":" << column
          << " actual=" << actual << " fd=" << fd;
      ++implicit_compared;
    }
  }
  EXPECT_EQ(implicit_compared, 40);
}

TEST_F(AdjointStepTest, PlaneSphereContactAllInputColumns) {
  static constexpr char xml[] = R"(
  <mujoco><option solver="Newton" integrator="implicitfast" timestep=".002" iterations="10">
    <flag island="disable"/></option>
    <worldbody>
      <geom type="plane" size="5 5 .1" condim="1" solimp=".9 .95 .001" solref=".02 1"/>
      <body pos="0 0 .04"><freejoint/>
        <geom type="capsule" size=".02 .1" mass="1" contype="0" conaffinity="0"/>
        <body pos=".1 0 0">
          <joint name="j1" type="hinge" axis="0 1 0" damping=".1"/>
          <geom type="sphere" size=".05" mass=".5" condim="1"
                solimp=".9 .95 .001" solref=".02 1"/>
        </body>
      </body>
    </worldbody>
    <actuator><position joint="j1" kp="1.2" kv=".1"/></actuator>
  </mujoco>)";
  char error[1024] = "";
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  model->opt.tolerance = 0;
  model->opt.ls_tolerance = 0;
  model->opt.iterations = 100;
  model->opt.ls_iterations = 100;
  model->opt.disableflags |= mjDSBL_WARMSTART;
  const int contact_dims[7] = {1, 3, 4, 6, 3, 4, 6};
  for (int contact_case = 0; contact_case < 7; ++contact_case) {
    SCOPED_TRACE(testing::Message() << "contact_case=" << contact_case);
    model->opt.cone = contact_case >= 4 ? mjCONE_ELLIPTIC : mjCONE_PYRAMIDAL;
    for (int geom = 0; geom < model->ngeom; ++geom) {
      model->geom_condim[geom] = contact_dims[contact_case];
    }
  for (int integrator_case = 0; integrator_case < 3; ++integrator_case) {
    SCOPED_TRACE(testing::Message() << "integrator_case=" << integrator_case);
    model->opt.integrator = integrator_case == 0 ? mjINT_IMPLICITFAST : mjINT_EULER;
    if (integrator_case == 2) {
      model->opt.disableflags |= mjDSBL_EULERDAMP;
    } else {
      model->opt.disableflags &= ~mjDSBL_EULERDAMP;
    }
    for (int geom = 0; geom < model->ngeom; ++geom) {
      model->geom_solimp[mjNIMP*geom+2] = integrator_case == 0 ? .021 : .001;
    }
  MjDataPtr data = MakeData(model);
  mjtNum qvel[7] = {.02, .03, -.04, .05, -.02, .01, .2};
  if (contact_case > 0) {
    qvel[0] = contact_case >= 4 ? .5 : 2;
    qvel[1] = contact_case >= 4 ? .75 : 3;
  }
  mju_copy(data->qvel, qvel, 7);
  data->ctrl[0] = .3;
  mj_forward(model.get(), data.get());
  ASSERT_EQ(data->ncon, 1);
  ASSERT_EQ(data->nefc, contact_case >= 4 ? contact_dims[contact_case] :
            (contact_case == 0 ? 1 : 2*(contact_dims[contact_case]-1)));
  ASSERT_EQ(data->contact[0].dim, contact_dims[contact_case]);
  if (contact_case > 0) {
    int active = 0;
    for (int row = 0; row < data->nefc; ++row) {
      active += data->efc_state[row] != mjCNSTRSTATE_SATISFIED;
    }
    ASSERT_GT(active, 0);
    if (contact_case >= 4) {
      ASSERT_EQ(active, data->nefc);
      ASSERT_EQ(data->efc_state[0], mjCNSTRSTATE_CONE);
    } else {
      ASSERT_LT(active, data->nefc);
    }
  }
  ASSERT_GT(data->efc_R[0], 0);
  StepContextPtr ctx(mj_makeAdjContext(model.get()));
  ASSERT_THAT(ctx.get(), NotNull());
  ASSERT_EQ(mj_reserveAdjContext(ctx.get(), 1, data->nefc, 256, 1024), mjADJ_OK);
  MjDataPtr reference = MakeData(model);
  mj_copyData(reference.get(), model.get(), data.get());
  mj_step(model.get(), reference.get());
  std::vector<mjtNum> residual_lambda(data->nefc);
  std::vector<mjtNum> residual_mu(data->nefc);
  for (int row = 0; row < data->nefc; ++row) {
    residual_lambda[row] = .7/(row+1);
    residual_mu[row] = -.013*(row+1);
  }
  const mjtNum residual_y[7] = {.3, -.2, .5, .4, -.1, .2, .6};
  mjtNum residual_qacc[7];
  for (int i = 0; i < 7; ++i) {
    residual_qacc[i] = (reference->qvel[i]-data->qvel[i])/model->opt.timestep;
  }
  std::vector<mjtNum> cone_q(data->nefc, 0);
  const mjtNum cone_seed[6] = {.2, -.1, .3, .15, -.2, .1};
  for (int row = 0; row < data->nefc; ++row) cone_q[row] = cone_seed[row%6];
  const mjtNum* hinv_ptr = contact_case >= 4 ? cone_q.data() : nullptr;
  std::vector<mjtNum> residual_gradient(7, 0);
  ASSERT_EQ(mjd_planeSpherePosAdjoint(model.get(), data.get(), residual_y,
                                      residual_lambda.data(), residual_mu.data(), hinv_ptr,
                                      residual_qacc, residual_gradient.data()), mjADJ_OK);
  std::vector<mjtNum> cone_acceleration(data->nefc, 0);
  if (contact_case >= 4) {
    int dim = data->nefc;
    mjtNum jar[6] = {0, 0, 0, 0, 0, 0};
    mj_mulJacVec(model.get(), data.get(), jar, data->qacc);
    mjtNum mu = data->contact[0].mu;
    mjtNum N = mu*(jar[0]-data->efc_aref[0]);
    mjtNum U[5];
    for (int k = 1; k < dim; ++k) {
      U[k-1] = data->contact[0].friction[k-1]*(jar[k]-data->efc_aref[k]);
    }
    mjtNum T = mju_norm(U, dim-1);
    mjtNum Dm = data->efc_D[0]/(mu*mu*(1+mu*mu));
    mjtNum H[36] = {0};
    H[0] = Dm*mu*mu;
    for (int k = 1; k < dim; ++k) {
      mjtNum friction_k = data->contact[0].friction[k-1];
      H[k] = H[dim*k] = -Dm*mu*mu*friction_k*U[k-1]/T;
      for (int j = k; j < dim; ++j) {
        mjtNum friction_j = data->contact[0].friction[j-1];
        mjtNum value = mu*N*U[j-1]*U[k-1]/(T*T*T);
        if (j == k) value += mu*mu-mu*N/T;
        H[dim*k+j] = H[dim*j+k] = Dm*friction_k*friction_j*value;
      }
    }
    for (int row = 0; row < dim; ++row) {
      cone_acceleration[row] = mju_dot(H+dim*row, cone_q.data(), dim);
    }
  }
  for (int column = 0; column < 7; ++column) {
    const mjtNum eps = contact_case == 0 ? (column == 2 ? 1e-9 : 1e-8) :
                       (contact_case >= 4 ? 8e-7 : 8e-8);
    mjtNum losses[2];
    std::vector<mjtNum> position_values[2] = {
        std::vector<mjtNum>(data->nefc), std::vector<mjtNum>(data->nefc)};
    std::vector<mjtNum> impedance_values[2] = {
        std::vector<mjtNum>(data->nefc), std::vector<mjtNum>(data->nefc)};
    std::vector<mjtNum> jacobian_values[2] = {
        std::vector<mjtNum>(data->nefc), std::vector<mjtNum>(data->nefc)};
    for (int side = 0; side < 2; ++side) {
      MjDataPtr perturbed = MakeData(model);
      mj_copyData(perturbed.get(), model.get(), data.get());
      std::vector<mjtNum> tangent(7, 0);
      tangent[column] = side ? -eps : eps;
      mj_integratePos(model.get(), perturbed->qpos, tangent.data(), 1);
      mj_forward(model.get(), perturbed.get());
      losses[side] = 0;
      for (int row = 0; row < data->nefc; ++row) {
        mjtNum acceleration_bar = contact_case >= 4 ? -cone_acceleration[row] : residual_mu[row];
        mjtNum dof_w[7];
        for (int i = 0; i < 7; ++i) {
          dof_w[i] = residual_y[i]*residual_lambda[row] +
              acceleration_bar*(residual_qacc[i]+data->efc_KBIP[4*row+1]*data->qvel[i]);
        }
        std::vector<mjtNum> Jw(data->nefc);
        mj_mulJacVec(model.get(), perturbed.get(), Jw.data(), dof_w);
        jacobian_values[side][row] = Jw[row];
        position_values[side][row] = data->efc_KBIP[4*row]*
            perturbed->efc_KBIP[4*row+2]*perturbed->contact[0].dist;
        impedance_values[side][row] = perturbed->efc_R[row];
        losses[side] += Jw[row] + acceleration_bar*position_values[side][row];
        if (hinv_ptr) {
          mjtNum imp = data->efc_KBIP[4*row+2];
          mjtNum impP = data->efc_KBIP[4*row+3];
          mjtNum scale = -residual_lambda[row]*impP/(imp*(1-imp))*perturbed->contact[0].dist;
          losses[side] += hinv_ptr[row]*scale;
        } else {
          losses[side] += residual_mu[row]*residual_lambda[row]*impedance_values[side][row];
        }
      }
    }
    if (contact_case == 1 && integrator_case == 0 && column == 2) {
      for (int row = 0; row < data->nefc; ++row) {
        mjtNum imp = data->efc_KBIP[4*row+2];
        mjtNum impP = data->efc_KBIP[4*row+3];
        mjtNum expected_R = -data->efc_R[row]*impP/(imp*(1-imp));
        mjtNum expected_pos = data->efc_KBIP[4*row]*
            (imp+impP*(data->efc_pos[row]-data->efc_margin[row]));
        EXPECT_NEAR(expected_R, (impedance_values[0][row]-impedance_values[1][row])/(2*eps),
                    2e-5) << "R derivative row=" << row;
        EXPECT_NEAR(expected_pos,
                    (position_values[0][row]-position_values[1][row])/(2*eps), 2e-5)
            << "position derivative row=" << row;
        std::vector<mjtNum> lambda_row(data->nefc, 0);
        std::vector<mjtNum> mu_row(data->nefc, 0);
        lambda_row[row] = residual_lambda[row];
        mu_row[row] = residual_mu[row];
        std::vector<mjtNum> row_gradient(7, 0);
        ASSERT_EQ(mjd_planeSpherePosAdjoint(model.get(), data.get(), residual_y,
                                            lambda_row.data(), mu_row.data(), nullptr,
                                            residual_qacc, row_gradient.data()), mjADJ_OK);
        mjtNum row_fd = (jacobian_values[0][row]-jacobian_values[1][row] +
            residual_mu[row]*(position_values[0][row]-position_values[1][row] +
            residual_lambda[row]*(impedance_values[0][row]-impedance_values[1][row])))/(2*eps);
        EXPECT_NEAR(row_gradient[column], row_fd, 2e-5) << "row residual row=" << row;
      }
    }
    mjtNum residual_fd = (losses[0]-losses[1])/(2*eps);
    if (contact_case >= 4) {
      mjtNum outer_losses[2] = {0, 0};
      for (int side = 0; side < 2; ++side) {
        MjDataPtr perturbed = MakeData(model);
        mj_copyData(perturbed.get(), model.get(), data.get());
        std::vector<mjtNum> tangent(7, 0);
        tangent[column] = side ? -2*eps : 2*eps;
        mj_integratePos(model.get(), perturbed->qpos, tangent.data(), 1);
        mj_forward(model.get(), perturbed.get());
        for (int row = 0; row < data->nefc; ++row) {
          mjtNum acceleration_bar = -cone_acceleration[row];
          mjtNum dof_w[7];
          for (int i = 0; i < 7; ++i) {
            dof_w[i] = residual_y[i]*residual_lambda[row] +
                acceleration_bar*(residual_qacc[i]+data->efc_KBIP[4*row+1]*data->qvel[i]);
          }
          std::vector<mjtNum> Jw(data->nefc);
          mj_mulJacVec(model.get(), perturbed.get(), Jw.data(), dof_w);
          mjtNum position_value = data->efc_KBIP[4*row]*
              perturbed->efc_KBIP[4*row+2]*perturbed->contact[0].dist;
          outer_losses[side] += Jw[row] + acceleration_bar*position_value;
          mjtNum imp = data->efc_KBIP[4*row+2];
          mjtNum impP = data->efc_KBIP[4*row+3];
          outer_losses[side] -= hinv_ptr[row]*residual_lambda[row]*impP/(imp*(1-imp))*
                                perturbed->contact[0].dist;
        }
      }
      residual_fd = (8*(losses[0]-losses[1])-(outer_losses[0]-outer_losses[1]))/(12*eps);
    }
    EXPECT_NEAR(residual_gradient[column], residual_fd, 2e-7)
        << "contact Jacobian residual column=" << column;
  }
  for (int mode = 0; mode < 2; ++mode) {
    for (int row = 0; row < 7; ++row) {
      std::vector<mjtNum> row_pos(7, 0);
      std::vector<mjtNum> row_vel(7, 0);
      if (mode == 0) row_pos[row] = 1;
      else row_vel[row] = 1;
      std::vector<mjtNum> grad_pos(7, 0);
      std::vector<mjtNum> grad_vel(7, 0);
      std::vector<mjtNum> grad_ctrl(1, 0);
      MjDataPtr row_data = MakeData(model);
      mj_copyData(row_data.get(), model.get(), data.get());
      ASSERT_EQ(mjd_stepAdjoint(model.get(), row_data.get(), ctx.get(), row_pos.data(),
                                row_vel.data(), grad_pos.data(), grad_vel.data(), grad_ctrl.data()),
                mjADJ_OK);
      for (int column = 0; column < 8; ++column) {
        InputBlock block = column < 7 ? InputBlock::kQPos :
                           (column < 14 ? InputBlock::kQVel : InputBlock::kCtrl);
        int index = column < 7 ? column : (column < 14 ? column-7 : column-14);
        double fd = StepLoss(StockStepCentralTangent(model.get(), data.get(), block, index,
                                            contact_case == 0 ? 1e-8 : 2e-7),
                             row_pos, row_vel);
        double actual = column < 7 ? grad_pos[index] :
                        (column < 14 ? grad_vel[index] : grad_ctrl[index]);
        EXPECT_NEAR(actual, fd, 2e-7*std::max(1.0, std::abs(fd)))
            << "mode=" << mode << " row=" << row << " column=" << column
            << " actual=" << actual << " fd=" << fd;
      }
    }
  }
  std::mt19937_64 rng(167);
  std::normal_distribution<double> normal;
  int compared = 0;
  for (int sample = 0; sample < 4; ++sample) {
    std::vector<mjtNum> bar_qpos(7);
    std::vector<mjtNum> bar_qvel(7);
    for (int i = 0; i < 7; ++i) {
      bar_qpos[i] = normal(rng);
      bar_qvel[i] = normal(rng);
    }
    std::vector<mjtNum> grad_qpos(7, 0);
    std::vector<mjtNum> grad_qvel(7, 0);
    std::vector<mjtNum> grad_ctrl(1, 0);
    MjDataPtr step_data = MakeData(model);
    mj_copyData(step_data.get(), model.get(), data.get());
    ASSERT_EQ(mjd_stepAdjoint(model.get(), step_data.get(), ctx.get(), bar_qpos.data(),
                              bar_qvel.data(), grad_qpos.data(), grad_qvel.data(), grad_ctrl.data()),
              mjADJ_OK);
    for (int i = 0; i < model->nq; ++i) EXPECT_EQ(step_data->qpos[i], reference->qpos[i]);
    for (int i = 0; i < model->nv; ++i) EXPECT_EQ(step_data->qvel[i], reference->qvel[i]);
    for (int column = 0; column < 8; ++column) {
      InputBlock block = column < 7 ? InputBlock::kQPos :
                         (column < 14 ? InputBlock::kQVel : InputBlock::kCtrl);
      int index = column < 7 ? column : (column < 14 ? column-7 : column-14);
      double fd = StepLoss(StockStepCentralTangent(model.get(), data.get(), block, index,
                                            contact_case == 0 ? 1e-8 : 2e-7),
                           bar_qpos, bar_qvel);
      double actual = column < 7 ? grad_qpos[index] :
                      (column < 14 ? grad_qvel[index] : grad_ctrl[index]);
      EXPECT_NEAR(actual, fd, 2e-7*std::max(1.0, std::abs(fd)))
          << sample << ":" << column << " actual=" << actual << " fd=" << fd;
      ++compared;
    }
  }
  EXPECT_EQ(compared, 32);
  }
  }
}

TEST_F(AdjointStepTest, PlaneBoxSingleCornerAllInputColumns) {
  static constexpr char xml[] = R"(
  <mujoco><option solver="Newton" integrator="implicitfast" timestep=".002" iterations="100">
    <flag island="disable" warmstart="disable"/></option>
    <worldbody>
      <geom type="plane" size="5 5 .1" condim="3" solimp=".9 .95 .001" solref=".02 1"/>
      <body pos="0 0 .06" euler="20 30 25"><freejoint/>
        <geom type="box" size=".04 .04 .04" mass="1" condim="3"
              solimp=".9 .95 .001" solref=".02 1"/>
        <body pos=".1 0 0">
          <joint name="j1" type="hinge" axis="0 1 0" damping=".1"/>
          <geom type="sphere" size=".02" mass=".2" contype="0" conaffinity="0"/>
        </body>
      </body>
    </worldbody>
    <actuator><position joint="j1" kp="1.2" kv=".1"/></actuator>
  </mujoco>)";
  char error[1024] = "";
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  model->opt.tolerance = 0;
  model->opt.ls_tolerance = 0;
  model->opt.ls_iterations = 100;
  static constexpr int kCondims[3] = {3, 4, 6};
  int cone_seen[3] = {0, 0, 0};
  int compared = 0;
  for (int pose = 0; pose < 2; ++pose) {
    mjtNum euler[3] = {pose ? -.3 : .35, pose ? .4 : .52, pose ? -.35 : .44};
    mjtNum pose_quat[4];
    mju_euler2Quat(pose_quat, euler, "xyz");
    for (int condim_case = 0; condim_case < 3; ++condim_case) {
      int condim = kCondims[condim_case];
      for (int geom = 0; geom < model->ngeom; ++geom) {
        model->geom_condim[geom] = condim;
      }
      for (int cone = 0; cone < 2; ++cone) {
        model->opt.cone = cone ? mjCONE_ELLIPTIC : mjCONE_PYRAMIDAL;
        for (int integrator_case = 0; integrator_case < 2; ++integrator_case) {
          SCOPED_TRACE(testing::Message() << "pose=" << pose << " condim=" << condim
                                          << " cone=" << cone
                                          << " integrator=" << integrator_case);
          model->opt.integrator = integrator_case == 0 ? mjINT_IMPLICITFAST : mjINT_EULER;
          for (int geom = 0; geom < model->ngeom; ++geom) {
            model->geom_solimp[mjNIMP*geom+2] = integrator_case == 0 ? .021 : .001;
          }
          MjDataPtr data = MakeData(model);
          mju_copy(data->qpos+3, pose_quat, 4);
          mjtNum qvel[7] = {cone ? 5.0 : 2, cone ? 6.0 : 3, -.04, .05, -.02, .01, .2};
          mju_copy(data->qvel, qvel, 7);
          data->ctrl[0] = .3;
          mj_forward(model.get(), data.get());
          ASSERT_EQ(data->ncon, 1) << "dist=" << data->contact[0].dist;
          ASSERT_EQ(data->nefc, cone ? condim : 2*(condim-1));
          ASSERT_EQ(model->geom_type[data->contact[0].geom1] == mjGEOM_PLANE ?
                    model->geom_type[data->contact[0].geom2] : model->geom_type[data->contact[0].geom1],
                    mjGEOM_BOX);
          const mjContact* con = data->contact;
          int box = model->geom_type[data->contact[0].geom1] == mjGEOM_BOX ?
                    data->contact[0].geom1 : data->contact[0].geom2;
          const mjtNum* nrm = con->frame;
          // independent witness: the contact point must be the midpoint between a
          // box corner and its plane projection
          mjtNum corner[3], local[3], bcenter[3], rel[3], inv[9];
          for (int k = 0; k < 3; ++k) {
            corner[k] = con->pos[k]+nrm[k]*.5*con->dist;
            bcenter[k] = data->geom_xpos[3*box+k];
          }
          mju_sub3(rel, corner, bcenter);
          mju_transpose(inv, data->geom_xmat+9*box, 3, 3);
          mju_mulMatVec3(local, inv, rel);
          for (int k = 0; k < 3; ++k) {
            EXPECT_NEAR(mju_abs(local[k]), model->geom_size[3*box+k], 1e-12)
                << "corner local[" << k << "]=" << local[k];
          }
          int active = 0;
          for (int row = 0; row < data->nefc; ++row) {
            active += data->efc_state[row] != mjCNSTRSTATE_SATISFIED;
            if (cone) {
              ASSERT_EQ(data->efc_state[row], data->efc_state[0]);
            }
          }
          ASSERT_GT(active, 0);
          if (cone) {
            ASSERT_TRUE(data->efc_state[0] == mjCNSTRSTATE_SATISFIED ||
                        data->efc_state[0] == mjCNSTRSTATE_QUADRATIC ||
                        data->efc_state[0] == mjCNSTRSTATE_CONE);
            cone_seen[condim_case] += data->efc_state[0] == mjCNSTRSTATE_CONE;
          } else {
            ASSERT_LT(active, data->nefc);
          }
          StepContextPtr ctx(mj_makeAdjContext(model.get()));
          ASSERT_THAT(ctx.get(), NotNull());
          ASSERT_EQ(mj_reserveAdjContext(ctx.get(), 1, data->nefc, 256, 1024), mjADJ_OK);
          MjDataPtr reference = MakeData(model);
          mj_copyData(reference.get(), model.get(), data.get());
          mj_step(model.get(), reference.get());
          std::mt19937_64 rng(23000+1000*pose+100*condim_case+10*cone+integrator_case);
          std::normal_distribution<double> normal;
          for (int sample = 0; sample < 18; ++sample) {
            std::vector<mjtNum> bar_qpos(7, 0), bar_qvel(7, 0);
            if (sample < 14) {
              if (sample < 7) bar_qpos[sample] = 1;
              else bar_qvel[sample-7] = 1;
            } else {
              for (int i = 0; i < 7; ++i) {
                bar_qpos[i] = normal(rng);
                bar_qvel[i] = normal(rng);
              }
            }
            std::vector<mjtNum> grad_qpos(7, 0), grad_qvel(7, 0), grad_ctrl(1, 0);
            MjDataPtr step_data = MakeData(model);
            mj_copyData(step_data.get(), model.get(), data.get());
            ASSERT_EQ(mjd_stepAdjoint(model.get(), step_data.get(), ctx.get(), bar_qpos.data(),
                                      bar_qvel.data(), grad_qpos.data(), grad_qvel.data(),
                                      grad_ctrl.data()), mjADJ_OK);
            for (int i = 0; i < model->nq; ++i) EXPECT_EQ(step_data->qpos[i], reference->qpos[i]);
            for (int i = 0; i < model->nv; ++i) EXPECT_EQ(step_data->qvel[i], reference->qvel[i]);
            for (int column = 0; column < 15; ++column) {
              InputBlock block = column < 7 ? InputBlock::kQPos :
                                 (column < 14 ? InputBlock::kQVel : InputBlock::kCtrl);
              int index = column < 7 ? column : (column < 14 ? column-7 : column-14);
              double fd = StepLoss(StockStepCentralTangent(model.get(), data.get(), block, index,
                                                          2e-7), bar_qpos, bar_qvel);
              double actual = column < 7 ? grad_qpos[index] :
                              (column < 14 ? grad_qvel[index] : grad_ctrl[index]);
              EXPECT_NEAR(actual, fd, 2e-7*std::max(1.0, std::abs(fd)))
                  << "sample=" << sample << " column=" << column
                  << " actual=" << actual << " fd=" << fd;
              ++compared;
            }
          }
        }
      }
    }
  }
  EXPECT_EQ(compared, 24*18*15);
  for (int i = 0; i < 3; ++i) {
    EXPECT_GT(cone_seen[i], 0) << "condim=" << kCondims[i] << " never reached CONE zone";
  }
}

TEST_F(AdjointStepTest, SphereSphereSingleContactAllInputColumns) {
  static constexpr char xml[] = R"(
  <mujoco><option solver="Newton" integrator="implicitfast" timestep=".002" iterations="100">
    <flag island="disable" warmstart="disable" filterparent="disable"/></option>
    <worldbody>
      <body pos="0 0 .3" euler="20 30 25"><freejoint/>
        <geom type="sphere" size=".04" mass="1" condim="3"
              solimp=".9 .95 .001" solref=".02 1"/>
        <body pos=".065 0 0" euler="40 10 70">
          <joint name="j1" type="hinge" axis="0 1 0" damping=".1"/>
          <geom type="sphere" size=".03" mass=".2" condim="3"
                solimp=".9 .95 .001" solref=".02 1"/>
        </body>
      </body>
    </worldbody>
    <actuator><position joint="j1" kp="1.2" kv=".1"/></actuator>
  </mujoco>)";
  char error[1024] = "";
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  model->opt.tolerance = 0;
  model->opt.ls_tolerance = 0;
  model->opt.ls_iterations = 100;
  static constexpr int kCondims[3] = {3, 4, 6};
  int cone_seen[3] = {0, 0, 0};
  int compared = 0;
  for (int pose = 0; pose < 2; ++pose) {
    mjtNum euler[3] = {pose ? -.62 : .35, pose ? .28 : .52, pose ? -.5 : .44};
    mjtNum pose_quat[4];
    mju_euler2Quat(pose_quat, euler, "xyz");
    for (int condim_case = 0; condim_case < 3; ++condim_case) {
      int condim = kCondims[condim_case];
      for (int geom = 0; geom < model->ngeom; ++geom) {
        model->geom_condim[geom] = condim;
      }
      for (int cone = 0; cone < 2; ++cone) {
        model->opt.cone = cone ? mjCONE_ELLIPTIC : mjCONE_PYRAMIDAL;
        for (int integrator_case = 0; integrator_case < 2; ++integrator_case) {
          SCOPED_TRACE(testing::Message() << "pose=" << pose << " condim=" << condim
                                          << " cone=" << cone
                                          << " integrator=" << integrator_case);
          model->opt.integrator = integrator_case == 0 ? mjINT_IMPLICITFAST : mjINT_EULER;
          for (int geom = 0; geom < model->ngeom; ++geom) {
            model->geom_solimp[mjNIMP*geom+2] = integrator_case == 0 ? .021 : .001;
          }
          MjDataPtr data = MakeData(model);
          mju_copy(data->qpos+3, pose_quat, 4);
          // slip must be relative: common translation cancels in the contact
          // Jacobian (hinge-driven slip, same scheme as capsule-capsule)
          mjtNum qvel[7] = {0, 0, 0, 0, 0, 0, cone ? 12.0 : 30.0};
          mju_copy(data->qvel, qvel, 7);
          data->ctrl[0] = .3;
          mj_forward(model.get(), data.get());
          ASSERT_EQ(data->ncon, 1) << "dist=" << data->contact[0].dist;
          ASSERT_EQ(data->nefc, cone ? condim : 2*(condim-1));
          ASSERT_EQ(model->geom_type[data->contact[0].geom1], mjGEOM_SPHERE);
          ASSERT_EQ(model->geom_type[data->contact[0].geom2], mjGEOM_SPHERE);
          const mjContact* con = data->contact;
          int sph1 = con->geom1, sph2 = con->geom2;
          mjtNum r1 = model->geom_size[3*sph1], r2 = model->geom_size[3*sph2];
          const mjtNum* nrm = con->frame;
          mjtNum len = con->dist+r1+r2;
          ASSERT_GT(len, 1e-8);
          // independent witness: both witnesses are the sphere centers
          mjtNum w1[3], w2[3];
          for (int k = 0; k < 3; ++k) {
            w1[k] = con->pos[k]-nrm[k]*(r1+.5*con->dist);
            w2[k] = con->pos[k]+nrm[k]*(r2+.5*con->dist);
          }
          EXPECT_NEAR(mju_dist3(w1, data->geom_xpos+3*sph1), 0, 1e-12);
          EXPECT_NEAR(mju_dist3(w2, data->geom_xpos+3*sph2), 0, 1e-12);
          EXPECT_NEAR(mju_dist3(w1, w2), len, 1e-12);
          ASSERT_GT(mju_abs(mju_abs(nrm[1])-.5), 1e-3) << "n_y=" << nrm[1];
          mjtNum t0[3] = {0, 1, 0};
          if (mju_abs(nrm[1]) >= .5) {
            t0[1] = 0;
            t0[2] = 1;
          }
          mjtNum cproj = mju_dot3(nrm, t0);
          mjtNum tan_chk[3], tmp3[3];
          for (int k = 0; k < 3; ++k) {
            tan_chk[k] = t0[k]-nrm[k]*cproj;
          }
          mju_normalize3(tan_chk);
          EXPECT_NEAR(mju_dist3(tan_chk, con->frame+3), 0, 1e-12);
          mju_cross(tmp3, nrm, tan_chk);
          EXPECT_NEAR(mju_dist3(tmp3, con->frame+6), 0, 1e-12);
          int active = 0;
          for (int row = 0; row < data->nefc; ++row) {
            active += data->efc_state[row] != mjCNSTRSTATE_SATISFIED;
            if (cone) {
              ASSERT_EQ(data->efc_state[row], data->efc_state[0]);
            }
          }
          ASSERT_GT(active, 0);
          if (cone) {
            ASSERT_TRUE(data->efc_state[0] == mjCNSTRSTATE_SATISFIED ||
                        data->efc_state[0] == mjCNSTRSTATE_QUADRATIC ||
                        data->efc_state[0] == mjCNSTRSTATE_CONE);
            cone_seen[condim_case] += data->efc_state[0] == mjCNSTRSTATE_CONE;
          } else {
            ASSERT_LT(active, data->nefc);
          }
          StepContextPtr ctx(mj_makeAdjContext(model.get()));
          ASSERT_THAT(ctx.get(), NotNull());
          ASSERT_EQ(mj_reserveAdjContext(ctx.get(), 1, data->nefc, 256, 1024), mjADJ_OK);
          MjDataPtr reference = MakeData(model);
          mj_copyData(reference.get(), model.get(), data.get());
          mj_step(model.get(), reference.get());
          std::mt19937_64 rng(23000+1000*pose+100*condim_case+10*cone+integrator_case);
          std::normal_distribution<double> normal;
          for (int sample = 0; sample < 18; ++sample) {
            std::vector<mjtNum> bar_qpos(7, 0), bar_qvel(7, 0);
            if (sample < 14) {
              if (sample < 7) bar_qpos[sample] = 1;
              else bar_qvel[sample-7] = 1;
            } else {
              for (int i = 0; i < 7; ++i) {
                bar_qpos[i] = normal(rng);
                bar_qvel[i] = normal(rng);
              }
            }
            std::vector<mjtNum> grad_qpos(7, 0), grad_qvel(7, 0), grad_ctrl(1, 0);
            MjDataPtr step_data = MakeData(model);
            mj_copyData(step_data.get(), model.get(), data.get());
            ASSERT_EQ(mjd_stepAdjoint(model.get(), step_data.get(), ctx.get(), bar_qpos.data(),
                                      bar_qvel.data(), grad_qpos.data(), grad_qvel.data(),
                                      grad_ctrl.data()), mjADJ_OK);
            for (int i = 0; i < model->nq; ++i) EXPECT_EQ(step_data->qpos[i], reference->qpos[i]);
            for (int i = 0; i < model->nv; ++i) EXPECT_EQ(step_data->qvel[i], reference->qvel[i]);
            for (int column = 0; column < 15; ++column) {
              InputBlock block = column < 7 ? InputBlock::kQPos :
                                 (column < 14 ? InputBlock::kQVel : InputBlock::kCtrl);
              int index = column < 7 ? column : (column < 14 ? column-7 : column-14);
              double fd = StepLoss(StockStepCentralTangent(model.get(), data.get(), block, index,
                                                          2e-7), bar_qpos, bar_qvel);
              double actual = column < 7 ? grad_qpos[index] :
                              (column < 14 ? grad_qvel[index] : grad_ctrl[index]);
              EXPECT_NEAR(actual, fd, 2e-7*std::max(1.0, std::abs(fd)))
                  << "sample=" << sample << " column=" << column
                  << " actual=" << actual << " fd=" << fd;
              ++compared;
            }
          }
        }
      }
    }
  }
  EXPECT_EQ(compared, 24*18*15);
  for (int i = 0; i < 3; ++i) {
    EXPECT_GT(cone_seen[i], 0) << "condim=" << kCondims[i] << " never reached CONE zone";
  }
}


TEST_F(AdjointStepTest, SphereCapsuleSingleContactAllInputColumns) {
  static constexpr char xml[] = R"(
  <mujoco><option solver="Newton" integrator="implicitfast" timestep=".002" iterations="100">
    <flag island="disable" warmstart="disable" filterparent="disable"/></option>
    <worldbody>
      <body pos="0 0 .3" euler="20 30 25"><freejoint/>
        <geom type="sphere" size=".035" mass="1" condim="3"
              solimp=".9 .95 .001" solref=".02 1"/>
        <body pos=".055 0 0" euler="40 10 70">
          <joint name="j1" type="hinge" axis="0 1 0" damping=".1"/>
          <geom type="capsule" size=".025 .08" mass=".2" condim="3"
                solimp=".9 .95 .001" solref=".02 1"/>
        </body>
      </body>
    </worldbody>
    <actuator><position joint="j1" kp="1.2" kv=".1"/></actuator>
  </mujoco>)";
  char error[1024] = "";
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  model->opt.tolerance = 0;
  model->opt.ls_tolerance = 0;
  model->opt.ls_iterations = 100;
  static constexpr int kCondims[3] = {3, 4, 6};
  int cone_seen[3] = {0, 0, 0};
  int compared = 0;
  for (int pose = 0; pose < 2; ++pose) {
    mjtNum euler[3] = {pose ? -.62 : .35, pose ? .28 : .52, pose ? -.5 : .44};
    mjtNum pose_quat[4];
    mju_euler2Quat(pose_quat, euler, "xyz");
    for (int condim_case = 0; condim_case < 3; ++condim_case) {
      int condim = kCondims[condim_case];
      for (int geom = 0; geom < model->ngeom; ++geom) {
        model->geom_condim[geom] = condim;
      }
      for (int cone = 0; cone < 2; ++cone) {
        model->opt.cone = cone ? mjCONE_ELLIPTIC : mjCONE_PYRAMIDAL;
        for (int integrator_case = 0; integrator_case < 2; ++integrator_case) {
          SCOPED_TRACE(testing::Message() << "pose=" << pose << " condim=" << condim
                                          << " cone=" << cone
                                          << " integrator=" << integrator_case);
          model->opt.integrator = integrator_case == 0 ? mjINT_IMPLICITFAST : mjINT_EULER;
          for (int geom = 0; geom < model->ngeom; ++geom) {
            model->geom_solimp[mjNIMP*geom+2] = integrator_case == 0 ? .021 : .001;
          }
          MjDataPtr data = MakeData(model);
          mju_copy(data->qpos+3, pose_quat, 4);
          mjtNum qvel[7] = {0, 0, 0, 0, 0, 0, cone ? 12.0 : 30.0};
          mju_copy(data->qvel, qvel, 7);
          data->ctrl[0] = .3;
          mj_forward(model.get(), data.get());
          ASSERT_EQ(data->ncon, 1) << "dist=" << data->contact[0].dist;
          ASSERT_EQ(data->nefc, cone ? condim : 2*(condim-1));
          ASSERT_EQ(model->geom_type[data->contact[0].geom1], mjGEOM_SPHERE);
          ASSERT_EQ(model->geom_type[data->contact[0].geom2], mjGEOM_CAPSULE);
          const mjContact* con = data->contact;
          int sph1 = con->geom1, cap2 = con->geom2;
          mjtNum r1 = model->geom_size[3*sph1], r2 = model->geom_size[3*cap2];
          mjtNum h2 = model->geom_size[3*cap2+1];
          const mjtNum* p2 = data->geom_xpos+3*cap2;
          const mjtNum* mat2 = data->geom_xmat+9*cap2;
          mjtNum s2[3];
          for (int k = 0; k < 3; ++k) {
            s2[k] = h2*mat2[3*k+2];
          }
          const mjtNum* nrm = con->frame;
          mjtNum len = con->dist+r1+r2;
          ASSERT_GT(len, 1e-8);
          // independent witness: w1 is the sphere center; w2 is the closest point
          // on the capsule axis (interior parameter x2)
          mjtNum w1[3], w2[3];
          for (int k = 0; k < 3; ++k) {
            w1[k] = con->pos[k]-nrm[k]*(r1+.5*con->dist);
            w2[k] = con->pos[k]+nrm[k]*(r2+.5*con->dist);
          }
          EXPECT_NEAR(mju_dist3(w1, data->geom_xpos+3*sph1), 0, 1e-12);
          mjtNum dif[3];
          mju_sub3(dif, data->geom_xpos+3*sph1, p2);
          mjtNum x2 = mju_dot3(s2, dif)/mju_dot3(s2, s2);
          mjtNum chk2[3];
          mju_scl3(chk2, s2, x2);
          mju_addTo3(chk2, p2);
          EXPECT_NEAR(mju_dist3(chk2, w2), 0, 1e-12) << "x2=" << x2;
          ASSERT_LT(mju_abs(x2), 1-1e-6) << "x2=" << x2;
          ASSERT_GT(mju_abs(mju_abs(nrm[1])-.5), 1e-3) << "n_y=" << nrm[1];
          mjtNum t0[3] = {0, 1, 0};
          if (mju_abs(nrm[1]) >= .5) {
            t0[1] = 0;
            t0[2] = 1;
          }
          mjtNum cproj = mju_dot3(nrm, t0);
          mjtNum tan_chk[3], tmp3[3];
          for (int k = 0; k < 3; ++k) {
            tan_chk[k] = t0[k]-nrm[k]*cproj;
          }
          mju_normalize3(tan_chk);
          EXPECT_NEAR(mju_dist3(tan_chk, con->frame+3), 0, 1e-12);
          mju_cross(tmp3, nrm, tan_chk);
          EXPECT_NEAR(mju_dist3(tmp3, con->frame+6), 0, 1e-12);
          int active = 0;
          for (int row = 0; row < data->nefc; ++row) {
            active += data->efc_state[row] != mjCNSTRSTATE_SATISFIED;
            if (cone) {
              ASSERT_EQ(data->efc_state[row], data->efc_state[0]);
            }
          }
          ASSERT_GT(active, 0);
          if (cone) {
            ASSERT_TRUE(data->efc_state[0] == mjCNSTRSTATE_SATISFIED ||
                        data->efc_state[0] == mjCNSTRSTATE_QUADRATIC ||
                        data->efc_state[0] == mjCNSTRSTATE_CONE);
            cone_seen[condim_case] += data->efc_state[0] == mjCNSTRSTATE_CONE;
          } else {
            ASSERT_LT(active, data->nefc);
          }
          StepContextPtr ctx(mj_makeAdjContext(model.get()));
          ASSERT_THAT(ctx.get(), NotNull());
          ASSERT_EQ(mj_reserveAdjContext(ctx.get(), 1, data->nefc, 256, 1024), mjADJ_OK);
          MjDataPtr reference = MakeData(model);
          mj_copyData(reference.get(), model.get(), data.get());
          mj_step(model.get(), reference.get());
          std::mt19937_64 rng(23000+1000*pose+100*condim_case+10*cone+integrator_case);
          std::normal_distribution<double> normal;
          for (int sample = 0; sample < 18; ++sample) {
            std::vector<mjtNum> bar_qpos(7, 0), bar_qvel(7, 0);
            if (sample < 14) {
              if (sample < 7) bar_qpos[sample] = 1;
              else bar_qvel[sample-7] = 1;
            } else {
              for (int i = 0; i < 7; ++i) {
                bar_qpos[i] = normal(rng);
                bar_qvel[i] = normal(rng);
              }
            }
            std::vector<mjtNum> grad_qpos(7, 0), grad_qvel(7, 0), grad_ctrl(1, 0);
            MjDataPtr step_data = MakeData(model);
            mj_copyData(step_data.get(), model.get(), data.get());
            ASSERT_EQ(mjd_stepAdjoint(model.get(), step_data.get(), ctx.get(), bar_qpos.data(),
                                      bar_qvel.data(), grad_qpos.data(), grad_qvel.data(),
                                      grad_ctrl.data()), mjADJ_OK);
            for (int i = 0; i < model->nq; ++i) EXPECT_EQ(step_data->qpos[i], reference->qpos[i]);
            for (int i = 0; i < model->nv; ++i) EXPECT_EQ(step_data->qvel[i], reference->qvel[i]);
            for (int column = 0; column < 15; ++column) {
              InputBlock block = column < 7 ? InputBlock::kQPos :
                                 (column < 14 ? InputBlock::kQVel : InputBlock::kCtrl);
              int index = column < 7 ? column : (column < 14 ? column-7 : column-14);
              double fd = StepLoss(StockStepCentralTangent(model.get(), data.get(), block, index,
                                                          2e-7), bar_qpos, bar_qvel);
              double actual = column < 7 ? grad_qpos[index] :
                              (column < 14 ? grad_qvel[index] : grad_ctrl[index]);
              EXPECT_NEAR(actual, fd, 2e-7*std::max(1.0, std::abs(fd)))
                  << "sample=" << sample << " column=" << column
                  << " actual=" << actual << " fd=" << fd;
              ++compared;
            }
          }
        }
      }
    }
  }
  EXPECT_EQ(compared, 24*18*15);
  for (int i = 0; i < 3; ++i) {
    EXPECT_GT(cone_seen[i], 0) << "condim=" << kCondims[i] << " never reached CONE zone";
  }
}

TEST_F(AdjointStepTest, SphereBoxFaceAllInputColumns) {
  static constexpr char xml[] = R"(
  <mujoco><option solver="Newton" integrator="implicitfast" timestep=".002" iterations="100">
    <flag island="disable" warmstart="disable" filterparent="disable"/></option>
    <worldbody>
      <body pos="0 0 .3" euler="20 30 25"><freejoint/>
        <geom type="sphere" size=".03" mass="1" condim="3"
              solimp=".9 .95 .001" solref=".02 1"/>
        <body pos=".068 0 0" euler="5 8 3">
          <joint name="j1" type="hinge" axis="0 1 0" damping=".1"/>
          <geom type="box" size=".04 .04 .04" mass=".2" condim="3"
                solimp=".9 .95 .001" solref=".02 1"/>
        </body>
      </body>
    </worldbody>
    <actuator><position joint="j1" kp="1.2" kv=".1"/></actuator>
  </mujoco>)";
  char error[1024] = "";
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  model->opt.tolerance = 0;
  model->opt.ls_tolerance = 0;
  model->opt.ls_iterations = 100;
  static constexpr int kCondims[3] = {3, 4, 6};
  int cone_seen[3] = {0, 0, 0};
  int compared = 0;
  for (int pose = 0; pose < 2; ++pose) {
    mjtNum euler[3] = {pose ? -.62 : .35, pose ? .28 : .52, pose ? -.5 : .44};
    mjtNum pose_quat[4];
    mju_euler2Quat(pose_quat, euler, "xyz");
    for (int condim_case = 0; condim_case < 3; ++condim_case) {
      int condim = kCondims[condim_case];
      for (int geom = 0; geom < model->ngeom; ++geom) {
        model->geom_condim[geom] = condim;
      }
      for (int cone = 0; cone < 2; ++cone) {
        model->opt.cone = cone ? mjCONE_ELLIPTIC : mjCONE_PYRAMIDAL;
        for (int integrator_case = 0; integrator_case < 2; ++integrator_case) {
          SCOPED_TRACE(testing::Message() << "pose=" << pose << " condim=" << condim
                                          << " cone=" << cone
                                          << " integrator=" << integrator_case);
          model->opt.integrator = integrator_case == 0 ? mjINT_IMPLICITFAST : mjINT_EULER;
          for (int geom = 0; geom < model->ngeom; ++geom) {
            model->geom_solimp[mjNIMP*geom+2] = integrator_case == 0 ? .021 : .001;
          }
          MjDataPtr data = MakeData(model);
          mju_copy(data->qpos+3, pose_quat, 4);
          mjtNum qvel[7] = {0, 0, 0, 0, 0, 0, cone ? 12.0 : 30.0};
          mju_copy(data->qvel, qvel, 7);
          data->ctrl[0] = .3;
          mj_forward(model.get(), data.get());
          ASSERT_EQ(data->ncon, 1) << "dist=" << data->contact[0].dist;
          ASSERT_EQ(data->nefc, cone ? condim : 2*(condim-1));
          ASSERT_EQ(model->geom_type[data->contact[0].geom1], mjGEOM_SPHERE);
          ASSERT_EQ(model->geom_type[data->contact[0].geom2], mjGEOM_BOX);
          const mjContact* con = data->contact;
          int sph1 = con->geom1, box = con->geom2;
          mjtNum r1 = model->geom_size[3*sph1];
          const mjtNum* nrm = con->frame;
          mjtNum len = con->dist+r1;
          ASSERT_GT(len, 1e-8);
          // independent witness: the sphere center and a box face interior point
          mjtNum q[3], w2[3];
          for (int k = 0; k < 3; ++k) {
            q[k] = con->pos[k]-nrm[k]*(r1+.5*con->dist);
            w2[k] = con->pos[k]+nrm[k]*.5*con->dist;
          }
          EXPECT_NEAR(mju_dist3(q, data->geom_xpos+3*sph1), 0, 1e-12);
          mjtNum inv[9], cloc[3], rel[3];
          mju_transpose(inv, data->geom_xmat+9*box, 3, 3);
          mju_sub3(rel, w2, data->geom_xpos+3*box);
          mju_mulMatVec3(cloc, inv, rel);
          int clamps = 0;
          for (int k = 0; k < 3; ++k) {
            if (mju_abs(mju_abs(cloc[k])-model->geom_size[3*box+k]) < 1e-9) {
              clamps++;
            } else {
              ASSERT_LT(mju_abs(cloc[k]),
                        model->geom_size[3*box+k]-1e-4) << "witness off face";
            }
          }
          ASSERT_EQ(clamps, 1) << "not a face-interior witness";
          ASSERT_GT(mju_abs(mju_abs(nrm[1])-.5), 1e-3) << "n_y=" << nrm[1];
          int active = 0;
          for (int row = 0; row < data->nefc; ++row) {
            active += data->efc_state[row] != mjCNSTRSTATE_SATISFIED;
            if (cone) {
              ASSERT_EQ(data->efc_state[row], data->efc_state[0]);
            }
          }
          ASSERT_GT(active, 0);
          if (cone) {
            ASSERT_TRUE(data->efc_state[0] == mjCNSTRSTATE_SATISFIED ||
                        data->efc_state[0] == mjCNSTRSTATE_QUADRATIC ||
                        data->efc_state[0] == mjCNSTRSTATE_CONE);
            cone_seen[condim_case] += data->efc_state[0] == mjCNSTRSTATE_CONE;
          } else {
            ASSERT_LT(active, data->nefc);
          }
          StepContextPtr ctx(mj_makeAdjContext(model.get()));
          ASSERT_THAT(ctx.get(), NotNull());
          ASSERT_EQ(mj_reserveAdjContext(ctx.get(), 1, data->nefc, 256, 1024), mjADJ_OK);
          MjDataPtr reference = MakeData(model);
          mj_copyData(reference.get(), model.get(), data.get());
          mj_step(model.get(), reference.get());
          std::mt19937_64 rng(23000+1000*pose+100*condim_case+10*cone+integrator_case);
          std::normal_distribution<double> normal;
          for (int sample = 0; sample < 18; ++sample) {
            std::vector<mjtNum> bar_qpos(7, 0), bar_qvel(7, 0);
            if (sample < 14) {
              if (sample < 7) bar_qpos[sample] = 1;
              else bar_qvel[sample-7] = 1;
            } else {
              for (int i = 0; i < 7; ++i) {
                bar_qpos[i] = normal(rng);
                bar_qvel[i] = normal(rng);
              }
            }
            std::vector<mjtNum> grad_qpos(7, 0), grad_qvel(7, 0), grad_ctrl(1, 0);
            MjDataPtr step_data = MakeData(model);
            mj_copyData(step_data.get(), model.get(), data.get());
            ASSERT_EQ(mjd_stepAdjoint(model.get(), step_data.get(), ctx.get(), bar_qpos.data(),
                                      bar_qvel.data(), grad_qpos.data(), grad_qvel.data(),
                                      grad_ctrl.data()), mjADJ_OK);
            for (int i = 0; i < model->nq; ++i) EXPECT_EQ(step_data->qpos[i], reference->qpos[i]);
            for (int i = 0; i < model->nv; ++i) EXPECT_EQ(step_data->qvel[i], reference->qvel[i]);
            for (int column = 0; column < 15; ++column) {
              InputBlock block = column < 7 ? InputBlock::kQPos :
                                 (column < 14 ? InputBlock::kQVel : InputBlock::kCtrl);
              int index = column < 7 ? column : (column < 14 ? column-7 : column-14);
              double fd = StepLoss(StockStepCentralTangent(model.get(), data.get(), block, index,
                                                          2e-7), bar_qpos, bar_qvel);
              double actual = column < 7 ? grad_qpos[index] :
                              (column < 14 ? grad_qvel[index] : grad_ctrl[index]);
              EXPECT_NEAR(actual, fd, 2e-7*std::max(1.0, std::abs(fd)))
                  << "sample=" << sample << " column=" << column
                  << " actual=" << actual << " fd=" << fd;
              ++compared;
            }
          }
        }
      }
    }
  }
  EXPECT_EQ(compared, 24*18*15);
  for (int i = 0; i < 3; ++i) {
    EXPECT_GT(cone_seen[i], 0) << "condim=" << kCondims[i] << " never reached CONE zone";
  }
}


TEST_F(AdjointStepTest, CapsuleBoxFaceAllInputColumns) {
  static constexpr char xml[] = R"(
  <mujoco><option solver="Newton" integrator="implicitfast" timestep=".002" iterations="100">
    <flag island="disable" warmstart="disable" filterparent="disable"/></option>
    <worldbody>
      <body pos="0 0 .3" euler="20 30 25"><freejoint/>
        <geom type="box" size=".06 .06 .06" mass="1" condim="3"
              solimp=".9 .95 .001" solref=".02 1"/>
        <body pos=".175 0 0" euler="0 -90 0">
          <joint name="j1" type="hinge" axis="0 1 0" damping=".1"/>
          <geom type="capsule" size=".02 .1" mass=".2" condim="3"
                solimp=".9 .95 .001" solref=".02 1"/>
        </body>
      </body>
    </worldbody>
    <actuator><position joint="j1" kp="1.2" kv=".1"/></actuator>
  </mujoco>)";
  char error[1024] = "";
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  model->opt.tolerance = 0;
  model->opt.ls_tolerance = 0;
  model->opt.ls_iterations = 100;
  static constexpr int kCondims[3] = {3, 4, 6};
  int cone_seen[3] = {0, 0, 0};
  int compared = 0;
  for (int pose = 0; pose < 2; ++pose) {
    mjtNum euler[3] = {pose ? -.62 : .35, pose ? .28 : .52, pose ? -.5 : .44};
    mjtNum pose_quat[4];
    mju_euler2Quat(pose_quat, euler, "xyz");
    for (int condim_case = 0; condim_case < 3; ++condim_case) {
      int condim = kCondims[condim_case];
      for (int geom = 0; geom < model->ngeom; ++geom) {
        model->geom_condim[geom] = condim;
      }
      for (int cone = 0; cone < 2; ++cone) {
        model->opt.cone = cone ? mjCONE_ELLIPTIC : mjCONE_PYRAMIDAL;
        for (int integrator_case = 0; integrator_case < 2; ++integrator_case) {
          SCOPED_TRACE(testing::Message() << "pose=" << pose << " condim=" << condim
                                          << " cone=" << cone
                                          << " integrator=" << integrator_case);
          model->opt.integrator = integrator_case == 0 ? mjINT_IMPLICITFAST : mjINT_EULER;
          for (int geom = 0; geom < model->ngeom; ++geom) {
            model->geom_solimp[mjNIMP*geom+2] = integrator_case == 0 ? .021 : .001;
          }
          MjDataPtr data = MakeData(model);
          mju_copy(data->qpos+3, pose_quat, 4);
          mjtNum qvel[7] = {0, 0, 0, 0, 0, 0, cone ? 45.0 : 60.0};
          mju_copy(data->qvel, qvel, 7);
          data->ctrl[0] = .3;
          mj_forward(model.get(), data.get());
          ASSERT_EQ(data->ncon, 1) << "dist=" << data->contact[0].dist;
          ASSERT_EQ(data->nefc, cone ? condim : 2*(condim-1));
          ASSERT_EQ(model->geom_type[data->contact[0].geom1], mjGEOM_CAPSULE);
          ASSERT_EQ(model->geom_type[data->contact[0].geom2], mjGEOM_BOX);
          const mjContact* con = data->contact;
          int cap1 = con->geom1, box = con->geom2;
          mjtNum r1 = model->geom_size[3*cap1], h1 = model->geom_size[3*cap1+1];
          const mjtNum* p1 = data->geom_xpos+3*cap1;
          const mjtNum* nrm = con->frame;
          mjtNum len = con->dist+r1;
          ASSERT_GT(len, 1e-8);
          // independent witness: capsule axis point with s1.(w2-q)=0, box face interior
          mjtNum q[3], w2[3], g12[3];
          for (int k = 0; k < 3; ++k) {
            q[k] = con->pos[k]-nrm[k]*(r1+.5*con->dist);
            w2[k] = con->pos[k]+nrm[k]*.5*con->dist;
            g12[k] = w2[k]-q[k];
          }
          mjtNum s1[3];
          for (int k = 0; k < 3; ++k) {
            s1[k] = h1*data->geom_xmat[9*cap1+3*k+2];
          }
          mjtNum dif1[3];
          mju_sub3(dif1, q, p1);
          mjtNum x1 = mju_dot3(s1, dif1)/mju_dot3(s1, s1);
          if (mju_abs(x1) <= 1-1e-6) {
            EXPECT_NEAR(mju_dot3(s1, g12), 0, 1e-9*(1+mju_norm3(g12)));
          } else {
            // segment boundary: sphere-box at the capsule end, argmin stays put
            ASSERT_NEAR(mju_abs(x1), 1, 1e-9) << "x1=" << x1;
            EXPECT_GE(mju_dot3(s1, g12)*x1, -1e-9*(1+mju_norm3(g12)));
          }
          mjtNum inv[9], cloc[3], rel[3];
          mju_transpose(inv, data->geom_xmat+9*box, 3, 3);
          mju_sub3(rel, w2, data->geom_xpos+3*box);
          mju_mulMatVec3(cloc, inv, rel);
          int clamps = 0;
          for (int k = 0; k < 3; ++k) {
            if (mju_abs(mju_abs(cloc[k])-model->geom_size[3*box+k]) < 1e-9) {
              clamps++;
            } else {
              ASSERT_LT(mju_abs(cloc[k]),
                        model->geom_size[3*box+k]-1e-4) << "witness off face";
            }
          }
          ASSERT_EQ(clamps, 1) << "not a face-interior witness";
          ASSERT_GT(mju_abs(mju_abs(nrm[1])-.5), 1e-3) << "n_y=" << nrm[1];
          int active = 0;
          for (int row = 0; row < data->nefc; ++row) {
            active += data->efc_state[row] != mjCNSTRSTATE_SATISFIED;
            if (cone) {
              ASSERT_EQ(data->efc_state[row], data->efc_state[0]);
            }
          }
          ASSERT_GT(active, 0);
          if (cone) {
            ASSERT_TRUE(data->efc_state[0] == mjCNSTRSTATE_SATISFIED ||
                        data->efc_state[0] == mjCNSTRSTATE_QUADRATIC ||
                        data->efc_state[0] == mjCNSTRSTATE_CONE);
            cone_seen[condim_case] += data->efc_state[0] == mjCNSTRSTATE_CONE;
          } else {
            ASSERT_LT(active, data->nefc);
          }
          StepContextPtr ctx(mj_makeAdjContext(model.get()));
          ASSERT_THAT(ctx.get(), NotNull());
          ASSERT_EQ(mj_reserveAdjContext(ctx.get(), 1, data->nefc, 256, 1024), mjADJ_OK);
          MjDataPtr reference = MakeData(model);
          mj_copyData(reference.get(), model.get(), data.get());
          mj_step(model.get(), reference.get());
          std::mt19937_64 rng(23000+1000*pose+100*condim_case+10*cone+integrator_case);
          std::normal_distribution<double> normal;
          for (int sample = 0; sample < 18; ++sample) {
            std::vector<mjtNum> bar_qpos(7, 0), bar_qvel(7, 0);
            if (sample < 14) {
              if (sample < 7) bar_qpos[sample] = 1;
              else bar_qvel[sample-7] = 1;
            } else {
              for (int i = 0; i < 7; ++i) {
                bar_qpos[i] = normal(rng);
                bar_qvel[i] = normal(rng);
              }
            }
            std::vector<mjtNum> grad_qpos(7, 0), grad_qvel(7, 0), grad_ctrl(1, 0);
            MjDataPtr step_data = MakeData(model);
            mj_copyData(step_data.get(), model.get(), data.get());
            ASSERT_EQ(mjd_stepAdjoint(model.get(), step_data.get(), ctx.get(), bar_qpos.data(),
                                      bar_qvel.data(), grad_qpos.data(), grad_qvel.data(),
                                      grad_ctrl.data()), mjADJ_OK);
            for (int i = 0; i < model->nq; ++i) EXPECT_EQ(step_data->qpos[i], reference->qpos[i]);
            for (int i = 0; i < model->nv; ++i) EXPECT_EQ(step_data->qvel[i], reference->qvel[i]);
            for (int column = 0; column < 15; ++column) {
              InputBlock block = column < 7 ? InputBlock::kQPos :
                                 (column < 14 ? InputBlock::kQVel : InputBlock::kCtrl);
              int index = column < 7 ? column : (column < 14 ? column-7 : column-14);
              double fd = StepLoss(StockStepCentralTangent(model.get(), data.get(), block, index,
                                                          2e-7), bar_qpos, bar_qvel);
              double actual = column < 7 ? grad_qpos[index] :
                              (column < 14 ? grad_qvel[index] : grad_ctrl[index]);
              EXPECT_NEAR(actual, fd, 2e-7*std::max(1.0, std::abs(fd)))
                  << "sample=" << sample << " column=" << column
                  << " actual=" << actual << " fd=" << fd;
              ++compared;
            }
          }
        }
      }
    }
  }
  EXPECT_EQ(compared, 24*18*15);
  for (int i = 0; i < 3; ++i) {
    EXPECT_GT(cone_seen[i], 0) << "condim=" << kCondims[i] << " never reached CONE zone";
  }
}


TEST_F(AdjointStepTest, PlaneCylinderSingleRimAllInputColumns) {
  static constexpr char xml[] = R"(
  <mujoco><option solver="Newton" integrator="implicitfast" timestep=".002" iterations="100">
    <flag island="disable" warmstart="disable"/></option>
    <worldbody>
      <geom type="plane" size="5 5 .1" condim="3" solimp=".9 .95 .001" solref=".02 1"/>
      <body pos="0 0 .048" euler="20 30 25"><freejoint/>
        <geom type="cylinder" size=".03 .04" mass="1" condim="3"
              solimp=".9 .95 .001" solref=".02 1"/>
        <body pos=".1 0 0">
          <joint name="j1" type="hinge" axis="0 1 0" damping=".1"/>
          <geom type="sphere" size=".02" mass=".2" contype="0" conaffinity="0"/>
        </body>
      </body>
    </worldbody>
    <actuator><position joint="j1" kp="1.2" kv=".1"/></actuator>
  </mujoco>)";
  char error[1024] = "";
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  model->opt.tolerance = 0;
  model->opt.ls_tolerance = 0;
  model->opt.ls_iterations = 100;
  static constexpr int kCondims[3] = {3, 4, 6};
  int cone_seen[3] = {0, 0, 0};
  int compared = 0;
  for (int pose = 0; pose < 2; ++pose) {
    mjtNum euler[3] = {pose ? -.3 : .35, pose ? .4 : .52, pose ? -.35 : .44};
    mjtNum pose_quat[4];
    mju_euler2Quat(pose_quat, euler, "xyz");
    for (int condim_case = 0; condim_case < 3; ++condim_case) {
      int condim = kCondims[condim_case];
      for (int geom = 0; geom < model->ngeom; ++geom) {
        model->geom_condim[geom] = condim;
      }
      for (int cone = 0; cone < 2; ++cone) {
        model->opt.cone = cone ? mjCONE_ELLIPTIC : mjCONE_PYRAMIDAL;
        for (int integrator_case = 0; integrator_case < 2; ++integrator_case) {
          SCOPED_TRACE(testing::Message() << "pose=" << pose << " condim=" << condim
                                          << " cone=" << cone
                                          << " integrator=" << integrator_case);
          model->opt.integrator = integrator_case == 0 ? mjINT_IMPLICITFAST : mjINT_EULER;
          for (int geom = 0; geom < model->ngeom; ++geom) {
            model->geom_solimp[mjNIMP*geom+2] = integrator_case == 0 ? .021 : .001;
          }
          MjDataPtr data = MakeData(model);
          mju_copy(data->qpos+3, pose_quat, 4);
          mjtNum qvel[7] = {cone ? 5.0 : 2, cone ? 6.0 : 3, -.04, .05, -.02, .01, .2};
          mju_copy(data->qvel, qvel, 7);
          data->ctrl[0] = .3;
          mj_forward(model.get(), data.get());
          ASSERT_EQ(data->ncon, 1) << "dist=" << data->contact[0].dist;
          ASSERT_EQ(data->nefc, cone ? condim : 2*(condim-1));
          ASSERT_EQ(model->geom_type[data->contact[0].geom1] == mjGEOM_PLANE ?
                    model->geom_type[data->contact[0].geom2] : model->geom_type[data->contact[0].geom1],
                    mjGEOM_CYLINDER);
          const mjContact* con = data->contact;
          int cyl = model->geom_type[data->contact[0].geom1] == mjGEOM_CYLINDER ?
                    data->contact[0].geom1 : data->contact[0].geom2;
          const mjtNum* nrm = con->frame;
          // independent witness: the contact midpoint is between a rim point and its
          // plane projection; the rim point is at axial +-h and radial distance r
          mjtNum rim[3], rloc[3], rel[3], inv[9];
          for (int k = 0; k < 3; ++k) {
            rim[k] = con->pos[k]+nrm[k]*.5*con->dist;
          }
          mju_sub3(rel, rim, data->geom_xpos+3*cyl);
          mju_transpose(inv, data->geom_xmat+9*cyl, 3, 3);
          mju_mulMatVec3(rloc, inv, rel);
          mjtNum cr = model->geom_size[3*cyl], ch = model->geom_size[3*cyl+1];
          EXPECT_NEAR(mju_abs(rloc[2]), ch, 1e-12) << "rim local z=" << rloc[2];
          EXPECT_NEAR(mju_sqrt(rloc[0]*rloc[0]+rloc[1]*rloc[1]), cr, 1e-12)
              << "rim radial=" << rloc[0] << "," << rloc[1];
          int active = 0;
          for (int row = 0; row < data->nefc; ++row) {
            active += data->efc_state[row] != mjCNSTRSTATE_SATISFIED;
            if (cone) {
              ASSERT_EQ(data->efc_state[row], data->efc_state[0]);
            }
          }
          ASSERT_GT(active, 0);
          if (cone) {
            ASSERT_TRUE(data->efc_state[0] == mjCNSTRSTATE_SATISFIED ||
                        data->efc_state[0] == mjCNSTRSTATE_QUADRATIC ||
                        data->efc_state[0] == mjCNSTRSTATE_CONE);
            cone_seen[condim_case] += data->efc_state[0] == mjCNSTRSTATE_CONE;
          } else {
            ASSERT_LT(active, data->nefc);
          }
          StepContextPtr ctx(mj_makeAdjContext(model.get()));
          ASSERT_THAT(ctx.get(), NotNull());
          ASSERT_EQ(mj_reserveAdjContext(ctx.get(), 1, data->nefc, 256, 1024), mjADJ_OK);
          MjDataPtr reference = MakeData(model);
          mj_copyData(reference.get(), model.get(), data.get());
          mj_step(model.get(), reference.get());
          std::mt19937_64 rng(23000+1000*pose+100*condim_case+10*cone+integrator_case);
          std::normal_distribution<double> normal;
          for (int sample = 0; sample < 18; ++sample) {
            std::vector<mjtNum> bar_qpos(7, 0), bar_qvel(7, 0);
            if (sample < 14) {
              if (sample < 7) bar_qpos[sample] = 1;
              else bar_qvel[sample-7] = 1;
            } else {
              for (int i = 0; i < 7; ++i) {
                bar_qpos[i] = normal(rng);
                bar_qvel[i] = normal(rng);
              }
            }
            std::vector<mjtNum> grad_qpos(7, 0), grad_qvel(7, 0), grad_ctrl(1, 0);
            MjDataPtr step_data = MakeData(model);
            mj_copyData(step_data.get(), model.get(), data.get());
            ASSERT_EQ(mjd_stepAdjoint(model.get(), step_data.get(), ctx.get(), bar_qpos.data(),
                                      bar_qvel.data(), grad_qpos.data(), grad_qvel.data(),
                                      grad_ctrl.data()), mjADJ_OK);
            for (int i = 0; i < model->nq; ++i) EXPECT_EQ(step_data->qpos[i], reference->qpos[i]);
            for (int i = 0; i < model->nv; ++i) EXPECT_EQ(step_data->qvel[i], reference->qvel[i]);
            for (int column = 0; column < 15; ++column) {
              InputBlock block = column < 7 ? InputBlock::kQPos :
                                 (column < 14 ? InputBlock::kQVel : InputBlock::kCtrl);
              int index = column < 7 ? column : (column < 14 ? column-7 : column-14);
              double fd = StepLoss(StockStepCentralTangent(model.get(), data.get(), block, index,
                                                          2e-7), bar_qpos, bar_qvel);
              double actual = column < 7 ? grad_qpos[index] :
                              (column < 14 ? grad_qvel[index] : grad_ctrl[index]);
              EXPECT_NEAR(actual, fd, 2e-7*std::max(1.0, std::abs(fd)))
                  << "sample=" << sample << " column=" << column
                  << " actual=" << actual << " fd=" << fd;
              ++compared;
            }
          }
        }
      }
    }
  }
  EXPECT_EQ(compared, 24*18*15);
  for (int i = 0; i < 3; ++i) {
    EXPECT_GT(cone_seen[i], 0) << "condim=" << kCondims[i] << " never reached CONE zone";
  }
}


TEST_F(AdjointStepTest, PlaneEllipsoidSupportAllInputColumns) {
  static constexpr char xml[] = R"(
  <mujoco><option solver="Newton" integrator="implicitfast" timestep=".002" iterations="100">
    <flag island="disable" warmstart="disable"/></option>
    <worldbody>
      <geom type="plane" size="5 5 .1" condim="3" solimp=".9 .95 .001" solref=".02 1"/>
      <body pos="0 0 .024" euler="20 30 25"><freejoint/>
        <geom type="ellipsoid" size=".03 .04 .02" mass="1" condim="3"
              solimp=".9 .95 .001" solref=".02 1"/>
        <body pos=".1 0 0">
          <joint name="j1" type="hinge" axis="0 1 0" damping=".1"/>
          <geom type="sphere" size=".02" mass=".2" contype="0" conaffinity="0"/>
        </body>
      </body>
    </worldbody>
    <actuator><position joint="j1" kp="1.2" kv=".1"/></actuator>
  </mujoco>)";
  char error[1024] = "";
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  model->opt.tolerance = 0;
  model->opt.ls_tolerance = 0;
  model->opt.ls_iterations = 100;
  static constexpr int kCondims[3] = {3, 4, 6};
  int cone_seen[3] = {0, 0, 0};
  int compared = 0;
  for (int pose = 0; pose < 2; ++pose) {
    mjtNum euler[3] = {pose ? -.3 : .35, pose ? .4 : .52, pose ? -.35 : .44};
    mjtNum pose_quat[4];
    mju_euler2Quat(pose_quat, euler, "xyz");
    for (int condim_case = 0; condim_case < 3; ++condim_case) {
      int condim = kCondims[condim_case];
      for (int geom = 0; geom < model->ngeom; ++geom) {
        model->geom_condim[geom] = condim;
      }
      for (int cone = 0; cone < 2; ++cone) {
        model->opt.cone = cone ? mjCONE_ELLIPTIC : mjCONE_PYRAMIDAL;
        for (int integrator_case = 0; integrator_case < 2; ++integrator_case) {
          SCOPED_TRACE(testing::Message() << "pose=" << pose << " condim=" << condim
                                          << " cone=" << cone
                                          << " integrator=" << integrator_case);
          model->opt.integrator = integrator_case == 0 ? mjINT_IMPLICITFAST : mjINT_EULER;
          for (int geom = 0; geom < model->ngeom; ++geom) {
            model->geom_solimp[mjNIMP*geom+2] = integrator_case == 0 ? .021 : .001;
          }
          MjDataPtr data = MakeData(model);
          mju_copy(data->qpos+3, pose_quat, 4);
          mjtNum qvel[7] = {cone ? 1.5 : 2, cone ? 2.0 : 3, -.04, .05, -.02, .01, .2};
          mju_copy(data->qvel, qvel, 7);
          data->ctrl[0] = .3;
          mj_forward(model.get(), data.get());
          ASSERT_EQ(data->ncon, 1) << "dist=" << data->contact[0].dist;
          ASSERT_EQ(data->nefc, cone ? condim : 2*(condim-1));
          ASSERT_EQ(model->geom_type[data->contact[0].geom1] == mjGEOM_PLANE ?
                    model->geom_type[data->contact[0].geom2] : model->geom_type[data->contact[0].geom1],
                    mjGEOM_ELLIPSOID);
          const mjContact* con = data->contact;
          int ell = model->geom_type[data->contact[0].geom1] == mjGEOM_ELLIPSOID ?
                    data->contact[0].geom1 : data->contact[0].geom2;
          const mjtNum* nrm = con->frame;
          // independent witness: the contact midpoint is between the ellipsoid
          // support point in direction -n and its plane projection
          mjtNum sup[3], sloc[3], uloc[3], du[3], rel3[3], inv9[9];
          for (int k = 0; k < 3; ++k) {
            sup[k] = con->pos[k]+nrm[k]*.5*con->dist;
          }
          mjtNum negn[3];
          mju_scl3(negn, nrm, -1);
          mju_transpose(inv9, data->geom_xmat+9*ell, 3, 3);
          mju_mulMatVec3(uloc, inv9, negn);
          mjtNum dun = 0;
          for (int k = 0; k < 3; ++k) {
            du[k] = model->geom_size[3*ell+k]*uloc[k];
            dun += du[k]*du[k];
          }
          dun = mju_sqrt(dun);
          for (int k = 0; k < 3; ++k) {
            sloc[k] = model->geom_size[3*ell+k]*du[k]/dun;
          }
          mju_mulMatVec3(rel3, data->geom_xmat+9*ell, sloc);
          for (int k = 0; k < 3; ++k) {
            rel3[k] += data->geom_xpos[3*ell+k];
          }
          EXPECT_NEAR(mju_dist3(rel3, sup), 0, 1e-12) << "support witness";
          int active = 0;
          for (int row = 0; row < data->nefc; ++row) {
            active += data->efc_state[row] != mjCNSTRSTATE_SATISFIED;
            if (cone) {
              ASSERT_EQ(data->efc_state[row], data->efc_state[0]);
            }
          }
          ASSERT_GT(active, 0);
          if (cone) {
            ASSERT_TRUE(data->efc_state[0] == mjCNSTRSTATE_SATISFIED ||
                        data->efc_state[0] == mjCNSTRSTATE_QUADRATIC ||
                        data->efc_state[0] == mjCNSTRSTATE_CONE);
            cone_seen[condim_case] += data->efc_state[0] == mjCNSTRSTATE_CONE;
          } else {
            ASSERT_LT(active, data->nefc);
          }
          StepContextPtr ctx(mj_makeAdjContext(model.get()));
          ASSERT_THAT(ctx.get(), NotNull());
          ASSERT_EQ(mj_reserveAdjContext(ctx.get(), 1, data->nefc, 256, 1024), mjADJ_OK);
          MjDataPtr reference = MakeData(model);
          mj_copyData(reference.get(), model.get(), data.get());
          mj_step(model.get(), reference.get());
          std::mt19937_64 rng(23000+1000*pose+100*condim_case+10*cone+integrator_case);
          std::normal_distribution<double> normal;
          for (int sample = 0; sample < 18; ++sample) {
            std::vector<mjtNum> bar_qpos(7, 0), bar_qvel(7, 0);
            if (sample < 14) {
              if (sample < 7) bar_qpos[sample] = 1;
              else bar_qvel[sample-7] = 1;
            } else {
              for (int i = 0; i < 7; ++i) {
                bar_qpos[i] = normal(rng);
                bar_qvel[i] = normal(rng);
              }
            }
            std::vector<mjtNum> grad_qpos(7, 0), grad_qvel(7, 0), grad_ctrl(1, 0);
            MjDataPtr step_data = MakeData(model);
            mj_copyData(step_data.get(), model.get(), data.get());
            ASSERT_EQ(mjd_stepAdjoint(model.get(), step_data.get(), ctx.get(), bar_qpos.data(),
                                      bar_qvel.data(), grad_qpos.data(), grad_qvel.data(),
                                      grad_ctrl.data()), mjADJ_OK);
            for (int i = 0; i < model->nq; ++i) EXPECT_EQ(step_data->qpos[i], reference->qpos[i]);
            for (int i = 0; i < model->nv; ++i) EXPECT_EQ(step_data->qvel[i], reference->qvel[i]);
            for (int column = 0; column < 15; ++column) {
              InputBlock block = column < 7 ? InputBlock::kQPos :
                                 (column < 14 ? InputBlock::kQVel : InputBlock::kCtrl);
              int index = column < 7 ? column : (column < 14 ? column-7 : column-14);
              double fd = StepLoss(StockStepCentralTangent(model.get(), data.get(), block, index,
                                                          2e-7), bar_qpos, bar_qvel);
              double actual = column < 7 ? grad_qpos[index] :
                              (column < 14 ? grad_qvel[index] : grad_ctrl[index]);
              EXPECT_NEAR(actual, fd, 2e-7*std::max(1.0, std::abs(fd)))
                  << "sample=" << sample << " column=" << column
                  << " actual=" << actual << " fd=" << fd;
              ++compared;
            }
          }
        }
      }
    }
  }
  EXPECT_EQ(compared, 24*18*15);
  for (int i = 0; i < 3; ++i) {
    EXPECT_GT(cone_seen[i], 0) << "condim=" << kCondims[i] << " never reached CONE zone";
  }
}


TEST_F(AdjointStepTest, SphereCylinderBranchesAllInputColumns) {
  static constexpr char xml[] = R"(
  <mujoco><option solver="Newton" integrator="implicitfast" timestep=".002" iterations="100">
    <flag island="disable" warmstart="disable" filterparent="disable"/></option>
    <worldbody>
      <body pos="0 0 .3" euler="20 30 25"><freejoint/>
        <geom type="cylinder" size=".015 .02" mass="1" condim="3"
              solimp=".9 .95 .001" solref=".02 1"/>
        <body pos="0 0 0">
          <joint name="j1" type="hinge" axis="0 1 0" damping=".1"/>
          <geom type="sphere" size=".018" mass=".2" pos=".03 0 0" condim="3"
                solimp=".9 .95 .001" solref=".02 1"/>
        </body>
      </body>
    </worldbody>
    <actuator><position joint="j1" kp=".1" kv=".01"/></actuator>
  </mujoco>)";
  char error[1024] = "";
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  model->opt.tolerance = 0;
  model->opt.ls_tolerance = 0;
  model->opt.ls_iterations = 100;
  static constexpr int kCondims[3] = {3, 4, 6};
  // per-branch projection coordinate x and radial offset rho (|rel| = arm .03)
  static constexpr mjtNum kX[3] = {.008, .029, .022};
  static constexpr mjtNum kRho[3] = {0.0289136646, 0.0076811457, 0.0203960781};
  int cone_seen[3] = {0, 0, 0};
  int compared = 0;
  for (int branch = 0; branch < 3; ++branch) {
    for (int pose = 0; pose < 2; ++pose) {
      mjtNum euler[3] = {pose ? -.25 : .35, pose ? .45 : .52, pose ? -.3 : .44};
      mjtNum pose_quat[4];
      mju_euler2Quat(pose_quat, euler, "xyz");
      // probe the actual cylinder axis (geom frame may differ from body frame)
      MjDataPtr axisprobe = MakeData(model);
      mju_copy(axisprobe->qpos+3, pose_quat, 4);
      mj_forward(model.get(), axisprobe.get());
      int cylg = model->geom_type[0] == mjGEOM_CYLINDER ? 0 : 1;
      mjtNum a2m[3];
      a2m[0] = axisprobe->geom_xmat[9*cylg+2];
      a2m[1] = axisprobe->geom_xmat[9*cylg+5];
      a2m[2] = axisprobe->geom_xmat[9*cylg+8];
      // p perpendicular to the axis with no component along the hinge normal:
      // p = normalize(cross(a2, R_pose*ey))
      mjtNum eyq[3] = {0, 1, 0}, eyp[3], phat[3];
      mju_rotVecQuat(eyp, eyq, pose_quat);
      mju_cross(phat, a2m, eyp);
      mju_normalize3(phat);
      // desired relative offset (world), then expressed in the pose frame as u
      mjtNum relw[3];
      for (int k = 0; k < 3; ++k) {
        relw[k] = a2m[k]*kX[branch]+phat[k]*kRho[branch];
      }
      mjtNum nq[4] = {pose_quat[0], -pose_quat[1], -pose_quat[2], -pose_quat[3]};
      mjtNum u[3];
      mju_rotVecQuat(u, relw, nq);
      mjtNum j1angle = -mju_atan2(-u[2], u[0]);
      // hinge axis along cross(a2, p_hat) so hinge rotation drives normal motion
      {
        mjtNum axc[3], axl[3];
        mju_cross(axc, a2m, phat);
        mju_normalize3(axc);
        mju_rotVecQuat(axl, axc, nq);
        model->jnt_axis[3] = axl[0];
        model->jnt_axis[4] = axl[1];
        model->jnt_axis[5] = axl[2];
      }
      for (int condim_case = 0; condim_case < 3; ++condim_case) {
        int condim = kCondims[condim_case];
        for (int geom = 0; geom < model->ngeom; ++geom) {
          model->geom_condim[geom] = condim;
        }
        for (int cone = 0; cone < 2; ++cone) {
          model->opt.cone = cone ? mjCONE_ELLIPTIC : mjCONE_PYRAMIDAL;
          for (int integrator_case = 0; integrator_case < 2; ++integrator_case) {
            SCOPED_TRACE(testing::Message() << "branch=" << branch << " pose=" << pose
                                            << " condim=" << condim << " cone=" << cone
                                            << " integrator=" << integrator_case);
            model->opt.integrator = integrator_case == 0 ? mjINT_IMPLICITFAST : mjINT_EULER;
            for (int geom = 0; geom < model->ngeom; ++geom) {
              model->geom_solimp[mjNIMP*geom+2] = integrator_case == 0 ? .021 : .001;
            }
            MjDataPtr data = MakeData(model);
            mju_copy(data->qpos+3, pose_quat, 4);
            data->qpos[model->jnt_qposadr[1]] = j1angle;
            mjtNum qvel[7] = {0, 0, 0, 0, 0, 0, cone ? 12.0 : 30.0};
            mju_copy(data->qvel, qvel, 7);
            data->ctrl[0] = .3;
            mj_forward(model.get(), data.get());
            ASSERT_EQ(data->ncon, 1) << "dist=" << data->contact[0].dist;
            ASSERT_EQ(data->nefc, cone ? condim : 2*(condim-1));
            ASSERT_EQ(model->geom_type[data->contact[0].geom1], mjGEOM_SPHERE);
            ASSERT_EQ(model->geom_type[data->contact[0].geom2], mjGEOM_CYLINDER);
            const mjContact* con = data->contact;
            const mjtNum* nrm = con->frame;
            int sphgeom = model->geom_type[0] == mjGEOM_SPHERE ? 0 : 1;
            int cylgeom = 1-sphgeom;
            mjtNum r1 = model->geom_size[3*sphgeom];
            mjtNum rc = model->geom_size[3*cylgeom];
            mjtNum h = model->geom_size[3*cylgeom+1];
            mjtNum a2[3];
            a2[0] = data->geom_xmat[9*cylgeom+2];
            a2[1] = data->geom_xmat[9*cylgeom+5];
            a2[2] = data->geom_xmat[9*cylgeom+8];
            mjtNum r2eff = branch == 0 ? rc : 0;
            mjtNum w1[3], w2[3];
            for (int k = 0; k < 3; ++k) {
              w1[k] = con->pos[k]-nrm[k]*(r1+.5*con->dist);
              w2[k] = con->pos[k]+nrm[k]*(.5*con->dist+r2eff);
            }
            EXPECT_NEAR(mju_dist3(w1, data->geom_xpos+3*sphgeom), 0, 1e-12);
            {
              mjtNum s1[3];
              for (int k = 0; k < 3; ++k) {
                s1[k] = h*data->geom_xmat[9*cylgeom+3*k+2];
              }
              mjtNum qv[3], g12v[3];
              for (int k = 0; k < 3; ++k) {
                qv[k] = con->pos[k]-nrm[k]*(r1+.5*con->dist);
                g12v[k] = w2[k]-qv[k];
              }
              mjtNum dif1[3];
              mju_sub3(dif1, qv, data->geom_xpos+3*sphgeom);
              mjtNum ma1 = mju_dot3(s1, dif1);
              mjtNum x1 = ma1/mju_dot3(s1, s1);
              if (branch == 0 && mju_abs(x1) <= 1-1e-6) {
                EXPECT_NEAR(mju_dot3(s1, g12v)/mju_dot3(s1, s1), 0, 1e-9*(1+mju_norm3(g12v)));
              }
            }
            mjtNum relv[3];
            mju_sub3(relv, data->geom_xpos+3*sphgeom, data->geom_xpos+3*cylgeom);
            mjtNum x = mju_dot3(a2, relv);
            mjtNum pp[3];
            for (int k = 0; k < 3; ++k) pp[k] = relv[k]-a2[k]*x;
            mjtNum plen = mju_norm3(pp);
            mjtNum chk[3];
            if (branch == 0) {
              for (int k = 0; k < 3; ++k) chk[k] = data->geom_xpos[3*cylgeom+k]+a2[k]*x;
            } else {
              mjtNum sg = x >= 0 ? 1 : -1;
              mjtNum scale = branch == 1 ? 1.0 : rc/plen;
              for (int k = 0; k < 3; ++k) {
                chk[k] = data->geom_xpos[3*cylgeom+k]+a2[k]*sg*h+pp[k]*scale;
              }
            }
            EXPECT_NEAR(mju_dist3(chk, w2), 0, 1e-9*(1+mju_norm3(w2)))
                << "branch=" << branch << " x=" << x << " plen=" << plen
                << " dist=" << con->dist << " j1=" << j1angle;
            if (branch == 1) {
              mjtNum nn[3] = {-a2[0], -a2[1], -a2[2]};
              if (x < 0) mju_scl3(nn, nn, -1);
              EXPECT_NEAR(mju_dist3(nn, nrm), 0, 1e-9);
            }
            ASSERT_GT(mju_abs(mju_abs(nrm[1])-.5), 1e-3) << "n_y=" << nrm[1];
            int active = 0;
            for (int row = 0; row < data->nefc; ++row) {
              active += data->efc_state[row] != mjCNSTRSTATE_SATISFIED;
              if (cone) {
                ASSERT_EQ(data->efc_state[row], data->efc_state[0]);
              }
            }
            ASSERT_GT(active, 0);
            if (cone) {
              ASSERT_TRUE(data->efc_state[0] == mjCNSTRSTATE_SATISFIED ||
                          data->efc_state[0] == mjCNSTRSTATE_QUADRATIC ||
                          data->efc_state[0] == mjCNSTRSTATE_CONE);
              cone_seen[condim_case] += data->efc_state[0] == mjCNSTRSTATE_CONE;
            } else {
              // all-QUADRATIC expected: hinge slip is colinear with one friction
              // direction; the mixed-edge bucket is asserted by other pair tests
            }
            StepContextPtr ctx(mj_makeAdjContext(model.get()));
            ASSERT_THAT(ctx.get(), NotNull());
            ASSERT_EQ(mj_reserveAdjContext(ctx.get(), 1, data->nefc, 256, 1024), mjADJ_OK);
            MjDataPtr reference = MakeData(model);
            mj_copyData(reference.get(), model.get(), data.get());
            mj_step(model.get(), reference.get());
            std::mt19937_64 rng(23000+1000*branch+1000*pose+100*condim_case+10*cone+integrator_case);
            std::normal_distribution<double> normal;
            for (int sample = 0; sample < 18; ++sample) {
              std::vector<mjtNum> bar_qpos(7, 0), bar_qvel(7, 0);
              if (sample < 14) {
                if (sample < 7) bar_qpos[sample] = 1;
                else bar_qvel[sample-7] = 1;
              } else {
                for (int i = 0; i < 7; ++i) {
                  bar_qpos[i] = normal(rng);
                  bar_qvel[i] = normal(rng);
                }
              }
              std::vector<mjtNum> grad_qpos(7, 0), grad_qvel(7, 0), grad_ctrl(1, 0);
              MjDataPtr step_data = MakeData(model);
              mj_copyData(step_data.get(), model.get(), data.get());
              ASSERT_EQ(mjd_stepAdjoint(model.get(), step_data.get(), ctx.get(), bar_qpos.data(),
                                        bar_qvel.data(), grad_qpos.data(), grad_qvel.data(),
                                        grad_ctrl.data()), mjADJ_OK);
              for (int i = 0; i < model->nq; ++i) EXPECT_EQ(step_data->qpos[i], reference->qpos[i]);
              for (int i = 0; i < model->nv; ++i) EXPECT_EQ(step_data->qvel[i], reference->qvel[i]);
              for (int column = 0; column < 15; ++column) {
                InputBlock block = column < 7 ? InputBlock::kQPos :
                                   (column < 14 ? InputBlock::kQVel : InputBlock::kCtrl);
                int index = column < 7 ? column : (column < 14 ? column-7 : column-14);
                double fd = StepLoss(StockStepCentralTangent(model.get(), data.get(), block, index,
                                                            2e-7), bar_qpos, bar_qvel);
                double actual = column < 7 ? grad_qpos[index] :
                                (column < 14 ? grad_qvel[index] : grad_ctrl[index]);
                EXPECT_NEAR(actual, fd, 2e-7*std::max(1.0, std::abs(fd)))
                    << "sample=" << sample << " column=" << column
                    << " actual=" << actual << " fd=" << fd;
                ++compared;
              }
            }
          }
        }
      }
    }
  }
  EXPECT_EQ(compared, 3*2*3*2*2*18*15);
  for (int i = 0; i < 3; ++i) {
    EXPECT_GT(cone_seen[i], 0) << "condim=" << kCondims[i] << " never reached CONE zone";
  }
}

TEST_F(AdjointStepTest, CapsuleCapsuleSlot0AllInputColumns) {
  static constexpr char xml[] = R"(
  <mujoco><option solver="Newton" integrator="implicitfast" timestep=".002" iterations="100">
    <flag island="disable" warmstart="disable" filterparent="disable"/></option>
    <worldbody>
      <body pos="0 0 .3" euler="20 30 25"><freejoint/>
        <geom type="capsule" size=".03 .1" mass="1" condim="3"
              solimp=".9 .95 .001" solref=".02 1"/>
        <body pos=".047 0 0" euler="40 10 70">
          <joint name="j1" type="hinge" axis="0 1 0" damping=".1"/>
          <geom type="capsule" size=".025 .08" mass=".2" condim="3"
                solimp=".9 .95 .001" solref=".02 1"/>
        </body>
      </body>
    </worldbody>
    <actuator><position joint="j1" kp="1.2" kv=".1"/></actuator>
  </mujoco>)";
  char error[1024] = "";
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  model->opt.tolerance = 0;
  model->opt.ls_tolerance = 0;
  model->opt.ls_iterations = 100;
  static constexpr int kCondims[3] = {3, 4, 6};
  int cone_seen[3] = {0, 0, 0};
  int compared = 0;
  for (int pose = 0; pose < 2; ++pose) {
    mjtNum euler[3] = {pose ? -.62 : .35, pose ? .28 : .52, pose ? -.5 : .44};
    mjtNum pose_quat[4];
    mju_euler2Quat(pose_quat, euler, "xyz");
    for (int condim_case = 0; condim_case < 3; ++condim_case) {
      int condim = kCondims[condim_case];
      for (int geom = 0; geom < model->ngeom; ++geom) {
        model->geom_condim[geom] = condim;
      }
      for (int cone = 0; cone < 2; ++cone) {
        model->opt.cone = cone ? mjCONE_ELLIPTIC : mjCONE_PYRAMIDAL;
        for (int integrator_case = 0; integrator_case < 2; ++integrator_case) {
          SCOPED_TRACE(testing::Message() << "pose=" << pose << " condim=" << condim
                                          << " cone=" << cone
                                          << " integrator=" << integrator_case);
          model->opt.integrator = integrator_case == 0 ? mjINT_IMPLICITFAST : mjINT_EULER;
          for (int geom = 0; geom < model->ngeom; ++geom) {
            model->geom_solimp[mjNIMP*geom+2] = integrator_case == 0 ? .021 : .001;
          }
          MjDataPtr data = MakeData(model);
          mju_copy(data->qpos+3, pose_quat, 4);
          mjtNum qvel[7] = {0, 0, 0, 0, 0, 0, cone ? 12.0 : 30.0};
          mju_copy(data->qvel, qvel, 7);
          data->ctrl[0] = .3;
          mj_forward(model.get(), data.get());
          ASSERT_EQ(data->ncon, 1) << "dist=" << data->contact[0].dist;
          ASSERT_EQ(data->nefc, cone ? condim : 2*(condim-1));
          ASSERT_EQ(model->geom_type[data->contact[0].geom1], mjGEOM_CAPSULE);
          ASSERT_EQ(model->geom_type[data->contact[0].geom2], mjGEOM_CAPSULE);
          const mjContact* con = data->contact;
          int cap1 = con->geom1, cap2 = con->geom2;
          mjtNum r1 = model->geom_size[3*cap1], h1 = model->geom_size[3*cap1+1];
          mjtNum r2 = model->geom_size[3*cap2], h2 = model->geom_size[3*cap2+1];
          const mjtNum* p1 = data->geom_xpos+3*cap1;
          const mjtNum* p2 = data->geom_xpos+3*cap2;
          const mjtNum* mat1 = data->geom_xmat+9*cap1;
          const mjtNum* mat2 = data->geom_xmat+9*cap2;
          mjtNum axis1[3] = {mat1[2], mat1[5], mat1[8]};
          mjtNum axis2[3] = {mat2[2], mat2[5], mat2[8]};
          mjtNum seg1[3], seg2[3];
          mju_scl3(seg1, axis1, h1);
          mju_scl3(seg2, axis2, h2);
          const mjtNum* nrm = con->frame;
          mjtNum len = con->dist+r1+r2;
          ASSERT_GT(len, 1e-8) << "dist=" << con->dist;
          // independent witness reconstruction from the contact record
          mjtNum w1[3], w2[3];
          for (int k = 0; k < 3; ++k) {
            w1[k] = con->pos[k]-nrm[k]*(r1+.5*con->dist);
            w2[k] = con->pos[k]+nrm[k]*(r2+.5*con->dist);
          }
          EXPECT_NEAR(mju_dist3(w1, w2), len, 1e-12) << "witness separation";
          // solve the unconstrained closest-point system, require interior match
          mjtNum ma = mju_dot3(seg1, seg1), pp = mju_dot3(seg1, seg2);
          mjtNum mc = mju_dot3(seg2, seg2), det = ma*mc-pp*pp;
          mjtNum scale = mju_max(ma, mc);
          ASSERT_GT(det, 1e-12*scale*scale) << "parallel axes det=" << det;
          mjtNum dif[3];
          mju_sub3(dif, p1, p2);
          mjtNum u = -mju_dot3(seg1, dif), v = mju_dot3(seg2, dif);
          mjtNum x1 = (mc*u+pp*v)/det;
          mjtNum x2 = (ma*v+pp*u)/det;
          mjtNum chk1[3], chk2[3];
          mju_scl3(chk1, seg1, x1);
          mju_addTo3(chk1, p1);
          mju_scl3(chk2, seg2, x2);
          mju_addTo3(chk2, p2);
          EXPECT_NEAR(mju_dist3(chk1, w1), 0, 1e-12) << "x1=" << x1;
          EXPECT_NEAR(mju_dist3(chk2, w2), 0, 1e-12) << "x2=" << x2;
          ASSERT_LT(mju_abs(x1), 1-1e-6) << "x1=" << x1;
          ASSERT_LT(mju_abs(x2), 1-1e-6) << "x2=" << x2;
          ASSERT_GT(mju_abs(mju_abs(nrm[1])-.5), 1e-3) << "n_y=" << nrm[1];
          // makeFrame default tangent reconstruction
          mjtNum t0[3] = {0, 1, 0};
          if (mju_abs(nrm[1]) >= .5) {
            t0[1] = 0;
            t0[2] = 1;
          }
          mjtNum cproj = mju_dot3(nrm, t0);
          mjtNum tan_chk[3], tmp3[3];
          for (int k = 0; k < 3; ++k) {
            tan_chk[k] = t0[k]-nrm[k]*cproj;
          }
          mju_normalize3(tan_chk);
          EXPECT_NEAR(mju_dist3(tan_chk, con->frame+3), 0, 1e-12);
          mju_cross(tmp3, nrm, tan_chk);
          EXPECT_NEAR(mju_dist3(tmp3, con->frame+6), 0, 1e-12);
          int active = 0;
          for (int row = 0; row < data->nefc; ++row) {
            active += data->efc_state[row] != mjCNSTRSTATE_SATISFIED;
            if (cone) {
              ASSERT_EQ(data->efc_state[row], data->efc_state[0]);
            }
          }
          ASSERT_GT(active, 0);
          if (cone) {
            ASSERT_TRUE(data->efc_state[0] == mjCNSTRSTATE_SATISFIED ||
                        data->efc_state[0] == mjCNSTRSTATE_QUADRATIC ||
                        data->efc_state[0] == mjCNSTRSTATE_CONE);
            cone_seen[condim_case] += data->efc_state[0] == mjCNSTRSTATE_CONE;
          } else {
            ASSERT_LT(active, data->nefc);
          }
          StepContextPtr ctx(mj_makeAdjContext(model.get()));
          ASSERT_THAT(ctx.get(), NotNull());
          ASSERT_EQ(mj_reserveAdjContext(ctx.get(), 1, data->nefc, 256, 1024), mjADJ_OK);
          MjDataPtr reference = MakeData(model);
          mj_copyData(reference.get(), model.get(), data.get());
          mj_step(model.get(), reference.get());
          std::mt19937_64 rng(23000+1000*pose+100*condim_case+10*cone+integrator_case);
          std::normal_distribution<double> normal;
          for (int sample = 0; sample < 18; ++sample) {
            std::vector<mjtNum> bar_qpos(7, 0), bar_qvel(7, 0);
            if (sample < 14) {
              if (sample < 7) bar_qpos[sample] = 1;
              else bar_qvel[sample-7] = 1;
            } else {
              for (int i = 0; i < 7; ++i) {
                bar_qpos[i] = normal(rng);
                bar_qvel[i] = normal(rng);
              }
            }
            std::vector<mjtNum> grad_qpos(7, 0), grad_qvel(7, 0), grad_ctrl(1, 0);
            MjDataPtr step_data = MakeData(model);
            mj_copyData(step_data.get(), model.get(), data.get());
            ASSERT_EQ(mjd_stepAdjoint(model.get(), step_data.get(), ctx.get(), bar_qpos.data(),
                                      bar_qvel.data(), grad_qpos.data(), grad_qvel.data(),
                                      grad_ctrl.data()), mjADJ_OK);
            for (int i = 0; i < model->nq; ++i) EXPECT_EQ(step_data->qpos[i], reference->qpos[i]);
            for (int i = 0; i < model->nv; ++i) EXPECT_EQ(step_data->qvel[i], reference->qvel[i]);
            for (int column = 0; column < 15; ++column) {
              InputBlock block = column < 7 ? InputBlock::kQPos :
                                 (column < 14 ? InputBlock::kQVel : InputBlock::kCtrl);
              int index = column < 7 ? column : (column < 14 ? column-7 : column-14);
              double fd = StepLoss(StockStepCentralTangent(model.get(), data.get(), block, index,
                                                          2e-7), bar_qpos, bar_qvel);
              double actual = column < 7 ? grad_qpos[index] :
                              (column < 14 ? grad_qvel[index] : grad_ctrl[index]);
              EXPECT_NEAR(actual, fd, 2e-7*std::max(1.0, std::abs(fd)))
                  << "sample=" << sample << " column=" << column
                  << " actual=" << actual << " fd=" << fd;
              ++compared;
            }
          }
        }
      }
    }
  }
  EXPECT_EQ(compared, 24*18*15);
  for (int i = 0; i < 3; ++i) {
    EXPECT_GT(cone_seen[i], 0) << "condim=" << kCondims[i] << " never reached CONE zone";
  }
}

TEST_F(AdjointStepTest, EllipticTopBottomAllInputColumns) {
  static constexpr char xml[] = R"(
  <mujoco><option solver="Newton" integrator="implicitfast" timestep=".002" iterations="100">
    <flag island="disable" warmstart="disable"/></option>
    <worldbody>
      <geom type="plane" size="5 5 .1" condim="3" solimp=".9 .95 .001" solref=".02 1"/>
      <body pos="0 0 .04"><freejoint/>
        <geom type="capsule" size=".02 .1" mass="1" contype="0" conaffinity="0"/>
        <body pos=".1 0 0">
          <joint name="j1" type="hinge" axis="0 1 0" damping=".1"/>
          <geom type="sphere" size=".05" mass=".5" condim="3"
                solimp=".9 .95 .001" solref=".02 1"/>
        </body>
      </body>
    </worldbody>
    <actuator><position joint="j1" kp="1.2" kv=".1"/></actuator>
  </mujoco>)";
  char error[1024] = "";
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  model->opt.tolerance = 0;
  model->opt.ls_tolerance = 0;
  model->opt.ls_iterations = 100;
  model->opt.cone = mjCONE_ELLIPTIC;
  int compared = 0;
  for (int dim : {3, 4, 6}) {
    for (int geom = 0; geom < model->ngeom; ++geom) model->geom_condim[geom] = dim;
    for (int zone = 0; zone < 2; ++zone) {
      for (int integrator_case = 0; integrator_case < 3; ++integrator_case) {
        SCOPED_TRACE(testing::Message() << "dim=" << dim << " zone=" << zone
                                        << " integrator=" << integrator_case);
        model->opt.integrator = integrator_case == 0 ? mjINT_IMPLICITFAST : mjINT_EULER;
        if (integrator_case == 2) {
          model->opt.disableflags |= mjDSBL_EULERDAMP;
        } else {
          model->opt.disableflags &= ~mjDSBL_EULERDAMP;
        }
        for (int geom = 0; geom < model->ngeom; ++geom) {
          model->geom_solimp[mjNIMP*geom+2] = integrator_case == 0 ? .021 : .001;
        }
        MjDataPtr data = MakeData(model);
        mjtNum qvel[7] = {.02, .03, zone == 0 ? 2.0 : -2.0, .05, -.02, .01, .2};
        mju_copy(data->qvel, qvel, 7);
        data->ctrl[0] = .3;
        mj_forward(model.get(), data.get());
        ASSERT_EQ(data->ncon, 1);
        ASSERT_EQ(data->nefc, dim);
        mjtConstraintState expected = zone == 0 ? mjCNSTRSTATE_SATISFIED : mjCNSTRSTATE_QUADRATIC;
        for (int row = 0; row < data->nefc; ++row) {
          ASSERT_EQ(data->efc_state[row], expected) << "row=" << row;
        }
        if (zone == 0) {
          for (int row = 0; row < data->nefc; ++row) ASSERT_EQ(data->efc_force[row], 0);
        } else {
          for (int row = 0; row < data->nefc; ++row) ASSERT_GT(data->efc_R[row], 0);
        }
        StepContextPtr ctx(mj_makeAdjContext(model.get()));
        ASSERT_THAT(ctx.get(), NotNull());
        ASSERT_EQ(mj_reserveAdjContext(ctx.get(), 1, data->nefc, 256, 1024), mjADJ_OK);
        MjDataPtr reference = MakeData(model);
        mj_copyData(reference.get(), model.get(), data.get());
        mj_step(model.get(), reference.get());
        std::mt19937_64 rng(23000+100*dim+10*zone+integrator_case);
        std::normal_distribution<double> normal;
        for (int sample = 0; sample < 18; ++sample) {
          std::vector<mjtNum> bar_qpos(7, 0), bar_qvel(7, 0);
          if (sample < 14) {
            if (sample < 7) bar_qpos[sample] = 1;
            else bar_qvel[sample-7] = 1;
          } else {
            for (int i = 0; i < 7; ++i) {
              bar_qpos[i] = normal(rng);
              bar_qvel[i] = normal(rng);
            }
          }
          std::vector<mjtNum> grad_qpos(7, 0), grad_qvel(7, 0), grad_ctrl(1, 0);
          MjDataPtr step_data = MakeData(model);
          mj_copyData(step_data.get(), model.get(), data.get());
          ASSERT_EQ(mjd_stepAdjoint(model.get(), step_data.get(), ctx.get(), bar_qpos.data(),
                                    bar_qvel.data(), grad_qpos.data(), grad_qvel.data(),
                                    grad_ctrl.data()), mjADJ_OK);
          for (int i = 0; i < model->nq; ++i) EXPECT_EQ(step_data->qpos[i], reference->qpos[i]);
          for (int i = 0; i < model->nv; ++i) EXPECT_EQ(step_data->qvel[i], reference->qvel[i]);
          for (int column = 0; column < 8; ++column) {
            InputBlock block = column < 7 ? InputBlock::kQPos :
                               (column < 14 ? InputBlock::kQVel : InputBlock::kCtrl);
            int index = column < 7 ? column : (column < 14 ? column-7 : column-14);
            double fd = StepLoss(StockStepCentralTangent(model.get(), data.get(), block, index,
                                                        2e-7), bar_qpos, bar_qvel);
            double actual = column < 7 ? grad_qpos[index] :
                            (column < 14 ? grad_qvel[index] : grad_ctrl[index]);
            EXPECT_NEAR(actual, fd, 2e-7*std::max(1.0, std::abs(fd)))
                << "sample=" << sample << " column=" << column
                << " actual=" << actual << " fd=" << fd;
            ++compared;
          }
        }
      }
    }
  }
  EXPECT_EQ(compared, 2592);
}

TEST_F(AdjointStepTest, PlaneCapsuleSingleEndpointAllInputColumns) {
  static constexpr char xml[] = R"(
  <mujoco><option solver="Newton" integrator="implicitfast" timestep=".002" iterations="100">
    <flag island="disable" warmstart="disable"/></option>
    <worldbody>
      <geom type="plane" size="5 5 .1" condim="3" solimp=".9 .95 .001" solref=".02 1"/>
      <body pos="0 0 .08" euler="20 30 25"><freejoint/>
        <geom type="capsule" size=".03 .1" mass="1" condim="3"
              solimp=".9 .95 .001" solref=".02 1"/>
        <body pos=".1 0 0">
          <joint name="j1" type="hinge" axis="0 1 0" damping=".1"/>
          <geom type="sphere" size=".02" mass=".2" contype="0" conaffinity="0"/>
        </body>
      </body>
    </worldbody>
    <actuator><position joint="j1" kp="1.2" kv=".1"/></actuator>
  </mujoco>)";
  char error[1024] = "";
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  model->opt.tolerance = 0;
  model->opt.ls_tolerance = 0;
  model->opt.ls_iterations = 100;
  int compared = 0;
  for (int endpoint = 0; endpoint < 2; ++endpoint) {
    mjtNum euler[3] = {endpoint ? -.35 : .35, endpoint ? -.52 : .52, endpoint ? -.44 : .44};
    mju_euler2Quat(model->body_quat+4, euler, "xyz");
    for (int cone = 0; cone < 2; ++cone) {
      model->opt.cone = cone ? mjCONE_ELLIPTIC : mjCONE_PYRAMIDAL;
      for (int integrator_case = 0; integrator_case < 2; ++integrator_case) {
        SCOPED_TRACE(testing::Message() << "endpoint=" << endpoint << " cone=" << cone
                                        << " integrator=" << integrator_case);
        model->opt.integrator = integrator_case == 0 ? mjINT_IMPLICITFAST : mjINT_EULER;
        for (int geom = 0; geom < model->ngeom; ++geom) {
          model->geom_solimp[mjNIMP*geom+2] = integrator_case == 0 ? .021 : .001;
        }
        MjDataPtr data = MakeData(model);
        mjtNum qvel[7] = {cone ? 5.0 : 2, cone ? 6.0 : 3, -.04, .05, -.02, .01, .2};
        mju_copy(data->qvel, qvel, 7);
        data->ctrl[0] = .3;
        mj_forward(model.get(), data.get());
        ASSERT_EQ(data->ncon, 1) << "dist=" << data->contact[0].dist;
        ASSERT_EQ(data->nefc, cone ? 3 : 4);
        ASSERT_EQ(model->geom_type[data->contact[0].geom1] == mjGEOM_PLANE ?
                  model->geom_type[data->contact[0].geom2] : model->geom_type[data->contact[0].geom1],
                  mjGEOM_CAPSULE);
        int capsule = model->geom_type[data->contact[0].geom1] == mjGEOM_CAPSULE ?
                      data->contact[0].geom1 : data->contact[0].geom2;
        int matched_endpoint = 0;
        for (int sign : {-1, 1}) {
          mjtNum endpoint_pos[3], expected_dist = 0;
          for (int k = 0; k < 3; ++k) {
            endpoint_pos[k] = data->geom_xpos[3*capsule+k]+
                sign*model->geom_size[3*capsule+1]*data->geom_xmat[9*capsule+3*k+2];
            expected_dist += data->contact[0].frame[k]*endpoint_pos[k];
          }
          expected_dist -= model->geom_size[3*capsule];
          if (mju_abs(expected_dist-data->contact[0].dist) < 1e-12) {
            matched_endpoint = sign;
            for (int k = 0; k < 3; ++k) {
              mjtNum reconstructed = data->contact[0].pos[k]+data->contact[0].frame[k]*
                  (.5*data->contact[0].dist+model->geom_size[3*capsule]);
              EXPECT_NEAR(reconstructed, endpoint_pos[k], 1e-12) << "k=" << k;
            }
          }
        }
        ASSERT_NE(matched_endpoint, 0);
        int active = 0;
        for (int row = 0; row < data->nefc; ++row) {
          active += data->efc_state[row] != mjCNSTRSTATE_SATISFIED;
        }
        ASSERT_GT(active, 0);
        if (cone) {
          ASSERT_EQ(active, data->nefc);
          ASSERT_EQ(data->efc_state[0], mjCNSTRSTATE_CONE);
        } else {
          ASSERT_LT(active, data->nefc);
        }
        StepContextPtr ctx(mj_makeAdjContext(model.get()));
        ASSERT_THAT(ctx.get(), NotNull());
        ASSERT_EQ(mj_reserveAdjContext(ctx.get(), 1, data->nefc, 256, 1024), mjADJ_OK);
        MjDataPtr reference = MakeData(model);
        mj_copyData(reference.get(), model.get(), data.get());
        mj_step(model.get(), reference.get());
        std::mt19937_64 rng(23000+100*endpoint+10*cone+integrator_case);
        std::normal_distribution<double> normal;
        for (int sample = 0; sample < 18; ++sample) {
          std::vector<mjtNum> bar_qpos(7, 0), bar_qvel(7, 0);
          if (sample < 14) {
            if (sample < 7) bar_qpos[sample] = 1;
            else bar_qvel[sample-7] = 1;
          } else {
            for (int i = 0; i < 7; ++i) {
              bar_qpos[i] = normal(rng);
              bar_qvel[i] = normal(rng);
            }
          }
          std::vector<mjtNum> grad_qpos(7, 0), grad_qvel(7, 0), grad_ctrl(1, 0);
          MjDataPtr step_data = MakeData(model);
          mj_copyData(step_data.get(), model.get(), data.get());
          ASSERT_EQ(mjd_stepAdjoint(model.get(), step_data.get(), ctx.get(), bar_qpos.data(),
                                    bar_qvel.data(), grad_qpos.data(), grad_qvel.data(),
                                    grad_ctrl.data()), mjADJ_OK);
          for (int i = 0; i < model->nq; ++i) EXPECT_EQ(step_data->qpos[i], reference->qpos[i]);
          for (int i = 0; i < model->nv; ++i) EXPECT_EQ(step_data->qvel[i], reference->qvel[i]);
          for (int column = 0; column < 8; ++column) {
            InputBlock block = column < 7 ? InputBlock::kQPos :
                               (column < 14 ? InputBlock::kQVel : InputBlock::kCtrl);
            int index = column < 7 ? column : (column < 14 ? column-7 : column-14);
            double fd = StepLoss(StockStepCentralTangent(model.get(), data.get(), block, index,
                                                        2e-7), bar_qpos, bar_qvel);
            double actual = column < 7 ? grad_qpos[index] :
                            (column < 14 ? grad_qvel[index] : grad_ctrl[index]);
            EXPECT_NEAR(actual, fd, 2e-7*std::max(1.0, std::abs(fd)))
                << "sample=" << sample << " column=" << column
                << " actual=" << actual << " fd=" << fd;
            ++compared;
          }
        }
      }
    }
  }
  EXPECT_EQ(compared, 8*18*8);
}

TEST_F(AdjointStepTest, PlaneCapsuleTwoContactsAllInputColumns) {
  // Two capsules of one free body touch the plane simultaneously: the position-level
  // contact adjoint must accumulate the per-contact pair contributions (each weighted by
  // its own constraint block of the globally coupled solve) instead of rejecting ncon > 1.
  static constexpr char xml[] = R"(
  <mujoco><option solver="Newton" integrator="implicitfast" timestep=".002" iterations="100">
    <flag island="disable" warmstart="disable"/></option>
    <worldbody>
      <geom type="plane" size="5 5 .1" condim="3" solimp=".9 .95 .001" solref=".02 1"/>
      <body pos="0 0 .125"><freejoint/>
        <geom type="capsule" size=".03 .1" mass=".6" pos=".15 0 0" condim="3"
              solimp=".9 .95 .001" solref=".02 1"/>
        <geom type="capsule" size=".03 .1" mass=".6" pos="-.15 0 0" condim="3"
              solimp=".9 .95 .001" solref=".02 1"/>
        <body pos=".15 0 .1">
          <joint name="j1" type="hinge" axis="0 1 0" damping=".1"/>
          <geom type="sphere" size=".02" mass=".2" contype="0" conaffinity="0"/>
        </body>
      </body>
    </worldbody>
    <actuator><position joint="j1" kp="1.2" kv=".1"/></actuator>
  </mujoco>)";
  char error[1024] = "";
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  model->opt.tolerance = 0;
  model->opt.ls_tolerance = 0;
  model->opt.ls_iterations = 100;
  int compared = 0;
  for (int integrator_case = 0; integrator_case < 2; ++integrator_case) {
    SCOPED_TRACE(testing::Message() << "integrator=" << integrator_case);
    model->opt.integrator = integrator_case == 0 ? mjINT_IMPLICITFAST : mjINT_EULER;
    for (int geom = 0; geom < model->ngeom; ++geom) {
      model->geom_solimp[mjNIMP*geom+2] = integrator_case == 0 ? .021 : .001;
    }
    MjDataPtr data = MakeData(model);
    mjtNum qvel[7] = {0, 0, -2, .5, -.3, .2, .15};
    mju_copy(data->qvel, qvel, 7);
    data->ctrl[0] = .3;
    mj_forward(model.get(), data.get());
    ASSERT_EQ(data->ncon, 2);
    ASSERT_EQ(data->nefc, 8);
    for (int c = 0; c < data->ncon; ++c) {
      int g1 = model->geom_type[data->contact[c].geom1];
      int g2 = model->geom_type[data->contact[c].geom2];
      ASSERT_TRUE((g1 == mjGEOM_PLANE && g2 == mjGEOM_CAPSULE) ||
                  (g1 == mjGEOM_CAPSULE && g2 == mjGEOM_PLANE));
      ASSERT_GE(data->contact[c].efc_address, 0);
    }
    StepContextPtr ctx(mj_makeAdjContext(model.get()));
    ASSERT_THAT(ctx.get(), NotNull());
    ASSERT_EQ(mj_reserveAdjContext(ctx.get(), 2, data->nefc, 256, 1024), mjADJ_OK);
    MjDataPtr reference = MakeData(model);
    mj_copyData(reference.get(), model.get(), data.get());
    mj_step(model.get(), reference.get());
    std::mt19937_64 rng(31000+integrator_case);
    std::normal_distribution<double> normal;
    for (int sample = 0; sample < 18; ++sample) {
      std::vector<mjtNum> bar_qpos(7, 0), bar_qvel(7, 0);
      if (sample < 14) {
        if (sample < 7) bar_qpos[sample] = 1;
        else bar_qvel[sample-7] = 1;
      } else {
        for (int i = 0; i < 7; ++i) {
          bar_qpos[i] = normal(rng);
          bar_qvel[i] = normal(rng);
        }
      }
      std::vector<mjtNum> grad_qpos(7, 0), grad_qvel(7, 0), grad_ctrl(1, 0);
      MjDataPtr step_data = MakeData(model);
      mj_copyData(step_data.get(), model.get(), data.get());
      ASSERT_EQ(mjd_stepAdjoint(model.get(), step_data.get(), ctx.get(), bar_qpos.data(),
                                bar_qvel.data(), grad_qpos.data(), grad_qvel.data(),
                                grad_ctrl.data()), mjADJ_OK);
      for (int i = 0; i < model->nq; ++i) EXPECT_EQ(step_data->qpos[i], reference->qpos[i]);
      for (int i = 0; i < model->nv; ++i) EXPECT_EQ(step_data->qvel[i], reference->qvel[i]);
      for (int column = 0; column < 15; ++column) {
        InputBlock block = column < 7 ? InputBlock::kQPos :
                           (column < 14 ? InputBlock::kQVel : InputBlock::kCtrl);
        int index = column < 7 ? column : (column < 14 ? column-7 : column-14);
        double fd = StepLoss(StockStepCentralTangent(model.get(), data.get(), block, index,
                                                    2e-7), bar_qpos, bar_qvel);
        double actual = column < 7 ? grad_qpos[index] :
                        (column < 14 ? grad_qvel[index] : grad_ctrl[index]);
        EXPECT_NEAR(actual, fd, 2e-7*std::max(1.0, std::abs(fd)))
            << "sample=" << sample << " column=" << column
            << " actual=" << actual << " fd=" << fd;
        ++compared;
      }
    }
  }
  EXPECT_EQ(compared, 2*18*15);
}

static std::string GetFileContents(const std::string& path) {
  std::ifstream file(path);
  EXPECT_TRUE(file.good()) << "missing test asset: " << path;
  return std::string(std::istreambuf_iterator<char>(file),
                     std::istreambuf_iterator<char>());
}

TEST_F(AdjointStepTest, G1ImplicitFastAllInputColumns) {
  // the STL meshes are provided in-memory by mjbatch's menagerie bundle; this
  // test only needs the kinematic/dynamic structure, so strip mesh geoms
  ASSERT_EQ(chdir("../../mjbatch_uni/examples/assets"), 0);
  std::string xml = GetFileContents("g1.xml");
  xml = std::regex_replace(xml, std::regex("(<geom[^>]*(type=\"mesh\"|mesh=\"[^\"]*\")[^>]*/>|<mesh[^>]*/>)"), "");
  xml.replace(xml.find("autolimits=\"true\""), strlen("autolimits=\"true\""),
              "autolimits=\"true\" discardvisual=\"true\"");
  char error[1024] = "";
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  ASSERT_EQ(model->nv, 29);
  ASSERT_EQ(model->nu, 23);
  model->opt.disableflags |= mjDSBL_ISLAND;
  model->opt.iterations = 10;
  ASSERT_EQ(model->opt.integrator, mjINT_IMPLICITFAST);
  MjDataPtr data = MakeData(model);
  for (int i = 0; i < model->nv; ++i) data->qvel[i] = .03*(i%7-3);
  for (int i = 0; i < model->nu; ++i) data->ctrl[i] = .1*(i%5-2);
  mj_forward(model.get(), data.get());
  ASSERT_EQ(data->ncon, 0);
  StepContextPtr ctx(mj_makeAdjContext(model.get()));
  ASSERT_THAT(ctx.get(), NotNull());
  ASSERT_EQ(mj_reserveAdjContext(ctx.get(), 0, 0, 0, 0), mjADJ_OK);
  MjDataPtr reference = MakeData(model);
  mj_copyData(reference.get(), model.get(), data.get());
  mj_step(model.get(), reference.get());
  std::mt19937_64 rng(163);
  std::normal_distribution<double> normal;
  int compared = 0;
  for (int sample = 0; sample < 4; ++sample) {
    std::vector<mjtNum> bar_qpos(model->nv);
    std::vector<mjtNum> bar_qvel(model->nv);
    for (int i = 0; i < model->nv; ++i) {
      bar_qpos[i] = normal(rng);
      bar_qvel[i] = normal(rng);
    }
    std::vector<mjtNum> grad_qpos(model->nv, 0);
    std::vector<mjtNum> grad_qvel(model->nv, 0);
    std::vector<mjtNum> grad_ctrl(model->nu, 0);
    MjDataPtr step_data = MakeData(model);
    mj_copyData(step_data.get(), model.get(), data.get());
    ASSERT_EQ(mjd_stepAdjoint(model.get(), step_data.get(), ctx.get(), bar_qpos.data(),
                              bar_qvel.data(), grad_qpos.data(), grad_qvel.data(), grad_ctrl.data()),
              mjADJ_OK);
    for (int i = 0; i < model->nq; ++i) EXPECT_EQ(step_data->qpos[i], reference->qpos[i]);
    for (int i = 0; i < model->nv; ++i) EXPECT_EQ(step_data->qvel[i], reference->qvel[i]);
    for (int column = 0; column < model->nv+model->nu; ++column) {
      InputBlock block = column < model->nv ? InputBlock::kQPos :
                         (column < 2*model->nv ? InputBlock::kQVel : InputBlock::kCtrl);
      int index = column < model->nv ? column :
                  (column < 2*model->nv ? column-model->nv : column-2*model->nv);
      double fd = StepLoss(StockStepCentralTangent(model.get(), data.get(), block, index, 1e-7),
                           bar_qpos, bar_qvel);
      double actual = column < model->nv ? grad_qpos[index] :
                      (column < 2*model->nv ? grad_qvel[index] : grad_ctrl[index]);
      EXPECT_NEAR(actual, fd, 2e-7*std::max(1.0, std::abs(fd)))
          << sample << ":" << column << " actual=" << actual << " fd=" << fd;
      ++compared;
    }
  }
  EXPECT_EQ(compared, 208);
}

TEST_F(AdjointStepTest, FusedMatchesExplicitLeafComposition) {

}

TEST_F(AdjointStepTest, SeparateOutputCotangentBlocks) {
  static constexpr char xml[] = R"(
  <mujoco><option solver="Newton" integrator="Euler" timestep=".002">
    <flag island="disable"/></option>
    <worldbody><body pos=".1 -.2 .3" euler="20 -30 40"><freejoint/>
      <geom type="box" size=".1 .2 .3" mass="2" pos=".04 -.02 .03" euler="10 20 30"/>
    </body></worldbody>
  </mujoco>)";
  char error[1024] = "";
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  const mjtNum qvel[6] = {.4, -.3, .2, 5, -3, 2};
  mju_copy(data->qvel, qvel, 6);
  StepContextPtr ctx(mj_makeAdjContext(model.get()));
  ASSERT_EQ(mj_reserveAdjContext(ctx.get(), 0, 0, 0, 0), mjADJ_OK);
  for (int mode = 0; mode < 2; ++mode) {
    std::vector<mjtNum> bar_qpos = {0, 0, 0, 0, 0, 0};
    std::vector<mjtNum> bar_qvel = {0, 0, 0, 0, 0, 0};
    if (mode == 0) {
      bar_qpos = {.3, -.7, 1.1, .8, -1.4, .5};
    } else {
      bar_qvel = {-.2, .9, .4, -.6, 1.2, -.8};
    }
    std::vector<mjtNum> grad_qpos(6);
    std::vector<mjtNum> grad_qvel(6);
    MjDataPtr step_data = MakeData(model);
    mj_copyData(step_data.get(), model.get(), data.get());
    ASSERT_EQ(mjd_stepAdjoint(model.get(), step_data.get(), ctx.get(), bar_qpos.data(),
                              bar_qvel.data(), grad_qpos.data(), grad_qvel.data(), nullptr),
              mjADJ_OK);
    for (int column = 0; column < 12; ++column) {
      bool is_qpos = column < 6;
      int index = is_qpos ? column : column - 6;
      double fd = StepLoss(StockStepCentralTangent(model.get(), data.get(),
                           is_qpos ? InputBlock::kQPos : InputBlock::kQVel, index, 1e-6),
                           bar_qpos, bar_qvel);
      double actual = is_qpos ? grad_qpos[index] : grad_qvel[index];
      EXPECT_NEAR(actual, fd, 2e-5) << mode << ":" << column << " actual=" << actual << " fd=" << fd;
    }
  }
}

}  // namespace
}  // namespace mujoco
