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

class AdjointMassTest : public MujocoTest {};

TEST_F(AdjointMassTest, FreeMassPositionAllColumns) {
  static constexpr char xml[] = R"(
  <mujoco><worldbody>
    <body pos=".1 -.2 .3" euler="20 -30 40"><freejoint/>
      <geom type="box" size=".1 .2 .3" mass="2" pos=".04 -.02 .03" euler="10 20 30"/>
    </body>
  </worldbody></mujoco>)";
  char error[1024] = "";
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());
  std::mt19937_64 rng(47);
  std::normal_distribution<double> normal;
  for (int sample = 0; sample < 4; ++sample) {
    mjtNum y[6];
    mjtNum w[6];
    for (int i = 0; i < 6; ++i) {
      y[i] = normal(rng);
      w[i] = normal(rng);
    }
    mjtNum gradient[6] = {0, 0, 0, 0, 0, 0};
    ASSERT_EQ(mjd_freeMassPosAdjoint(model.get(), data.get(), y, w, gradient), mjADJ_OK);
    for (int column = 0; column < 6; ++column) {
      mjtNum losses[2];
      for (int side = 0; side < 2; ++side) {
        MjDataPtr perturbed = MakeData(model);
        mj_copyData(perturbed.get(), model.get(), data.get());
        mjtNum direction[6] = {0, 0, 0, 0, 0, 0};
        direction[column] = side == 0 ? 1e-6 : -1e-6;
        mj_integratePos(model.get(), perturbed->qpos, direction, 1);
        mj_forward(model.get(), perturbed.get());
        mjtNum dense[36];
        mj_fullM(model.get(), perturbed.get(), dense);
        losses[side] = 0;
        for (int row = 0; row < 6; ++row) {
          for (int col = 0; col < 6; ++col) {
            losses[side] += y[row]*dense[6*row+col]*w[col];
          }
        }
      }
      EXPECT_NEAR(gradient[column], (losses[0] - losses[1])/2e-6, 3e-8) << sample << ":" << column;
    }
  }
  mjtNum y[6] = {1, 0, 0, 0, 0, 0};
  mjtNum w[6] = {0, 1, 0, 0, 0, 0};
  mjtNum gradient[6] = {0, 0, 0, 0, 0, 0};
  model->jnt_type[0] = mjJNT_BALL;
  EXPECT_EQ(mjd_freeMassPosAdjoint(model.get(), data.get(), y, w, gradient), mjADJ_UNSUPPORTED);
}

TEST_F(AdjointMassTest, FreeBiasPositionAllColumns) {
  static constexpr char xml[] = R"(
  <mujoco><worldbody>
    <body pos=".1 -.2 .3" euler="20 -30 40"><freejoint/>
      <geom type="box" size=".1 .2 .3" mass="2" pos=".04 -.02 .03" euler="10 20 30"/>
    </body>
  </worldbody></mujoco>)";
  char error[1024] = "";
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  const mjtNum qvel[6] = {.4, -.3, .2, 5, -3, 2};
  mju_copy(data->qvel, qvel, 6);
  mj_forward(model.get(), data.get());
  mjtNum bar[6] = {.3, -.7, 1.1, .8, -1.4, .5};
  mjtNum gradient[6] = {0, 0, 0, 0, 0, 0};
  ASSERT_EQ(mjd_freeBiasPosAdjoint(model.get(), data.get(), bar, gradient), mjADJ_OK);
  for (int column = 0; column < 6; ++column) {
    mjtNum bias_plus[6];
    mjtNum bias_minus[6];
    MjDataPtr plus = MakeData(model);
    MjDataPtr minus = MakeData(model);
    mj_copyData(plus.get(), model.get(), data.get());
    mj_copyData(minus.get(), model.get(), data.get());
    mjtNum direction_plus[6] = {0, 0, 0, 0, 0, 0};
    mjtNum direction_minus[6] = {0, 0, 0, 0, 0, 0};
    direction_plus[column] = 1e-6;
    direction_minus[column] = -1e-6;
    mj_integratePos(model.get(), plus->qpos, direction_plus, 1);
    mj_integratePos(model.get(), minus->qpos, direction_minus, 1);
    mj_forward(model.get(), plus.get());
    mj_forward(model.get(), minus.get());
    mju_copy(bias_plus, plus->qfrc_bias, 6);
    mju_copy(bias_minus, minus->qfrc_bias, 6);
    mjtNum fd = 0;
    for (int row = 0; row < 6; ++row) {
      fd += bar[row]*(bias_plus[row] - bias_minus[row])/2e-6;
    }
    EXPECT_NEAR(gradient[column], fd, 5e-8) << column;
  }
}

TEST_F(AdjointMassTest, FreeSmoothAccelerationPositionAllColumns) {
  static constexpr char xml[] = R"(
  <mujoco><option solver="Newton" integrator="Euler"><flag island="disable"/></option>
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
  mj_forward(model.get(), data.get());
  mjtNum bar[6] = {.3, -.7, 1.1, .8, -1.4, .5};
  mjtNum y[6];
  mjtNum gradient[6] = {0, 0, 0, 0, 0, 0};
  ASSERT_EQ(mjd_freeSmoothAccPosAdjoint(model.get(), data.get(), bar, y, gradient), mjADJ_OK);
  for (int column = 0; column < 6; ++column) {
    mjtNum losses[2];
    for (int side = 0; side < 2; ++side) {
      MjDataPtr perturbed = MakeData(model);
      mj_copyData(perturbed.get(), model.get(), data.get());
      mjtNum direction[6] = {0, 0, 0, 0, 0, 0};
      direction[column] = side == 0 ? 1e-6 : -1e-6;
      mj_integratePos(model.get(), perturbed->qpos, direction, 1);
      mj_forward(model.get(), perturbed.get());
      losses[side] = 0;
      for (int row = 0; row < 6; ++row) losses[side] += bar[row]*perturbed->qacc_smooth[row];
    }
    mjtNum fd = (losses[0] - losses[1])/2e-6;
    mjtNum scale = fd < 0 ? -fd : fd;
    mjtNum error = gradient[column] - fd;
    error = error < 0 ? -error : error;
    EXPECT_LT(error/(scale > 1 ? scale : 1), 1e-8) << column;
  }
  model->dof_damping[0] = .2;
  EXPECT_EQ(mjd_freeSmoothAccPosAdjoint(model.get(), data.get(), bar, y, gradient),
            mjADJ_UNSUPPORTED);
}

TEST_F(AdjointMassTest, FixedTreeSmoothAccelerationPositionAllColumns) {
  static constexpr char xml[] = R"(
  <mujoco><option solver="Newton" integrator="Euler"><flag island="disable"/></option>
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
  mj_forward(model.get(), data.get());
  std::mt19937_64 rng(131);
  std::normal_distribution<double> normal;
  for (int sample = 0; sample < 4; ++sample) {
    mjtNum bar[3];
    for (int i = 0; i < 3; ++i) bar[i] = normal(rng);
    mjtNum y[3];
    mjtNum gradient[3];
    ASSERT_EQ(mjd_fixedSmoothAccPosAdjoint(model.get(), data.get(), bar, y, gradient), mjADJ_OK);
    for (int column = 0; column < 3; ++column) {
      mjtNum losses[2];
      for (int side = 0; side < 2; ++side) {
        MjDataPtr perturbed = MakeData(model);
        mj_copyData(perturbed.get(), model.get(), data.get());
        mjtNum direction[3] = {0, 0, 0};
        direction[column] = side == 0 ? 1e-6 : -1e-6;
        mj_integratePos(model.get(), perturbed->qpos, direction, 1);
        mj_forward(model.get(), perturbed.get());
        losses[side] = mju_dot(bar, perturbed->qacc_smooth, 3);
      }
      mjtNum fd = (losses[0]-losses[1])/2e-6;
      EXPECT_NEAR(gradient[column], fd, 1e-8*std::max(1.0, std::abs(fd)))
          << sample << ":" << column << " analytic=" << gradient[column] << " fd=" << fd;
    }
  }
  model->jnt_type[0] = mjJNT_BALL;
  mjtNum bar[3] = {1, 0, 0};
  mjtNum y[3];
  mjtNum gradient[3];
  EXPECT_EQ(mjd_fixedSmoothAccPosAdjoint(model.get(), data.get(), bar, y, gradient),
            mjADJ_UNSUPPORTED);
}

TEST_F(AdjointMassTest, FreeTreeMassPositionAllColumns) {
  static constexpr char xml[] = R"(
  <mujoco><worldbody>
    <body pos=".1 -.2 .3" euler="20 -30 40"><freejoint/>
      <geom type="capsule" size=".08 .2" mass="1.2" pos=".04 -.02 .03" euler="10 20 30"/>
      <body pos=".25 .1 -.15" euler="-15 25 35">
        <joint type="hinge" axis="-.2 .6 .7"/>
        <geom type="box" size=".07 .09 .11" mass=".8" pos="-.03 .04 .02" euler="25 -10 15"/>
        <body pos=".12 -.08 .16" euler="20 10 -25">
          <joint type="slide" axis=".5 -.3 .8"/>
          <geom type="sphere" size=".1" mass=".4" pos=".03 .02 -.04"/>
        </body>
      </body>
    </body>
  </worldbody></mujoco>)";
  char error[1024] = "";
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  ASSERT_EQ(model->nv, 8);
  MjDataPtr data = MakeData(model);
  data->qpos[7] = .3;
  data->qpos[8] = -.4;
  mj_forward(model.get(), data.get());
  std::mt19937_64 rng(151);
  std::normal_distribution<double> normal;
  for (int sample = 0; sample < 4; ++sample) {
    std::vector<mjtNum> y(8);
    std::vector<mjtNum> w(8);
    for (int i = 0; i < 8; ++i) {
      y[i] = normal(rng);
      w[i] = normal(rng);
    }
    std::vector<mjtNum> gradient(8, 0);
    ASSERT_EQ(mjd_massPosAdjoint(model.get(), data.get(), y.data(), w.data(), gradient.data()), mjADJ_OK);
    for (int column = 0; column < 8; ++column) {
      mjtNum losses[2];
      for (int side = 0; side < 2; ++side) {
        MjDataPtr perturbed = MakeData(model);
        mj_copyData(perturbed.get(), model.get(), data.get());
        std::vector<mjtNum> direction(8, 0);
        direction[column] = side == 0 ? 1e-6 : -1e-6;
        mj_integratePos(model.get(), perturbed->qpos, direction.data(), 1);
        mj_forward(model.get(), perturbed.get());
        std::vector<mjtNum> dense(64);
        mj_fullM(model.get(), perturbed.get(), dense.data());
        losses[side] = 0;
        for (int row = 0; row < 8; ++row) {
          for (int col = 0; col < 8; ++col) losses[side] += y[row]*dense[8*row+col]*w[col];
        }
      }
      mjtNum fd = (losses[0]-losses[1])/2e-6;
      EXPECT_NEAR(gradient[column], fd, 2e-8*std::max(1.0, std::abs(fd)))
          << sample << ":" << column << " analytic=" << gradient[column] << " fd=" << fd;
    }
  }
}

TEST_F(AdjointMassTest, SingleHingeMassPosition) {
  static constexpr char xml[] = R"(
  <mujoco><worldbody>
    <body pos=".1 -.2 .3" euler="20 -30 40">
      <joint type="hinge" axis=".3 .8 .5"/>
      <geom type="capsule" size=".08 .2" mass="1.2" pos=".04 -.02 .03" euler="10 20 30"/>
      <body pos=".25 .1 -.15" euler="-15 25 35">
        <geom type="box" size=".07 .09 .11" mass=".8" pos="-.03 .04 .02" euler="25 -10 15"/>
      </body>
    </body>
  </worldbody></mujoco>)";
  char error[1024] = "";
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  data->qpos[0] = .3;
  mj_forward(model.get(), data.get());
  mjtNum y[1] = {.7};
  mjtNum w[1] = {-1.1};
  mjtNum gradient[1] = {0};
  ASSERT_EQ(mjd_massPosAdjoint(model.get(), data.get(), y, w, gradient), mjADJ_OK);
  mjtNum losses[2];
  for (int side = 0; side < 2; ++side) {
    MjDataPtr perturbed = MakeData(model);
    mj_copyData(perturbed.get(), model.get(), data.get());
    mjtNum direction[1] = {side == 0 ? 1e-6 : -1e-6};
    mj_integratePos(model.get(), perturbed->qpos, direction, 1);
    mj_forward(model.get(), perturbed.get());
    losses[side] = y[0]*perturbed->M[0]*w[0];
  }
  EXPECT_NEAR(gradient[0], (losses[0]-losses[1])/2e-6, 2e-8);
}

TEST_F(AdjointMassTest, DoubleHingeMassPosition) {
  static constexpr char xml[] = R"(
  <mujoco><worldbody>
    <body pos=".1 -.2 .3" euler="20 -30 40">
      <joint type="hinge" axis=".3 .8 .5"/>
      <geom type="capsule" size=".08 .2" mass="1.2" pos=".04 -.02 .03" euler="10 20 30"/>
      <body pos=".25 .1 -.15" euler="-15 25 35">
        <joint type="hinge" axis="-.2 .6 .7"/>
        <geom type="box" size=".07 .09 .11" mass=".8" pos="-.03 .04 .02" euler="25 -10 15"/>
      </body>
    </body>
  </worldbody></mujoco>)";
  char error[1024] = "";
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  const mjtNum qpos[2] = {.3, -.4};
  mju_copy(data->qpos, qpos, 2);
  mj_forward(model.get(), data.get());
  mjtNum y[2] = {.7, -1.1};
  mjtNum w[2] = {.4, .9};
  mjtNum gradient[2] = {0, 0};
  ASSERT_EQ(mjd_massPosAdjoint(model.get(), data.get(), y, w, gradient), mjADJ_OK);
  for (int column = 0; column < 2; ++column) {
    mjtNum losses[2];
    for (int side = 0; side < 2; ++side) {
      MjDataPtr perturbed = MakeData(model);
      mj_copyData(perturbed.get(), model.get(), data.get());
      mjtNum direction[2] = {0, 0};
      direction[column] = side == 0 ? 1e-6 : -1e-6;
      mj_integratePos(model.get(), perturbed->qpos, direction, 1);
      mj_forward(model.get(), perturbed.get());
      mjtNum dense[4];
      mj_fullM(model.get(), perturbed.get(), dense);
      losses[side] = 0;
      for (int row = 0; row < 2; ++row) {
        for (int col = 0; col < 2; ++col) losses[side] += y[row]*dense[2*row+col]*w[col];
      }
    }
    EXPECT_NEAR(gradient[column], (losses[0]-losses[1])/2e-6, 2e-8) << column;
  }
  y[1] = w[1] = 0;
  mju_zero(gradient, 2);
  ASSERT_EQ(mjd_massPosAdjoint(model.get(), data.get(), y, w, gradient), mjADJ_OK);
  mjtNum losses[2];
  for (int side = 0; side < 2; ++side) {
    MjDataPtr perturbed = MakeData(model);
    mj_copyData(perturbed.get(), model.get(), data.get());
    mjtNum direction[2] = {side == 0 ? 1e-6 : -1e-6, 0};
    mj_integratePos(model.get(), perturbed->qpos, direction, 1);
    mj_forward(model.get(), perturbed.get());
    mjtNum dense[4];
    mj_fullM(model.get(), perturbed.get(), dense);
    losses[side] = y[0]*dense[0]*w[0];
  }
  EXPECT_NEAR(gradient[0], (losses[0]-losses[1])/2e-6, 2e-8);
  y[0] = .7;
  y[1] = -1.1;
  w[0] = .4;
  w[1] = .9;
  mju_zero(model->body_inertia+6, 3);
  mj_forward(model.get(), data.get());
  mju_zero(gradient, 2);
  ASSERT_EQ(mjd_massPosAdjoint(model.get(), data.get(), y, w, gradient), mjADJ_OK);
  for (int side = 0; side < 2; ++side) {
    MjDataPtr perturbed = MakeData(model);
    mj_copyData(perturbed.get(), model.get(), data.get());
    mjtNum direction[2] = {side == 0 ? 1e-6 : -1e-6, 0};
    mj_integratePos(model.get(), perturbed->qpos, direction, 1);
    mj_forward(model.get(), perturbed.get());
    mjtNum dense[4];
    mj_fullM(model.get(), perturbed.get(), dense);
    losses[side] = 0;
    for (int row = 0; row < 2; ++row) {
      for (int col = 0; col < 2; ++col) losses[side] += y[row]*dense[2*row+col]*w[col];
    }
  }
  EXPECT_NEAR(gradient[0], (losses[0]-losses[1])/2e-6, 2e-8);
}

TEST_F(AdjointMassTest, FixedTreeMassPositionAllColumns) {
  static constexpr char xml[] = R"(
  <mujoco><worldbody>
    <body pos=".1 -.2 .3" euler="20 -30 40">
      <joint type="hinge" axis=".3 .8 .5"/>
      <geom type="capsule" size=".08 .2" mass="1.2" pos=".04 -.02 .03" euler="10 20 30"/>
      <body pos=".25 .1 -.15" euler="-15 25 35">
        <joint type="hinge" axis="-.2 .6 .7"/>
        <geom type="box" size=".07 .09 .11" mass=".8" pos="-.03 .04 .02" euler="25 -10 15"/>
        <body pos=".12 -.08 .16" euler="20 10 -25">
          <joint type="slide" axis=".5 -.3 .8"/>
          <geom type="sphere" size=".1" mass=".4" pos=".03 .02 -.04"/>
        </body>
      </body>
    </body>
  </worldbody></mujoco>)";
  char error[1024] = "";
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  ASSERT_EQ(model->nv, 3);
  MjDataPtr data = MakeData(model);
  const mjtNum qpos[3] = {.3, -.4, .25};
  mju_copy(data->qpos, qpos, 3);
  mj_forward(model.get(), data.get());
  std::mt19937_64 rng(91);
  std::normal_distribution<double> normal;
  for (int sample = 0; sample < 4; ++sample) {
    mjtNum y[3];
    mjtNum w[3];
    for (int i = 0; i < 3; ++i) {
      y[i] = normal(rng);
      w[i] = normal(rng);
    }
    mjtNum gradient[3] = {0, 0, 0};
    ASSERT_EQ(mjd_massPosAdjoint(model.get(), data.get(), y, w, gradient), mjADJ_OK);
    for (int column = 0; column < 3; ++column) {
      mjtNum losses[2];
      for (int side = 0; side < 2; ++side) {
        MjDataPtr perturbed = MakeData(model);
        mj_copyData(perturbed.get(), model.get(), data.get());
        mjtNum direction[3] = {0, 0, 0};
        direction[column] = side == 0 ? 1e-6 : -1e-6;
        mj_integratePos(model.get(), perturbed->qpos, direction, 1);
        mj_forward(model.get(), perturbed.get());
        mjtNum dense[9];
        mj_fullM(model.get(), perturbed.get(), dense);
        losses[side] = 0;
        for (int row = 0; row < 3; ++row) {
          for (int col = 0; col < 3; ++col) {
            losses[side] += y[row]*dense[3*row+col]*w[col];
          }
        }
      }
      mjtNum fd = (losses[0]-losses[1])/2e-6;
      EXPECT_NEAR(gradient[column], fd, 2e-8) << sample << ":" << column;
    }
  }
  model->jnt_type[0] = mjJNT_BALL;
  mjtNum y[3] = {1, 0, 0};
  mjtNum w[3] = {0, 1, 0};
  mjtNum gradient[3] = {0, 0, 0};
  EXPECT_EQ(mjd_massPosAdjoint(model.get(), data.get(), y, w, gradient), mjADJ_UNSUPPORTED);
}

}  // namespace
}  // namespace mujoco
