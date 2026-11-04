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

class AdjointRneTest : public MujocoTest {};

TEST_F(AdjointRneTest, HingeChainJvpVjpAndPersistentDerivative) {
  static constexpr char xml[] = R"(
  <mujoco><option solver="Newton"><flag island="disable"/></option>
    <worldbody><body><joint name="j0" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size=".03" fromto="0 0 0 0 0 .4" mass="1"/>
      <body pos="0 0 .4"><joint name="j1" type="hinge" axis="1 0 0"/>
        <geom type="capsule" size=".02" fromto="0 0 0 .3 0 0" mass=".5"/>
      </body>
    </body></worldbody>
  </mujoco>)";
  char error[1024] = "";
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  data->qpos[0] = .4;
  data->qpos[1] = -.7;
  data->qvel[0] = .8;
  data->qvel[1] = -1.2;
  mj_forward(model.get(), data.get());
  std::vector<mjtNum> persistent(data->qDeriv, data->qDeriv + model->nD);
  std::vector<mjtNum> scratch(model->nD);
  std::mt19937_64 rng(23);
  std::normal_distribution<double> normal;
  for (int sample = 0; sample < 4; ++sample) {
    std::vector<mjtNum> bar = {normal(rng), normal(rng)};
    std::vector<mjtNum> direction = {normal(rng), normal(rng)};
    std::vector<mjtNum> gradient(2, 0);
    ASSERT_EQ(mjd_rneBiasVelAdjoint(model.get(), data.get(), bar.data(), scratch.data(),
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
      jvp_vjp += bar[i]*(plus->qfrc_bias[i] - minus->qfrc_bias[i])/(2*eps);
      vjp_jvp += gradient[i]*direction[i];
    }
    EXPECT_NEAR(vjp_jvp, jvp_vjp, 2e-8);
    EXPECT_EQ(std::memcmp(persistent.data(), data->qDeriv, model->nD*sizeof(mjtNum)), 0);
  }
}

TEST_F(AdjointRneTest, FreeBodyAllColumnsIndependentRneFd) {
  static constexpr char xml[] = R"(
  <mujoco><option solver="Newton"><flag island="disable"/></option>
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
  std::vector<mjtNum> bar = {.3, -.7, 1.1, .8, -1.4, .5};
  std::vector<mjtNum> scratch(model->nD);
  std::vector<mjtNum> gradient(6, 0);
  ASSERT_EQ(mjd_rneBiasVelAdjoint(model.get(), data.get(), bar.data(), scratch.data(),
                                  gradient.data()), mjADJ_OK);
  for (int column = 0; column < 6; ++column) {
    mjtNum bias_plus[6];
    mjtNum bias_minus[6];
    MjDataPtr plus = MakeData(model);
    MjDataPtr minus = MakeData(model);
    mj_copyData(plus.get(), model.get(), data.get());
    mj_copyData(minus.get(), model.get(), data.get());
    plus->qvel[column] += 1e-6;
    minus->qvel[column] -= 1e-6;
    mj_comVel(model.get(), plus.get());
    mj_comVel(model.get(), minus.get());
    mj_rne(model.get(), plus.get(), 0, bias_plus);
    mj_rne(model.get(), minus.get(), 0, bias_minus);
    mjtNum fd = 0;
    for (int row = 0; row < 6; ++row) {
      fd += bar[row]*(bias_plus[row] - bias_minus[row])/2e-6;
    }
    EXPECT_NEAR(gradient[column], fd, 2e-8) << column;
  }
}

static void ExpectPositionGradientMatchesFd(const char* xml,
                                            const std::vector<mjtNum>& qpos,
                                            const std::vector<mjtNum>& qvel,
                                            bool check_reference = true) {
  char error[1024] = "";
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  mju_copy(data->qpos, qpos.data(), qpos.size());
  mju_copy(data->qvel, qvel.data(), qvel.size());
  mj_forward(model.get(), data.get());
  std::vector<mjtNum> storage(18*model->nbody, 0);
  mjtNum* omega = storage.data();
  mjtNum* alpha = omega+3*model->nbody;
  mjtNum* acc = alpha+3*model->nbody;
  mjtNum* axes = acc+3*model->nbody;
  mjtNum* force = axes+3*model->nbody;
  mjtNum* moment = force+3*model->nbody;
  mju_scl3(acc, model->opt.gravity, -1);
  for (int body = 1; body < model->nbody; ++body) {
    int parent = model->body_parentid[body];
    mjtNum offset[3], tmp[3], tmp1[3];
    mju_sub3(offset, data->xpos+3*body, data->xpos+3*parent);
    mju_copy(omega+3*body, omega+3*parent, 3);
    mju_copy(alpha+3*body, alpha+3*parent, 3);
    mju_cross(acc+3*body, alpha+3*parent, offset);
    mju_addTo3(acc+3*body, acc+3*parent);
    mju_cross(tmp, omega+3*parent, offset);
    mju_cross(tmp1, omega+3*parent, tmp);
    mju_addTo3(acc+3*body, tmp1);
    if (model->body_jntnum[body]) {
      int joint = model->body_jntadr[body];
      int dof = model->jnt_dofadr[joint];
      mju_mulMatVec3(axes+3*body, data->xmat+9*body, model->jnt_axis+3*joint);
      mju_scl3(tmp, axes+3*body, data->qvel[dof]);
      if (model->jnt_type[joint] == mjJNT_HINGE) {
        mju_addTo3(omega+3*body, tmp);
        mju_cross(tmp1, omega+3*parent, tmp);
        mju_addTo3(alpha+3*body, tmp1);
      } else {
        mju_cross(tmp1, omega+3*parent, tmp);
        mju_scl3(tmp1, tmp1, 2);
        mju_addTo3(acc+3*body, tmp1);
      }
    }
    mjtNum com[3], centrifugal[3];
    mju_mulMatVec3(com, data->xmat+9*body, model->body_ipos+3*body);
    mju_cross(tmp, alpha+3*body, com);
    mju_cross(tmp1, omega+3*body, com);
    mju_cross(centrifugal, omega+3*body, tmp1);
    for (int k = 0; k < 3; ++k) {
      force[3*body+k] = model->body_mass[body]*(acc[3*body+k]+tmp[k]+centrifugal[k]);
    }
    mjtNum local[3], inertia_omega[3];
    mju_mulMatTVec3(local, data->ximat+9*body, omega+3*body);
    for (int k = 0; k < 3; ++k) local[k] *= model->body_inertia[3*body+k];
    mju_mulMatVec3(inertia_omega, data->ximat+9*body, local);
    mju_mulMatTVec3(local, data->ximat+9*body, alpha+3*body);
    for (int k = 0; k < 3; ++k) local[k] *= model->body_inertia[3*body+k];
    mju_mulMatVec3(moment+3*body, data->ximat+9*body, local);
    mju_cross(tmp, omega+3*body, inertia_omega);
    mju_addTo3(moment+3*body, tmp);
    mju_cross(tmp, com, force+3*body);
    mju_addTo3(moment+3*body, tmp);
  }
  for (int body = model->nbody-1; body > 0; --body) {
    int parent = model->body_parentid[body];
    if (!parent) continue;
    mjtNum offset[3], tmp[3];
    mju_sub3(offset, data->xpos+3*body, data->xpos+3*parent);
    mju_addTo3(moment+3*parent, moment+3*body);
    mju_cross(tmp, offset, force+3*body);
    mju_addTo3(moment+3*parent, tmp);
    mju_addTo3(force+3*parent, force+3*body);
  }
  for (int body = 1; body < model->nbody; ++body) {
    if (!model->body_jntnum[body]) continue;
    int joint = model->body_jntadr[body];
    int dof = model->jnt_dofadr[joint];
    mjtNum expected = model->jnt_type[joint] == mjJNT_HINGE ?
        mju_dot3(axes+3*body, moment+3*body) : mju_dot3(axes+3*body, force+3*body);
    if (check_reference) {
    EXPECT_NEAR(expected, data->qfrc_bias[dof], 1e-10) << "base body=" << body;
  }
  }
  std::mt19937_64 rng(117);
  std::normal_distribution<double> normal;
  for (int sample = 0; sample < 4; ++sample) {
    std::vector<mjtNum> bar(model->nv);
    for (int i = 0; i < model->nv; ++i) bar[i] = normal(rng);
    std::vector<mjtNum> gradient(model->nv, 0);
    ASSERT_EQ(mjd_rneBiasPosAdjoint(model.get(), data.get(), bar.data(), gradient.data()), mjADJ_OK);
    for (int column = 0; column < model->nv; ++column) {
      mjtNum losses[2];
      for (int side = 0; side < 2; ++side) {
        MjDataPtr perturbed = MakeData(model);
        mj_copyData(perturbed.get(), model.get(), data.get());
        std::vector<mjtNum> direction(model->nv, 0);
        direction[column] = side == 0 ? 1e-6 : -1e-6;
        mj_integratePos(model.get(), perturbed->qpos, direction.data(), 1);
        mj_forward(model.get(), perturbed.get());
        losses[side] = mju_dot(bar.data(), perturbed->qfrc_bias, model->nv);
      }
      mjtNum fd = (losses[0]-losses[1])/2e-6;
      EXPECT_NEAR(gradient[column], fd, 1e-8*std::max(1.0, std::abs(fd)))
          << sample << ":" << column << " analytic=" << gradient[column] << " fd=" << fd
          << " bar=" << bar[column];
    }
  }
  for (int row = 0; row < model->nv; ++row) {
    std::vector<mjtNum> bar(model->nv, 0);
    bar[row] = 1;
    std::vector<mjtNum> gradient(model->nv, 0);
    ASSERT_EQ(mjd_rneBiasPosAdjoint(model.get(), data.get(), bar.data(), gradient.data()), mjADJ_OK);
    for (int column = 0; column < model->nv; ++column) {
      MjDataPtr plus = MakeData(model);
      MjDataPtr minus = MakeData(model);
      mj_copyData(plus.get(), model.get(), data.get());
      mj_copyData(minus.get(), model.get(), data.get());
      std::vector<mjtNum> direction(model->nv, 0);
      direction[column] = 1e-6;
      mj_integratePos(model.get(), plus->qpos, direction.data(), 1);
      mju_scl(direction.data(), direction.data(), -1, model->nv);
      mj_integratePos(model.get(), minus->qpos, direction.data(), 1);
      mj_forward(model.get(), plus.get());
      mj_forward(model.get(), minus.get());
      mjtNum fd = (plus->qfrc_bias[row]-minus->qfrc_bias[row])/2e-6;
      EXPECT_NEAR(gradient[column], fd, 1e-8*std::max(1.0, std::abs(fd)))
          << "row=" << row << " column=" << column << " qvel0=" << qvel[0]
          << " qvel1=" << qvel[1] << " analytic=" << gradient[column] << " fd=" << fd;
    }
  }
}

TEST_F(AdjointRneTest, SingleHingePositionAllColumnsIndependentFd) {
  static constexpr char xml[] = R"(
  <mujoco><worldbody>
    <body pos=".1 -.2 .3" euler="20 -30 40">
      <joint type="hinge" axis=".3 .8 .5"/>
      <geom type="capsule" size=".08 .2" mass="1.2" pos=".04 -.02 .03" euler="10 20 30"/>
    </body>
  </worldbody></mujoco>)";
  ExpectPositionGradientMatchesFd(xml, {.3}, {.8});
}

TEST_F(AdjointRneTest, DoubleHingePositionAllColumnsIndependentFd) {
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
  ExpectPositionGradientMatchesFd(xml, {.3, -.4}, {.8, -1.2});
  ExpectPositionGradientMatchesFd(xml, {.3, -.4}, {.8, 0});
  ExpectPositionGradientMatchesFd(xml, {.3, -.4}, {0, -1.2});
  ExpectPositionGradientMatchesFd(xml, {.3, -.4}, {0, 0});
  char error[1024] = "";
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  data->qpos[0] = .3;
  data->qpos[1] = -.4;
  mj_forward(model.get(), data.get());
  mjtNum axis[3], com[3], gravity_force[3], gravity_moment[3];
  mju_mulMatVec3(axis, data->xmat+18, model->jnt_axis+3);
  mju_sub3(com, data->xipos+6, data->xpos+6);
  mju_scl3(gravity_force, model->opt.gravity, -model->body_mass[2]);
  mju_cross(gravity_moment, com, gravity_force);
  EXPECT_NEAR(data->qfrc_bias[1], mju_dot3(axis, gravity_moment), 1e-12);
  mjtNum delta_com[3], delta_moment[3];
  mju_cross(delta_com, axis, com);
  mju_cross(delta_moment, delta_com, gravity_force);
  mjtNum expected = mju_dot3(axis, delta_moment);
  mjtNum bar[2] = {0, 1};
  mjtNum gradient[2];
  ASSERT_EQ(mjd_rneBiasPosAdjoint(model.get(), data.get(), bar, gradient), mjADJ_OK);
  MjDataPtr plus = MakeData(model);
  MjDataPtr minus = MakeData(model);
  mj_copyData(plus.get(), model.get(), data.get());
  mj_copyData(minus.get(), model.get(), data.get());
  mjtNum direction[2] = {0, 1e-6};
  mj_integratePos(model.get(), plus->qpos, direction, 1);
  mju_scl(direction, direction, -1, 2);
  mj_integratePos(model.get(), minus->qpos, direction, 1);
  mj_forward(model.get(), plus.get());
  mj_forward(model.get(), minus.get());
  mjtNum fd = (plus->qfrc_bias[1]-minus->qfrc_bias[1])/2e-6;
  EXPECT_NEAR(gradient[1], expected, 1e-12);
  EXPECT_NEAR(gradient[1], fd, 1e-8) << "analytic=" << gradient[1] << " fd=" << fd;
}

TEST_F(AdjointRneTest, FixedChainPositionAllColumnsIndependentFd) {
  static constexpr char xml[] = R"(
  <mujoco><worldbody>
    <body pos=".1 -.2 .3" euler="20 -30 40">
      <joint type="hinge" axis=".3 .8 .5"/>
      <body pos=".25 .1 -.15" euler="-15 25 35">
        <body pos=".245 -.155 .36" euler="20 10 -25">
          <geom type="sphere" size=".1" mass=".4" pos=".03 .02 -.04"/>
        </body>
      </body>
    </body>
  </worldbody></mujoco>)";
  ExpectPositionGradientMatchesFd(xml, {.3}, {.8});
  ExpectPositionGradientMatchesFd(xml, {.3}, {0});
}

TEST_F(AdjointRneTest, RootOnlyPositionRepro) {
  static constexpr char xml[] = R"(
  <mujoco><worldbody>
    <body pos=".1 -.2 .3" euler="20 -30 40">
      <joint type="hinge" axis=".3 .8 .5"/>
      <geom type="capsule" size=".08 .2" mass="1.2" pos=".04 -.02 .03" euler="10 20 30"/>
      <body pos=".25 .1 -.15" euler="-15 25 35">
        <joint type="hinge" axis="-.2 .6 .7"/>
        <geom type="box" size=".07 .09 .11" mass=".8" pos="-.03 .04 .02" euler="25 -10 15"/>
        <body pos=".245 -.155 .36" euler="20 10 -25">
          <geom type="sphere" size=".1" mass=".4" pos=".03 .02 -.04"/>
        </body>
      </body>
    </body>
  </worldbody></mujoco>)";
  ExpectPositionGradientMatchesFd(xml, {.3, -.4}, {.8, 0});
}

TEST_F(AdjointRneTest, FixedExtensionPositionAllColumnsIndependentFd) {
  static constexpr char xml[] = R"(
  <mujoco><worldbody>
    <body pos=".1 -.2 .3" euler="20 -30 40">
      <joint type="hinge" axis=".3 .8 .5"/>
      <geom type="capsule" size=".08 .2" mass="1.2" pos=".04 -.02 .03" euler="10 20 30"/>
      <body pos=".25 .1 -.15" euler="-15 25 35">
        <joint type="hinge" axis="-.2 .6 .7"/>
        <geom type="box" size=".07 .09 .11" mass=".8" pos="-.03 .04 .02" euler="25 -10 15"/>
        <body pos=".245 -.155 .36" euler="20 10 -25">
          <geom type="sphere" size=".1" mass=".4" pos=".03 .02 -.04"/>
        </body>
      </body>
    </body>
  </worldbody></mujoco>)";
  ExpectPositionGradientMatchesFd(xml, {.3, -.4}, {.8, -1.2});
  ExpectPositionGradientMatchesFd(xml, {.3, -.4}, {.8, 0});
  ExpectPositionGradientMatchesFd(xml, {.3, -.4}, {0, -1.2});
  ExpectPositionGradientMatchesFd(xml, {.3, -.4}, {0, 0});
}

TEST_F(AdjointRneTest, FixedTreePositionAllColumnsIndependentFd) {
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
  const mjtNum qvel[3] = {.8, -1.2, .6};
  mju_copy(data->qpos, qpos, 3);
  mju_copy(data->qvel, qvel, 3);
  mj_forward(model.get(), data.get());
  std::mt19937_64 rng(117);
  std::normal_distribution<double> normal;
  for (int sample = 0; sample < 4; ++sample) {
    mjtNum bar[3];
    for (int i = 0; i < 3; ++i) bar[i] = normal(rng);
    mjtNum gradient[3] = {0, 0, 0};
    ASSERT_EQ(mjd_rneBiasPosAdjoint(model.get(), data.get(), bar, gradient), mjADJ_OK);
    for (int column = 0; column < 3; ++column) {
      mjtNum losses[2];
      for (int side = 0; side < 2; ++side) {
        MjDataPtr perturbed = MakeData(model);
        mj_copyData(perturbed.get(), model.get(), data.get());
        mjtNum direction[3] = {0, 0, 0};
        direction[column] = side == 0 ? 1e-6 : -1e-6;
        mj_integratePos(model.get(), perturbed->qpos, direction, 1);
        mj_forward(model.get(), perturbed.get());
        losses[side] = 0;
        for (int row = 0; row < 3; ++row) losses[side] += bar[row]*perturbed->qfrc_bias[row];
      }
      mjtNum fd = (losses[0]-losses[1])/2e-6;
      mjtNum scale = fd < 0 ? -fd : fd;
      mjtNum error = gradient[column]-fd;
      error = error < 0 ? -error : error;
      EXPECT_LT(error/(scale > 1 ? scale : 1), 1e-8)
          << sample << ":" << column << " analytic=" << gradient[column] << " fd=" << fd;
    }
  }
  for (int row = 0; row < 3; ++row) {
    mjtNum row_bar[3] = {0, 0, 0};
    row_bar[row] = 1;
    mjtNum row_gradient[3];
    ASSERT_EQ(mjd_rneBiasPosAdjoint(model.get(), data.get(), row_bar, row_gradient), mjADJ_OK);
    MjDataPtr plus = MakeData(model);
    MjDataPtr minus = MakeData(model);
    mj_copyData(plus.get(), model.get(), data.get());
    mj_copyData(minus.get(), model.get(), data.get());
    mjtNum direction[3] = {1e-6, 0, 0};
    mj_integratePos(model.get(), plus->qpos, direction, 1);
    mju_scl(direction, direction, -1, 3);
    mj_integratePos(model.get(), minus->qpos, direction, 1);
    mj_forward(model.get(), plus.get());
    mj_forward(model.get(), minus.get());
    mjtNum fd = (plus->qfrc_bias[row]-minus->qfrc_bias[row])/2e-6;
    EXPECT_NEAR(row_gradient[0], fd, 1e-8*std::max(1.0, std::abs(fd)))
        << "row=" << row << " analytic=" << row_gradient[0] << " fd=" << fd;
  }
  ExpectPositionGradientMatchesFd(xml, {.3, -.4, .25}, {.8, -1.2, 0});
  ExpectPositionGradientMatchesFd(xml, {.3, -.4, .25}, {0, 0, .6});
  ExpectPositionGradientMatchesFd(xml, {.3, -.4, .25}, {0, 0, 0});
  model->jnt_type[0] = mjJNT_BALL;
  mjtNum bar[3] = {1, 0, 0};
  mjtNum gradient[3] = {0, 0, 0};
  EXPECT_EQ(mjd_rneBiasPosAdjoint(model.get(), data.get(), bar, gradient), mjADJ_UNSUPPORTED);
}

TEST_F(AdjointRneTest, FreeRootOnlyPositionAllColumnsIndependentFd) {
  static constexpr char xml[] = R"(
  <mujoco><worldbody><body pos=".1 -.2 .3" euler="20 -30 40"><freejoint/>
      <geom type="box" size=".1 .2 .3" mass="2" pos=".04 -.02 .03" euler="10 20 30"/>
    </body></worldbody></mujoco>)";
  mjtNum qpos[7] = {.24, -.18, .31, .91249457, -.2644027, .12337187, .28139351};
  ExpectPositionGradientMatchesFd(xml,
      std::vector<mjtNum>(qpos, qpos+7), {0, 0, 0, 0, 0, 0}, false);
}

TEST_F(AdjointRneTest, FreeRootHingeDescendantsPositionAllColumnsIndependentFd) {
  static constexpr char xml[] = R"(
  <mujoco><worldbody>
    <body pos=".1 -.2 .3" euler="20 -30 40"><freejoint/>
      <geom type="capsule" size=".08 .2" mass="1.2" pos=".04 -.02 .03" euler="10 20 30"/>
      <body pos=".25 .1 -.15" euler="-15 25 35">
        <joint type="hinge" axis="-.2 .6 .7"/>
        <geom type="box" size=".07 .09 .11" mass=".8" pos="-.03 .04 .02" euler="25 -10 15"/>
        <body pos=".12 -.08 .16" euler="20 10 -25">
          <geom type="sphere" size=".1" mass=".4" pos=".03 .02 -.04"/>
        </body>
      </body>
    </body>
  </worldbody></mujoco>)";
  mjtNum qpos[10] = {.24, -.18, .31, .91249457, -.2644027, .12337187, .28139351,
                     -.4, 0, 0};
  ExpectPositionGradientMatchesFd(xml,
      std::vector<mjtNum>(qpos, qpos+10), {.4, -.3, .2, .5, -.6, .7, .8}, false);
  ExpectPositionGradientMatchesFd(xml,
      std::vector<mjtNum>(qpos, qpos+10), {0, 0, 0, 0, 0, 0, .8}, false);
  ExpectPositionGradientMatchesFd(xml,
      std::vector<mjtNum>(qpos, qpos+10), {0, 0, 0, 0, 0, 0, 0}, false);
}

TEST_F(AdjointRneTest, FreeTreePositionAllColumnsIndependentFd) {
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
  const mjtNum qvel[8] = {.4, -.3, .2, .5, -.6, .7, .8, -1.2};
  mju_copy(data->qvel, qvel, 8);
  data->qpos[7] = .3;
  data->qpos[8] = -.4;
  mj_forward(model.get(), data.get());
  std::mt19937_64 rng(149);
  std::normal_distribution<double> normal;
  for (int sample = 0; sample < 4; ++sample) {
    std::vector<mjtNum> bar(8);
    for (int i = 0; i < 8; ++i) bar[i] = normal(rng);
    std::vector<mjtNum> gradient(8, 0);
    ASSERT_EQ(mjd_rneBiasPosAdjoint(model.get(), data.get(), bar.data(), gradient.data()), mjADJ_OK);
    for (int column = 0; column < 8; ++column) {
      mjtNum losses[2];
      for (int side = 0; side < 2; ++side) {
        MjDataPtr perturbed = MakeData(model);
        mj_copyData(perturbed.get(), model.get(), data.get());
        std::vector<mjtNum> direction(8, 0);
        direction[column] = side == 0 ? 1e-6 : -1e-6;
        mj_integratePos(model.get(), perturbed->qpos, direction.data(), 1);
        mj_forward(model.get(), perturbed.get());
        losses[side] = mju_dot(bar.data(), perturbed->qfrc_bias, 8);
      }
      mjtNum fd = (losses[0]-losses[1])/2e-6;
      EXPECT_NEAR(gradient[column], fd, 1e-8*std::max(1.0, std::abs(fd)))
          << sample << ":" << column << " analytic=" << gradient[column] << " fd=" << fd;
    }
  }
  for (int row = 0; row < 8; ++row) {
    std::vector<mjtNum> bar(8, 0);
    bar[row] = 1;
    std::vector<mjtNum> gradient(8, 0);
    ASSERT_EQ(mjd_rneBiasPosAdjoint(model.get(), data.get(), bar.data(), gradient.data()), mjADJ_OK);
    for (int column = 0; column < 8; ++column) {
      MjDataPtr plus = MakeData(model);
      MjDataPtr minus = MakeData(model);
      mj_copyData(plus.get(), model.get(), data.get());
      mj_copyData(minus.get(), model.get(), data.get());
      std::vector<mjtNum> direction(8, 0);
      direction[column] = 1e-6;
      mj_integratePos(model.get(), plus->qpos, direction.data(), 1);
      mju_scl(direction.data(), direction.data(), -1, 8);
      mj_integratePos(model.get(), minus->qpos, direction.data(), 1);
      mj_forward(model.get(), plus.get());
      mj_forward(model.get(), minus.get());
      mjtNum fd = (plus->qfrc_bias[row]-minus->qfrc_bias[row])/2e-6;
      EXPECT_NEAR(gradient[column], fd, 1e-8*std::max(1.0, std::abs(fd)))
          << "row=" << row << " column=" << column
          << " analytic=" << gradient[column] << " fd=" << fd;
    }
  }
}

}  // namespace
}  // namespace mujoco
