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

#include <mujoco/mjadjoint.h>

#include <cstring>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mujoco/mujoco.h>
#include "test/fixture.h"

namespace mujoco {
namespace {

using ::testing::NotNull;
using ::testing::StrEq;

struct AdjContextDeleter {
  void operator()(mjAdjContext* ctx) const { mj_deleteAdjContext(ctx); }
};
using AdjContextPtr = std::unique_ptr<mjAdjContext, AdjContextDeleter>;

class AdjointTest : public MujocoTest {
 protected:
  void SetUp() override {
    mj_resetCallbacks();
    static constexpr char xml[] = R"(
    <mujoco>
      <option timestep=".002" solver="Newton" integrator="Euler">
        <flag island="disable"/>
      </option>
      <worldbody>
        <body name="hinge">
          <joint name="hinge" type="hinge" axis="0 1 0"/>
          <geom type="capsule" size=".05 .2" mass="1"/>
        </body>
        <body name="slide" pos="1 0 0">
          <joint name="slide" type="slide" axis="1 0 0"/>
          <geom type="sphere" size=".1" mass="1"/>
        </body>
      </worldbody>
      <actuator>
        <motor joint="hinge" gear="2"/>
        <motor joint="slide" gear="3"/>
      </actuator>
    </mujoco>)";
    char error[1024] = "";
    model_ = LoadModelFromString(xml, error, sizeof(error));
    ASSERT_THAT(model_.get(), NotNull()) << error;
    data_ = MakeData(model_);
    ASSERT_THAT(data_.get(), NotNull());
  }

  MjModelPtr model_;
  MjDataPtr data_;
};

TEST_F(AdjointTest, AbiAndStatusStrings) {
  EXPECT_EQ(mjd_adjointVersion(), mjADJOINT_ABI_VERSION);
  EXPECT_NE(mjd_adjointCapabilities() & mjADJOINT_CAP_STEP_VJP, 0);
  EXPECT_NE(mjd_adjointCapabilities() & mjADJOINT_CAP_EULER, 0);
  EXPECT_NE(mjd_adjointCapabilities() & mjADJOINT_CAP_IMPLICITFAST, 0);
  EXPECT_EQ(mjd_adjointCapabilities() & mjADJOINT_CAP_CONTACTS, 0);
  EXPECT_THAT(mjd_adjointStatusString(mjADJ_OK), StrEq("OK"));
  EXPECT_THAT(mjd_adjointStatusString(mjADJ_STOP_GRADIENT), StrEq("STOP_GRADIENT"));
  EXPECT_THAT(mjd_adjointStatusString(mjADJ_UNSUPPORTED), StrEq("UNSUPPORTED"));
  EXPECT_THAT(mjd_adjointStatusString(mjADJ_NON_SMOOTH), StrEq("NON_SMOOTH"));
  EXPECT_THAT(mjd_adjointStatusString(mjADJ_NOT_CONVERGED), StrEq("NOT_CONVERGED"));
  EXPECT_THAT(mjd_adjointStatusString(mjADJ_INVALID_ARGUMENT), StrEq("INVALID_ARGUMENT"));
  EXPECT_THAT(mjd_adjointStatusString(mjADJ_ENGINE_ERROR), StrEq("ENGINE_ERROR"));
  EXPECT_THAT(mjd_adjointStatusString(-1), StrEq("UNKNOWN"));
}

TEST_F(AdjointTest, ContextLifecycleAndReserve) {
  ASSERT_EQ(mjd_checkAdjointModel(model_.get()), mjADJ_OK);
  EXPECT_EQ(mj_makeAdjContext(nullptr), nullptr);
  AdjContextPtr ctx = AdjContextPtr(mj_makeAdjContext(model_.get()));
  ASSERT_THAT(ctx.get(), NotNull());
  EXPECT_EQ(mj_reserveAdjContext(ctx.get(), 4, 12, 24, 16), mjADJ_OK);
  EXPECT_EQ(mj_reserveAdjContext(ctx.get(), -1, 0, 0, 0), mjADJ_INVALID_ARGUMENT);
  EXPECT_EQ(mj_reserveAdjContext(nullptr, 0, 0, 0, 0), mjADJ_INVALID_ARGUMENT);
  mj_deleteAdjContext(nullptr);
}

TEST_F(AdjointTest, RejectsEveryOptionGateWithoutCommit) {
  struct OptionCase {
    const char* name;
    int solver;
    int integrator;
    int noslip;
    int disableflags;
    int enableflags;
  };
  const OptionCase cases[] = {
      {"solver", mjSOL_CG, mjINT_EULER, 0, mjDSBL_ISLAND, 0},
      {"integrator", mjSOL_NEWTON, mjINT_RK4, 0, mjDSBL_ISLAND, 0},
      {"noslip", mjSOL_NEWTON, mjINT_EULER, 1, mjDSBL_ISLAND, 0},
      {"island", mjSOL_NEWTON, mjINT_EULER, 0, 0, 0},
      {"diagexact", mjSOL_NEWTON, mjINT_EULER, 0, mjDSBL_ISLAND, mjENBL_DIAGEXACT},
      {"sleep", mjSOL_NEWTON, mjINT_EULER, 0, mjDSBL_ISLAND, mjENBL_SLEEP},
  };
  AdjContextPtr ctx = AdjContextPtr(mj_makeAdjContext(model_.get()));
  ASSERT_THAT(ctx.get(), NotNull());
  std::vector<mjtNum> qpos(model_->nq), qvel(model_->nv), ctrl(model_->nu);
  for (const OptionCase& test : cases) {
    SCOPED_TRACE(test.name);
    model_->opt.solver = test.solver;
    model_->opt.integrator = test.integrator;
    model_->opt.noslip_iterations = test.noslip;
    model_->opt.disableflags = test.disableflags;
    model_->opt.enableflags = test.enableflags;
    EXPECT_EQ(mjd_checkAdjointModel(model_.get()), mjADJ_UNSUPPORTED);
  }
}

TEST_F(AdjointTest, RejectsActuatorAndTopologyGates) {
  EXPECT_EQ(mjd_checkAdjointModel(model_.get()), mjADJ_OK);
  model_->actuator_dyntype[0] = mjDYN_FILTER;
  EXPECT_EQ(mjd_checkAdjointModel(model_.get()), mjADJ_UNSUPPORTED);
  model_->actuator_dyntype[0] = mjDYN_NONE;
  model_->actuator_gaintype[0] = mjGAIN_AFFINE;
  EXPECT_EQ(mjd_checkAdjointModel(model_.get()), mjADJ_UNSUPPORTED);
  model_->actuator_gaintype[0] = mjGAIN_FIXED;
  model_->actuator_biastype[0] = mjBIAS_MUSCLE;
  EXPECT_EQ(mjd_checkAdjointModel(model_.get()), mjADJ_UNSUPPORTED);
  model_->actuator_biastype[0] = mjBIAS_NONE;
  model_->actuator_delay[0] = .01;
  EXPECT_EQ(mjd_checkAdjointModel(model_.get()), mjADJ_UNSUPPORTED);
  model_->actuator_delay[0] = 0;
  model_->actuator_trntype[0] = mjTRN_SITE;
  EXPECT_EQ(mjd_checkAdjointModel(model_.get()), mjADJ_UNSUPPORTED);
}

TEST_F(AdjointTest, RejectsCallbacksAndRestoresGate) {
  EXPECT_EQ(mjd_checkAdjointModel(model_.get()), mjADJ_OK);
  mjcb_control = +[](const mjModel*, mjData*) {};
  EXPECT_EQ(mjd_checkAdjointModel(model_.get()), mjADJ_UNSUPPORTED);
  mjcb_control = nullptr;
  EXPECT_EQ(mjd_checkAdjointModel(model_.get()), mjADJ_OK);
}

TEST_F(AdjointTest, UnsupportedStepDoesNotCommitOrWriteGradients) {
  AdjContextPtr ctx = AdjContextPtr(mj_makeAdjContext(model_.get()));
  ASSERT_THAT(ctx.get(), NotNull());
  ASSERT_EQ(mj_reserveAdjContext(ctx.get(), 1, 1, model_->nv, 1), mjADJ_OK);
  model_->opt.integrator = mjINT_RK4;
  std::vector<mjtNum> qpos_before(data_->qpos, data_->qpos + model_->nq);
  std::vector<mjtNum> qvel_before(data_->qvel, data_->qvel + model_->nv);
  std::vector<mjtNum> ctrl_before(data_->ctrl, data_->ctrl + model_->nu);
  std::vector<mjtNum> bar_qpos(model_->nv, .25);
  std::vector<mjtNum> bar_qvel(model_->nv, -.5);
  std::vector<mjtNum> grad_qpos(model_->nv, 11);
  std::vector<mjtNum> grad_qvel(model_->nv, 12);
  std::vector<mjtNum> grad_ctrl(model_->nu, 13);
  EXPECT_EQ(mjd_stepAdjoint(model_.get(), data_.get(), ctx.get(), bar_qpos.data(),
                            bar_qvel.data(), grad_qpos.data(), grad_qvel.data(),
                            grad_ctrl.data()), mjADJ_UNSUPPORTED);
  EXPECT_EQ(std::memcmp(qpos_before.data(), data_->qpos, model_->nq * sizeof(mjtNum)), 0);
  EXPECT_EQ(std::memcmp(qvel_before.data(), data_->qvel, model_->nv * sizeof(mjtNum)), 0);
  EXPECT_EQ(std::memcmp(ctrl_before.data(), data_->ctrl, model_->nu * sizeof(mjtNum)), 0);
  EXPECT_EQ(grad_qpos, std::vector<mjtNum>(model_->nv, 11));
  EXPECT_EQ(grad_qvel, std::vector<mjtNum>(model_->nv, 12));
  EXPECT_EQ(grad_ctrl, std::vector<mjtNum>(model_->nu, 13));
}

TEST_F(AdjointTest, RejectsIllegalOverlapBeforeTouchingContext) {
  AdjContextPtr ctx = AdjContextPtr(mj_makeAdjContext(model_.get()));
  ASSERT_THAT(ctx.get(), NotNull());
  std::vector<mjtNum> values(model_->nv, 1);
  std::vector<mjtNum> other(model_->nv, 2);
  std::vector<mjtNum> ctrl(model_->nu, 3);
  data_->nefc = 1;
  EXPECT_EQ(mjd_stepAdjoint(model_.get(), data_.get(), ctx.get(), values.data(),
                            other.data(), values.data(), other.data(), ctrl.data()),
            mjADJ_ENGINE_ERROR);
  data_->nefc = 0;
  EXPECT_EQ(mj_reserveAdjContext(ctx.get(), 0, 0, 0, 0), mjADJ_OK);
  model_->opt.integrator = mjINT_RK4;
  EXPECT_EQ(mjd_stepAdjoint(model_.get(), data_.get(), ctx.get(), values.data(),
                            other.data(), values.data(), other.data(), ctrl.data()),
            mjADJ_UNSUPPORTED);
  EXPECT_EQ(mjd_stepAdjoint(model_.get(), data_.get(), ctx.get(), values.data(),
                            other.data(), other.data(), values.data(), ctrl.data()),
            mjADJ_INVALID_ARGUMENT);
  EXPECT_EQ(mjd_stepAdjoint(model_.get(), data_.get(), nullptr, values.data(),
                            other.data(), values.data(), other.data(), ctrl.data()),
            mjADJ_INVALID_ARGUMENT);
}

TEST_F(AdjointTest, RejectsContextFromDifferentSparseLayout) {
  AdjContextPtr ctx = AdjContextPtr(mj_makeAdjContext(model_.get()));
  ASSERT_THAT(ctx.get(), NotNull());
  ASSERT_EQ(mj_reserveAdjContext(ctx.get(), 0, 0, 0, 0), mjADJ_OK);
  std::vector<mjtNum> values(model_->nv, 1);
  std::vector<mjtNum> other(model_->nv, 2);
  std::vector<mjtNum> ctrl(model_->nu, 3);
  int saved_parent = model_->dof_parentid[1];
  model_->dof_parentid[1] = saved_parent == -1 ? 0 : -1;
  EXPECT_EQ(mjd_stepAdjoint(model_.get(), data_.get(), ctx.get(), values.data(),
                            other.data(), values.data(), other.data(), ctrl.data()),
            mjADJ_INVALID_ARGUMENT);
  model_->dof_parentid[1] = saved_parent;
}

}  // namespace
}  // namespace mujoco
