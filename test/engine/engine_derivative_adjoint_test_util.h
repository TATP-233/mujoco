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

#ifndef MUJOCO_TEST_ENGINE_ENGINE_DERIVATIVE_ADJOINT_TEST_UTIL_H_
#define MUJOCO_TEST_ENGINE_ENGINE_DERIVATIVE_ADJOINT_TEST_UTIL_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include <mujoco/mujoco.h>
#include "test/fixture.h"

namespace mujoco {
namespace adjoint_test {

enum class InputBlock { kQPos, kQVel, kCtrl };

struct StepTangent {
  std::vector<mjtNum> qpos;
  std::vector<mjtNum> qvel;
};

struct Comparison {
  double max_absolute = 0;
  double max_relative = 0;
  double cosine = 0;
  int nonfinite = 0;
};

inline StepTangent StockStepCentralTangent(const mjModel* m, const mjData* d,
                                           InputBlock block, int index, mjtNum eps) {
  MjDataPtr plus(mj_makeData(m));
  MjDataPtr minus(mj_makeData(m));
  mj_copyData(plus.get(), m, d);
  mj_copyData(minus.get(), m, d);
  if (block == InputBlock::kQPos) {
    std::vector<mjtNum> tangent(m->nv, 0);
    tangent[index] = eps;
    mj_integratePos(m, plus->qpos, tangent.data(), 1);
    tangent[index] = -eps;
    mj_integratePos(m, minus->qpos, tangent.data(), 1);
  } else {
    mjtNum* plus_input = block == InputBlock::kQVel ? plus->qvel : plus->ctrl;
    mjtNum* minus_input = block == InputBlock::kQVel ? minus->qvel : minus->ctrl;
    plus_input[index] += eps;
    minus_input[index] -= eps;
  }
  mj_step(m, plus.get());
  mj_step(m, minus.get());
  StepTangent result;
  result.qpos.resize(m->nv);
  result.qvel.resize(m->nv);
  mj_differentiatePos(m, result.qpos.data(), 1, plus->qpos, minus->qpos);
  for (int i = 0; i < m->nv; ++i) {
    result.qpos[i] /= -2 * eps;
    result.qvel[i] = (plus->qvel[i] - minus->qvel[i]) / (2 * eps);
  }
  return result;
}

inline double StepLoss(const StepTangent& value, const std::vector<mjtNum>& bar_qpos,
                       const std::vector<mjtNum>& bar_qvel) {
  double loss = 0;
  for (std::size_t i = 0; i < value.qpos.size(); ++i) {
    loss += value.qpos[i] * bar_qpos[i] + value.qvel[i] * bar_qvel[i];
  }
  return loss;
}

inline Comparison CompareVectors(const std::vector<double>& actual,
                                 const std::vector<double>& expected) {
  Comparison result;
  if (actual.size() != expected.size() || actual.empty()) {
    result.nonfinite = 1;
    return result;
  }
  double dot = 0;
  double actual_norm = 0;
  double expected_norm = 0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) {
      ++result.nonfinite;
      continue;
    }
    double error = std::abs(actual[i] - expected[i]);
    result.max_absolute = std::max(result.max_absolute, error);
    result.max_relative = std::max(result.max_relative,
                                   error / std::max(1e-12, std::abs(expected[i])));
    dot += actual[i] * expected[i];
    actual_norm += actual[i] * actual[i];
    expected_norm += expected[i] * expected[i];
  }
  if (actual_norm > 0 && expected_norm > 0) {
    result.cosine = dot / std::sqrt(actual_norm * expected_norm);
  }
  return result;
}

}  // namespace adjoint_test
}  // namespace mujoco

#endif  // MUJOCO_TEST_ENGINE_ENGINE_DERIVATIVE_ADJOINT_TEST_UTIL_H_
