/** \file
 */
/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_RESHAPE_MOVER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_RESHAPE_MOVER_H_

#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

/**
 * Google docs:
 * > A pass which moves Reshapes and Transposes to let later passes combine them.
 * > This now only moves them outputward across elementwise ops all whose operands
 * > are equivalent Reshapes or Transposes, but in future could potentially move
 * > them inputward also.
 *
 *
 * Implementation note from Google doc:
 *
 * The general idea behind this pass is that we're converting from this:
 * ```
 *   %param.A = OldShape
 *   %param.B = OldShape
 *   %reshape.A = NewShape reshape(%param.A)
 *   %reshape.B = NewShape reshape(%param.B)
 *   %instruction = NewShape instruction(%reshape.A, %reshape.B)
 * ```
 * To this:
 * ```
 *   %param.A = OldShape
 *   %param.B = OldShape
 *   %instruction = OldShape instruction(%param.A, %param.B)
 *   %reshape = NewShape reshape(%instruction)
 * ```
 *
 * Where the instruction must be elementwise, and both reshapes and transposes
 * are moved.
 *
 * Most elementwise instructions support implicit broadcast of scalar operands,
 * but select is a special-case.  The signature is Select(Pred, A, B), and the
 * only implicit scalar broadcast is on Pred, not on A or B. Since reshapes or
 * transposes to a scalar should be cheap, we simply never move them.
*/
class ReshapeMover : public HloPassInterface {
 public:
  /**
   * Return internal name "reshape-mover"
   */
  tensorflow::StringPiece name() const override { return "reshape-mover"; }

  StatusOr<bool> Run(HloModule* module) override;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_RESHAPE_MOVER_H_
