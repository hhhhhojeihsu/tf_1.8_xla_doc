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
#include "tensorflow/compiler/xla/client/compile_only_client.h"
#include "llvm/ADT/Triple.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
namespace xla {
/**
 * \brief Compiles a list of computations for ahead-of-time execution.  This is
 *        intended for use in static compilation. The |options| parameter describes
 *        the target for which the compiler should emit code.
 * 1. Create a vector of `CompileOnlyService::AotComputationInstance`
 * 2. Convert all the `xla::CompileOnlyClient::AotComputationInstance` objects (basically represents an [`xla::Computation`](https://hhhhhojeihsu.github.io/tensorflow_1.8_woboq/tensorflow_1.8_xla/tensorflow/tensorflow/compiler/xla/client/computation.h.html#xla::Computation) object) into the `xla::CompileOnlyService::AotComputationInstance` objects.
 * 3. Call `xla::CompileOnlyService::CompileAheadOfTime`
 */
StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
CompileOnlyClient::CompileAheadOfTime(
    const tensorflow::gtl::ArraySlice<AotComputationInstance> computations,
    const AotCompilationOptions& options) {
  std::vector<CompileOnlyService::AotComputationInstance> service_instances;
  service_instances.reserve(computations.size());
  for (const AotComputationInstance& instance : computations) {
    service_instances.push_back({});
    CompileOnlyService::AotComputationInstance& service_instance =
        service_instances.back();
    TF_RET_CHECK(instance.computation != nullptr);
    service_instance.computation = instance.computation->handle();
    service_instance.argument_layouts = instance.argument_layouts;
    service_instance.result_layout = instance.result_layout;
  }
  return compiler_service_->CompileAheadOfTime(service_instances, options);
}
int64 CompileOnlyClient::PointerSizeForTriple(tensorflow::StringPiece triple) {
  llvm::Triple llvm_triple(
      llvm::Triple::normalize(llvm::StringRef(triple.data(), triple.size())));
  if (llvm_triple.isArch64Bit()) {
    return 8;
  } else if (llvm_triple.isArch32Bit()) {
    return 4;
  } else {
    CHECK(llvm_triple.isArch16Bit());
    return 2;
  }
}
}  // namespace xla

