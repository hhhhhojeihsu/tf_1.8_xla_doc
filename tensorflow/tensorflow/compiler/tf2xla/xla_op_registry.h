#ifndef TENSORFLOW_COMPILER_TF2XLA_XLA_OP_REGISTRY_H_
#define TENSORFLOW_COMPILER_TF2XLA_XLA_OP_REGISTRY_H_
namespace tensorflow {
class XlaOpRegistry {
 public:
  /** \brief Registers all JIT kernels on JIT devices, if not already registered.
   *         Does nothing otherwise.
	 */
  static void RegisterCompilationKernels();
};
