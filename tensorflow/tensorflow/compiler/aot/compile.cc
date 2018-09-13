#include "tensorflow/compiler/aot/compile.h"
#include "tensorflow/compiler/tf2xla/tf2xla.h"
namespace tensorflow {
namespace tfcompile {
/**
 * \brief Converts the graph into an XLA computation, and compiles the computation.
 *        Called by `tensorflow::tfcompile::Main`.
 *        Convert graph definition to object file that include graph's function
 * 1. [TODO] Fiddling about client
 * 2. Call `ConvertGraphDefToXla`
 * 3. [TODO] Call `CompileXLA`
 */
Status CompileGraph(const GraphDef& graph_def, const tf2xla::Config& config,
                    const MainFlags& flags, CompileResult* compile_result) {
  namespace gpu = perftools::gputools;
  gpu::Platform* cpu_platform =
      gpu::MultiPlatformManager::PlatformWithName("Host").ValueOrDie();
  xla::CompileOnlyClient* client =
      xla::ClientLibrary::GetOrCreateCompileOnlyClient(cpu_platform)
          .ValueOrDie();
  xla::Computation computation;
  TF_RETURN_IF_ERROR(
      ConvertGraphDefToXla(graph_def, config, client, &computation));
	// ...
	// Do something
	// ...
  return CompileXla(client, computation, aot_opts, compile_result);
}
}  // namespace tfcompile
}  // namespace tensorflow

