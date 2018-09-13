#include "tensorflow/compiler/aot/compile.h"

namespace tensorflow {
namespace tfcompile {
/**
 * Called by `main`
 * 1. Process config(generate config & graph_def)
 * 2. Call `CompileGraph`
 * 3. [TODO] Generate output (object, header, etc)
 */
Status Main(const MainFlags& flags) {
  tf2xla::Config config; //! Graph configuration
  GraphDef graph_def; //! Graph definition
  CompileResult compile_result;
	// ...
  // Process config.
	// ...
  TF_RETURN_IF_ERROR(CompileGraph(graph_def, config, flags, &compile_result));
  // Write output files.
  Env* env = Env::Default();
  const std::vector<char>& obj = compile_result.aot->object_file_data();
  TF_RETURN_IF_ERROR(WriteStringToFile(env, flags.out_function_object,
                                       StringPiece(obj.data(), obj.size())));
  CodegenOpts codegen_opts;
  codegen_opts.gen_name_to_index = flags.gen_name_to_index;
  codegen_opts.gen_program_shape = flags.gen_program_shape;
  codegen_opts.target_triple = flags.target_triple;
  if (flags.cpp_class.empty()) {
    return errors::InvalidArgument("Must specify --cpp_class");
  }
  TF_RETURN_IF_ERROR(ParseCppClass(flags.cpp_class, &codegen_opts.class_name,
                                   &codegen_opts.namespaces));
  MetadataResult metadata_result;
  TF_RETURN_IF_ERROR(
      GenerateMetadata(codegen_opts, compile_result, &metadata_result));
  TF_RETURN_IF_ERROR(WriteStringToFile(env, flags.out_metadata_object,
                                       metadata_result.object_file_data));
  string header;
  TF_RETURN_IF_ERROR(GenerateHeader(codegen_opts, config, compile_result,
                                    metadata_result, &header));
  TF_RETURN_IF_ERROR(WriteStringToFile(env, flags.out_header, header));
  return Status::OK();
}
}  // end namespace tfcompile
}  // end namespace tensorflow
/**
 *  Main entry point of tfcompile:
 *  1. Something not that important ...
 *  2. Process configuration
 *  3. Call `tensorflow::tfcompile::Main`
 */
int main(int argc, char** argv) {
  tensorflow::tfcompile::MainFlags flags;
	// ...
	// Flag initialization
	// ...
  tensorflow::Status status = tensorflow::tfcompile::Main(flags);
  return 0;
}
