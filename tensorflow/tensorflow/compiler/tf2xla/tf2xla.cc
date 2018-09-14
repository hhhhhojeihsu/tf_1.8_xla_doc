#include "tensorflow/compiler/tf2xla/tf2xla.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/graph/graph_constructor.h"
namespace tensorflow {
const char* const kArgOp = "_Arg";
const char* const kRetvalOp = "_Retval";
const char* const kFeedIdAttr = "_feed_id";
const char* const kFetchIdAttr = "_fetch_id";
const char* const kShapeAttr = "_shape";
const char* const kDebugNameAttr = "_debug_name";
namespace {
/**
 * 1. Generate a `_Arg` node and remove the original placeholder node by removing edge and add a new edge to `_Arg` node.
 * 2. For each fetch tensor, create a `_Retval` node and add it to the end.
 * 3. Do BFS from `_Retval` nodes and remove all unreachable nodes. Thus removing placeholders creating by `tensorflow::AddPlaceholdersForFeeds()`
 * 4. Connect nodes without input edge with `Source` node by using control edge. And `Sink` node to nodes without output edge.
 *
 * Google Docs:
 * > RewriteAndPruneGraph identifies input and output edges (named by the feed and
 * > fetch ids respectively), and rewrites the edges so that inputs flow from _Arg
 * > nodes, and outputs flow to _Retval nodes.  This allows the symbolic graph
 * > execution to know the input and output args for the generated function.
 */
Status RewriteAndPruneGraph(
    Graph* graph, const tf2xla::Config& config,
    const std::unordered_map<string, string>& feed_remapping) {
  NodeMap node_map;
  for (Node* n : graph->nodes()) {
    node_map[n->name()] = n;
  }
  TF_RETURN_IF_ERROR(
      AddArgNodes(graph, node_map, config.feed(), feed_remapping));
  std::unordered_set<const Node*> retval_nodes;
  TF_RETURN_IF_ERROR(
      AddRetvalNodes(graph, node_map, config.fetch(), &retval_nodes));
  VLOG(2) << "Post rewrite: "
          << dump_graph::DumpGraphToFile("tf2xla_post_rewrite", *graph);
  PruneForReverseReachability(graph, retval_nodes);
  FixupSourceAndSinkEdges(graph);
  VLOG(2) << "Post prune: "
          << dump_graph::DumpGraphToFile("tfcompile_post_prune", *graph);
  // Sanity-check, to make sure the feeds and fetches still exist post-pruning.
  std::set<string> missing_feeds, missing_fetches;
  for (const tf2xla::Feed& feed : config.feed()) {
    missing_feeds.insert(TensorIdToString(feed.id()));
  }
  for (const tf2xla::Fetch& fetch : config.fetch()) {
    missing_fetches.insert(TensorIdToString(fetch.id()));
  }
  for (const Node* n : graph->op_nodes()) {
    if (n->type_string() == kArgOp) {
      string feed_id;
      TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), kFeedIdAttr, &feed_id));
      if (missing_feeds.erase(feed_id) == 0) {
        return errors::Aborted(kArgOp,
                               " node found with unknown feed id: ", feed_id);
      }
    } else if (n->type_string() == kRetvalOp) {
      string fetch_id;
      TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), kFetchIdAttr, &fetch_id));
      if (missing_fetches.erase(fetch_id) == 0) {
        return errors::Aborted(kRetvalOp,
                               " node found with unknown fetch id: ", fetch_id);
      }
    }
  }
  if (!missing_feeds.empty() || !missing_fetches.empty()) {
    return errors::Aborted(
        "Post graph-pruning",
        ", missing feeds: ", str_util::Join(missing_feeds, ", "),
        ", missing fetches: ", str_util::Join(missing_fetches, ", "));
  }
  return Status::OK();
}
/**
 * \brief Converts the TensorFlow graph into an XLA computation, by executing the
 *        graph symbolically, with each op building up the XLA HLO.
 * 1. [UNCLEAR] `tensorflow::XlaOpRegistry::RegisterCompilationKernels()`
 * 2. Traverse nodes and record the index of assigned device name (hard code with "/device:XLA_CPU_JIT")
 * 3. Generate XLA Arguments by `_Arg` nodes (Ensure the index and type attrs of each nodes initialized correctly first)
 * 4. Compile graph to XLA UserComputation
 *   1. Set options for object `XlaCompiler`
 *   2. Call `CompileGraph`, method of object `XlaCompiler`
 * 5. Check compilation result. Throw error if there's a generated function returns constant value (Result of invalid config).
 */
Status ConvertGraphToXla(std::unique_ptr<Graph> graph, xla::Client* client,
                         xla::Computation* computation) {
  XlaOpRegistry::RegisterCompilationKernels();
  // ...
  // 2.
  // ...
  std::vector<XlaCompiler::Argument> xla_args;
  TF_RETURN_IF_ERROR(CreateXlaArgs(*graph, &xla_args));

  // Compile the graph into an XLA computation.
  XlaCompiler::Options compiler_options;
  // ...
  // 4-1.
  // ...
  XlaCompiler compiler(compiler_options);
  XlaCompiler::CompilationResult result;
  TF_RETURN_IF_ERROR(compiler.CompileGraph(XlaCompiler::CompileOptions(),
                                           "tfcompile", std::move(graph),
                                           xla_args, &result));
  // ...
  // 5.
  // ...
  return Status::OK();
}
/**
 * \brief InitGraph creates a graph based on the graph_def, that may then be
 *        convert to an xla::Computation via ConvertGraphToXla.
 *
 * Google Doc:
 * > The graph is rewritten with _Arg and _Retval nodes, representing the inputs
 * > and outputs of the function that will be compiled.  Each feed id causes a new
 * > _Arg node to be created, where we first collect all existing edges pointing
 * > from the named node's output index, and then rewrite them to point from that
 * > _Arg node instead.  Each fetch id causes a new _Retval node to be created,
 * > with a new edge pointing from the named node's output index to that _Retval
 * > node.
 *
 * 1. `tensorflow::ValidateConfig()`
 * 2. Generate mapping for user defined function.
 *   1. gtl::FlatMap<string, std::unique_ptr<FunctionDefAndOpRegistration>> function_defs_
 *   2. gtl::FlatMap<string, string> func_grad_
 * 3. `tensorflow::AddPlaceholdersForFeeds()`
 * 4. `tensorflow::PruneGraphDefInfo()`
 * 5. `tensorflow::AddDefaultAttrsToGraphDef()`
 * 6. `tensorflow::ConvertGraphDefToGraph()`
 * 7. `tensorflow::RewriteAndPruneGraph()`
 */
Status InitGraph(const GraphDef& graph_def, const tf2xla::Config& config,
                 std::unique_ptr<Graph>* graph) {
  TF_RETURN_IF_ERROR(ValidateConfig(config));
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), graph_def.library());
  std::unique_ptr<Graph> g(new Graph(flib_def));
  // Replace references to fed tensors with references to newly added
  // placeholders.
  GraphDef first_copy_def = graph_def;
  // Maps from name:port of a feed to the name:port of the placeholder to use.
  std::unordered_map<string, string> feed_remapping;
  TF_RETURN_IF_ERROR(AddPlaceholdersForFeeds(config, g->op_registry(),
                                             &feed_remapping, &first_copy_def));
  // Prune the GraphDef first so that unknown ops that we aren't compiling get
  // filtered out.
  GraphDef second_copy_def;
  TF_RETURN_IF_ERROR(
      PruneGraphDefInto(config, first_copy_def, &second_copy_def));
  TF_RETURN_IF_ERROR(AddDefaultAttrsToGraphDef(
      &second_copy_def, *g->op_registry(), /*node_offset=*/0));
  TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(GraphConstructorOptions(),
                                            second_copy_def, g.get()));
  TF_RETURN_IF_ERROR(RewriteAndPruneGraph(g.get(), config, feed_remapping));
  *graph = std::move(g);
  return Status::OK();
}
}  // namespace
/**
 * \brief Convert tensorflow graph_def to XLA UserComputation
 * 1. Call `tensorflow::anonymous_namespace{tf2xla.cc}::InitGraph`
 * 2. Call `tensorflow::anonymous_namespace{tf2xla.cc}::ConvertGraphToXla`
 */
Status ConvertGraphDefToXla(const GraphDef& graph_def,
                            const tf2xla::Config& config, xla::Client* client,
                            xla::Computation* computation) {
  std::unique_ptr<Graph> graph;
  TF_RETURN_IF_ERROR(InitGraph(graph_def, config, &graph));
  TF_RETURN_IF_ERROR(ConvertGraphToXla(std::move(graph), client, computation));
  return Status::OK();
}
}  // namespace tensorflow

