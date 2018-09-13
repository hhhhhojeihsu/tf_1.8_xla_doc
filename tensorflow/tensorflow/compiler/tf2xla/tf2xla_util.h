#ifndef TENSORFLOW_COMPILER_TF2XLA_TF2XLA_UTIL_H_
#define TENSORFLOW_COMPILER_TF2XLA_TF2XLA_UTIL_H_
namespace tensorflow {
/**
 * \brief Check settings of input config(feed & fetch). For example, if `id`, `shape` are legal, duplicated names...
 */
Status ValidateConfig(const tf2xla::Config& config);
/**
 * \brief Generate corresponding placeholder for each feed tensor. Traverse each inputs of
 *        each nodes. Replace feeds to corresponding placeholders
 * Modifies <graph_def> to include placeholders for each fed tensor, and
 * update references to the fed tensors to refer to the placeholders.
 * The existing nodes referenced by the feeds are not removed or modified
 * (except where their input edges are modified by the replacement of other
 * feeds).
 */
Status AddPlaceholdersForFeeds(
    const tf2xla::Config& config, const OpRegistryInterface* op_registry,
    std::unordered_map<string, string>* feed_remapping, GraphDef* graph_def);
/**
 * \brief Remove source nodes of feed tensor. Because they are removed at `tensorflow::AddPlaceholdersForFeeds`
 * Returns in <out> a copy of <in>, pruned to only include fetches from
 * <config>.
 */
Status PruneGraphDefInto(const tf2xla::Config& config, const GraphDef& in,
                         GraphDef* out);
#endif  // TENSORFLOW_COMPILER_TF2XLA_TF2XLA_UTIL_H_

