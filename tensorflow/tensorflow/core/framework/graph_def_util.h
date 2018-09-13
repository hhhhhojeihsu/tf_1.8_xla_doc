#ifndef TENSORFLOW_FRAMEWORK_GRAPH_DEF_UTIL_H_
#define TENSORFLOW_FRAMEWORK_GRAPH_DEF_UTIL_H_
namespace tensorflow {
// Forward declare proto so that it's symbols can be removed from .so exports
class GraphDef;
Status ValidateExternalGraphDefSyntax(const GraphDef& graph_def);
/**
 * \brief Assign each `NodeDefs` of `GraphDef` default values
 * Google Docs:
 * > Adds default attributes to NodeDefs in 'graph_def' starting
 * > from the 'node_offset' node in 'graph_def'.
 * >
 * > Default attributes are defined by 'op_registry'.
 * >
 * > Returns OK on success, an error if 'graph_def' has a NodeDef
 * > that cannot be found in 'op_registry'.
 * >
 * > REQUIRES: 'graph_def' and 'op_registry' are not nullptr.
 *
 */
Status AddDefaultAttrsToGraphDef(GraphDef* graph_def,
                                 const OpRegistryInterface& op_registry,
                                 int node_offset);
}  // namespace tensorflow
#endif  // TENSORFLOW_FRAMEWORK_GRAPH_DEF_UTIL_H_
