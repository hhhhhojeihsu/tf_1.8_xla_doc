#ifndef TENSORFLOW_GRAPH_GRAPH_CONSTRUCTOR_H_
namespace tensorflow {
/**
 * \brief Convert `GraphDef` defined protobuf to type `Graph`
 */
extern Status ConvertGraphDefToGraph(const GraphConstructorOptions& opts,
                                     const GraphDef& gdef, Graph* g);
}  // namespace tensorflow
#endif  // TENSORFLOW_GRAPH_GRAPH_CONSTRUCTOR_H_
