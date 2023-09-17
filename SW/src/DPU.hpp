#include <xir/graph/graph.hpp>
#include <xir/tensor/tensor.hpp>
#include <xir/util/data_type.hpp>
#include <vector>

//function to get subgraph (Code from Vitis AI Github)
std::vector<const xir::Subgraph*> get_dpu_subgraph(const xir::Graph* graph);