#include "DPU.hpp"
#include <vart/runner.hpp>
#include <xir/graph/graph.hpp>
#include <xir/tensor/tensor.hpp>
#include <xir/util/data_type.hpp>
#include <vector>

/**
 * @brief get_dpu_subraph 
 *
 * Function to get subgraph (Code from Vitis AI Github) 
 *
 * @param *graph[inout]: dpu subgraph 
 *  
 */
std::vector<const xir::Subgraph*> get_dpu_subgraph(const xir::Graph* graph) {
  auto root = graph->get_root_subgraph();
  auto children = root->children_topological_sort();
  auto ret = std::vector<const xir::Subgraph*>();
  for (auto c : children) {
    CHECK(c->has_attr("device"));
    auto device = c->get_attr<std::string>("device");
    if (device == "DPU") {
      ret.emplace_back(c);
    }
  }
  return ret;
}