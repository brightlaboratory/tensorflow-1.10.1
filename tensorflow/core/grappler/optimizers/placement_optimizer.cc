
#include "tensorflow/core/grappler/optimizers/placement_optimizer.h"

#include <set>
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/cost_graph.pb.h"
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/grappler/costs/analytical_cost_estimator.h"
using namespace std;

namespace tensorflow {
namespace grappler {

#define MIN_EXECUTION_TIME 1000

Status PlacementOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,
                                    GraphDef* optimized_graph) {
  VLOG(0) << "Optimize Grappler item: id=" << item.id;
  PrintDeviceStats(cluster);

  AnalyticalCostEstimator estimator(cluster, true);
  Status initStatus = estimator.Initialize(item);

  if (initStatus != Status::OK()) {
    return initStatus;
  }

  CostGraphDef cost_graph;
  Costs summary;
  Status predictStatus =
      estimator.PredictCosts(item.graph, &cost_graph, &summary);

  if (predictStatus != Status::OK()) {
    return predictStatus;
  }

  VLOG(0) << "summary.execution_time: " << summary.execution_time << "\n";

  if (summary.execution_time >= Costs::Duration(MIN_EXECUTION_TIME)) {
  }

  // PrintCostStats(item, cost_graph);

  CreateDefaultPlacement(item.graph, optimized_graph);
  return Status::OK();
}

void PlacementOptimizer::CreateDefaultPlacement(const GraphDef& graph_def,
                                                GraphDef* optimized_graph) {
  set<string> devices;
  set<string>::iterator it1;

  for (int i = 0; i < graph_def.node_size(); i++) {
    const NodeDef& node = graph_def.node(i);
    devices.insert(node.device());
  }

  VLOG(0) << "number_of_distinct_devices: " << devices.size() << "\n";
  for (it1 = devices.begin(); it1 != devices.end(); it1++) {
    VLOG(0) << "mapped_device: " << *it1 << "\n";
  }

  if (devices.size() > 0) {
    string default_device = *devices.begin();

    for (const NodeDef& node : graph_def.node()) {
      NodeDef* new_node = optimized_graph->add_node();
      *new_node = node;
      new_node.set_device(default_device);
    }

    *optimized_graph->mutable_versions() = graph_def.versions();

    VLOG(0) << "All ops mapped to: " << default_device << "\n";
  } else {
    *optimized_graph = graph_def;
    VLOG(0) << "The original graph is unmodified\n";
  }
}

void PlacementOptimizer::PrintDeviceStats(Cluster* cluster) {
  const DeviceSet* device_set = cluster->GetDeviceSet();
  const std::vector<Device*>& devices = device_set->devices();

  VLOG(0) << "Number of devices: " << devices.size();
  for (int i = 0; i < devices.size(); i++) {
    VLOG(0) << devices.at(i)->name()
            << " 's attributes: " << devices.at(i)->DebugString() << "\n";
  }
}

void PlacementOptimizer::PrintCostStats(const GrapplerItem& item,
                                        CostGraphDef& cost_graph) {
  std::unordered_map<string, const CostGraphDef::Node*> name_to_cost;

  for (int i = 0; i < cost_graph.node_size(); i++) {
    const CostGraphDef::Node& cnode = cost_graph.node(i);
    name_to_cost[cnode.name()] = &cnode;
  }

  const GraphDef& graph_def = item.graph;
  for (int i = 0; i < graph_def.node_size(); i++) {
    const NodeDef& node = graph_def.node(i);

    VLOG(0) << "Node: " << node.name() << "device: " << node.device() << "\n";

    auto it = name_to_cost.find(node.name());
    const CostGraphDef::Node* cost_node;
    if (it != name_to_cost.end()) {
      cost_node = it->second;
    } else {
      cost_node = NULL;
    }

    if (cost_node) {
      VLOG(0) << "Op: " << node.name()
              << " max_memory_size: " << cost_node->max_memory_size()
              << " memory_time: " << cost_node->memory_time()
              << " compute_time: " << cost_node->compute_time()
              << " compute_cost: " << cost_node->compute_cost() << "\n";
    } else {
      VLOG(0) << "Op: " << node.name() << " has no cost estimate\n";
    }
  }
}

void PlacementOptimizer::Feedback(Cluster* cluster, const GrapplerItem& item,
                                  const GraphDef& optimized_graph,
                                  double result) {
  // nothing to be done
}

}  // namespace grappler
}  // namespace tensorflow