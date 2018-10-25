
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
  PrintGrapplerItemStats(item);

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
    VLOG(0) << "Invoking CreateDefaultPlacement\n";
    CreateDefaultPlacement(cluster, item.graph, optimized_graph);
  } else {
    VLOG(0) << "Returning the same graph\n";
    *optimized_graph = item.graph;
  }

  // PrintCostStats(item, cost_graph);
  return Status::OK();
}

void PlacementOptimizer::CreateDefaultPlacement(Cluster* cluster,
                                                const GraphDef& graph_def,
                                                GraphDef* optimized_graph) {
  set<string> devices = GetMappedDevices(graph_def);
  int MIN_DEVICES = 2;  // CPU + at least 1 GPU
  set<string> pinned_devices = GetPinnedDeviceStrings(devices);
  string default_device =
      GetDefaultDevice(cluster->GetDeviceNames(), pinned_devices);

  if (default_device.empty()) {
    VLOG(0) << "There are no non-CPU devices to map the Ops to\n";
    *optimized_graph = graph_def;
  } else {
    for (const NodeDef& node : graph_def.node()) {
      NodeDef* new_node = optimized_graph->add_node();
      *new_node = node;

      if (!new_node->device().empty()) {
        if ((pinned_devices.find(new_node->device()) == pinned_devices.end()) &&
            (new_node->device() != default_device)) {
          VLOG(0) << "node_remapping of " << new_node->name() << " from "
                  << new_node->device() << " to " << default_device << "\n";
          new_node->set_device(default_device);
        }
      }
    }

    *optimized_graph->mutable_versions() = graph_def.versions();
    VLOG(0) << "All ops mapped to: " << default_device << "\n";
  }
}

set<string> PlacementOptimizer::GetMappedDevices(const GraphDef& graph_def) {
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

  return devices;
}

set<string> PlacementOptimizer::GetWhitelistedOps() {
  set<string> ops;
  ops.insert("MatMul");
  return ops;
}

string PlacementOptimizer::GetDefaultDevice(const vector<string>& devices,
                                            set<string>& pinned_devices) {
  string default_device;
  for (const auto& it1 : devices) {
    if (pinned_devices.find(it1) == pinned_devices.end()) {
      default_device = it1;
      break;
    }
  }

  return default_device;
}

set<string> PlacementOptimizer::GetPinnedDeviceStrings(set<string>& devices) {
  set<string> pinned_devices;
  set<string>::iterator it1;
  string pinned_device_string = "CPU";

  for (it1 = devices.begin(); it1 != devices.end(); it1++) {
    if ((*it1).find(pinned_device_string) != std::string::npos) {
      VLOG(0) << "pinned_device: " << *it1 << "\n";
      pinned_devices.insert(*it1);
    }
  }

  return pinned_devices;
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

void PlacementOptimizer::PrintGrapplerItemStats(const GrapplerItem& item) {
  VLOG(0) << "Feed tensors:\n";

  for (const auto& it1 : item.feed) {
    VLOG(0) << "Name: " << it1.first
            << " Description: " << (it1.second).DebugString();
  }

  VLOG(0) << "fetch: \n";
  for (const auto& it2 : item.fetch) {
    VLOG(0) << it2 << "\n";
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