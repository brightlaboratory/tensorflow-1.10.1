
#include "tensorflow/core/grappler/optimizers/placement_optimizer.h"

#include <set>
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/cost_graph.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/grappler/costs/analytical_cost_estimator.h"
#include "tensorflow/core/grappler/utils.h"

using namespace std;

namespace tensorflow {
namespace grappler {

#define MIN_EXECUTION_TIME 1000

Status PlacementOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,
                                    GraphDef* optimized_graph) {
  VLOG(0) << "Optimize Grappler item: id=" << item.id;
  VLOG(0) << "optimized_graph statistics:\n";
  PrintGraphDefStats(optimized_graph);
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
    MinCutPlacement(cluster, item.graph, cost_graph, optimized_graph);
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
  set<string> whitelisted_ops = GetWhitelistedOps();

  if (default_device.empty()) {
    VLOG(0) << "There are no non-CPU devices to map the Ops to\n";
    *optimized_graph = graph_def;
  } else {
    for (const NodeDef& node : graph_def.node()) {
      NodeDef* new_node = optimized_graph->add_node();
      *new_node = node;

      const OpDef* op_def = nullptr;
      OpRegistry::Global()->LookUpOpDef(new_node->op(), &op_def);

      if (op_def != nullptr && !op_def->is_stateful() &&
          (whitelisted_ops.find(new_node->op()) != whitelisted_ops.end())) {
        if (!new_node->device().empty()) {
          if ((pinned_devices.find(new_node->device()) ==
               pinned_devices.end()) &&
              (new_node->device() != default_device)) {
            VLOG(0) << "node_remapping of " << new_node->name()
                    << " op : " << new_node->op() << " from "
                    << new_node->device() << " to " << default_device << "\n";
            new_node->set_device(default_device);
          }
        }
      } else {
        if (op_def == nullptr) {
          VLOG(0) << new_node->op()
                  << "cannot be found in global op registry\n";
        } else {
          VLOG(0) << new_node->op() << " is stateful.\n";
        }
      }
    }

    *optimized_graph->mutable_versions() = graph_def.versions();
    VLOG(0) << "All ops mapped to: " << default_device << "\n";
  }
}

void PlacementOptimizer::PrintGraphDefStats(GraphDef* graph_def) {
  VLOG(0) << "node_size: " << graph_def->node_size() << "\n";
  for (int i = 0; i < graph_def->node_size(); i++) {
    const NodeDef& node = graph_def->node(i);
    VLOG(0) << "Node name: " << node.name();
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
  ops.insert("Add");
  ops.insert("Mul");
  ops.insert("ConcatV2");
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

set<string> PlacementOptimizer::GetDevices(Cluster* cluster) {
  const DeviceSet* device_set = cluster->GetDeviceSet();
  const std::vector<Device*>& devices = device_set->devices();
  set<string> device_strings;
  for (int i = 0; i < devices.size(); i++) {
    device_strings.insert(devices.at(i)->name());
  }

  return device_strings;
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

void PlacementOptimizer::MinCutPlacement(Cluster* cluster,
                                         const GraphDef& graph_def,
                                         CostGraphDef& cost_graph,
                                         GraphDef* optimized_graph) {
  set<string> devices = GetMappedDevices(graph_def);
  set<string> pinned_devices = GetPinnedDeviceStrings(devices);
  set<string> whitelisted_ops = GetWhitelistedOps();
  string default_device =
      GetDefaultDevice(cluster->GetDeviceNames(), pinned_devices);

  if (default_device.empty()) {
    VLOG(0) << "There are no non-CPU devices to map the Ops to\n";
    *optimized_graph = graph_def;
  } else {
    for (const NodeDef& node : graph_def.node()) {
      NodeDef* new_node = optimized_graph->add_node();
      *new_node = node;
    }

    std::unordered_map<const NodeDef*, struct NodeCommCost*> node_to_commcost;
    std::unordered_map<string, const CostGraphDef::Node*> name_to_cost;
    std::unordered_map<string, const NodeDef*> name_to_node;
    ComputeNodeCommCosts(graph_def, cost_graph, pinned_devices, whitelisted_ops,
                         node_to_commcost, name_to_cost, name_to_node);
    PartitionTheGraph(cluster, node_to_commcost, name_to_cost, name_to_node);
    FreeLocallyAllocatedMemory(node_to_commcost);
    *optimized_graph->mutable_versions() = graph_def.versions();
  }
}

void PlacementOptimizer::PartitionTheGraph(
    Cluster* cluster,
    std::unordered_map<const NodeDef*, struct NodeCommCost*>& node_to_commcost,
    std::unordered_map<string, const CostGraphDef::Node*>& name_to_cost,
    std::unordered_map<string, const NodeDef*>& name_to_node) {
  set<string> devices = GetDevices(cluster);
  ReassignNodes(devices, node_to_commcost, name_to_cost, name_to_node);
}

int PlacementOptimizer::ReassignNodes(
    set<string>& devices,
    std::unordered_map<const NodeDef*, struct NodeCommCost*>& node_to_commcost,
    std::unordered_map<string, const CostGraphDef::Node*>& name_to_cost,
    std::unordered_map<string, const NodeDef*>& name_to_node) {
  int numReassigned = 0;

  for (auto i : node_to_commcost) {
    const NodeDef* node = i.first;
    struct NodeCommCost* current_cost_node = i.second;

    string orig_device = node->device();
    string new_device = orig_device;
    int64 current_comm_cost = commcost->ec - commcost->ic;
    struct NodeCommCost* new_comm_cost = NULL;
    for (auto device : devices) {
      if (device != orig_device) {
        node->set_device(device);
        struct NodeCommCost* node_commcost =
            ComputeNodeCommCost(node, name_to_cost, name_to_node);

        int64 new_comm_cost = node_commcost->ec - node_commcost->ic;

        if (new_comm_cost < current_comm_cost) {
          if (current_cost_node) {
            free(current_cost_node);
          }

          current_cost_node = node_commcost;
          new_device = device;
        } else {
          free(node_commcost);
        }
      }
    }

    if (new_device != orig_device) {
      node->set_device(new_device);
      numReassigned++;
    }
  }

  return numReassigned;
}

void PlacementOptimizer::ComputeNodeCommCosts(
    const GraphDef& graph_def, CostGraphDef& cost_graph,
    set<string>& pinned_devices, set<string>& whitelisted_ops,
    std::unordered_map<const NodeDef*, struct NodeCommCost*>& node_to_commcost,
    std::unordered_map<string, const CostGraphDef::Node*>& name_to_cost,
    std::unordered_map<string, const NodeDef*>& name_to_node) {
  for (int i = 0; i < graph_def.node_size(); i++) {
    const NodeDef& node = graph_def.node(i);
    name_to_node[node.name()] = &node;
  }

  for (int i = 0; i < cost_graph.node_size(); i++) {
    const CostGraphDef::Node& cnode = cost_graph.node(i);
    name_to_cost[cnode.name()] = &cnode;
  }

  for (int i = 0; i < graph_def.node_size(); i++) {
    const NodeDef& node = graph_def.node(i);
    if (IsEligibleForRelocation(&node, pinned_devices, whitelisted_ops)) {
      struct NodeCommCost* node_comm_cost =
          ComputeNodeCommCost(node, name_to_cost, name_to_node);
      node_to_commcost[&node] = node_comm_cost;
      VLOG(0) << "node_comm_cost.name: " << node.name
              << " node_comm_cost.ec: " << node_comm_cost->ec
              << " node_comm_cost.ic: " << node_comm_cost->ic << "\n";
    }
  }
}

struct NodeCommCost* PlacementOptimizer::ComputeNodeCommCost(
    const NodeDef& node,
    std::unordered_map<string, const CostGraphDef::Node*>& name_to_cost,
    std::unordered_map<string, const NodeDef*>& name_to_node) {
  struct NodeCommCost* node_comm_cost =
      (struct NodeCommCost*)malloc(sizeof(struct NodeCommCost));

  for (int i = 0; i < node.input_size(); ++i) {
    const string input_name = node.input(i);
    if (IsControlInput(input_name)) {
      continue;
    }

    TensorId input_tensor_id = ParseTensorName(input_name);
    const string input_node_name = input_tensor_id.first.ToString();

    auto it1 = name_to_node.find(input_node_name);
    const NodeDef* adj_node;
    if (it1 != name_to_node.end()) {
      adj_node = it1->second;
    } else {
      adj_node = NULL;
      continue;
    }

    auto it2 = name_to_cost.find(input_node_name);
    const CostGraphDef::Node* cost_node;
    if (it2 != name_to_cost.end()) {
      cost_node = it2->second;
    } else {
      cost_node = NULL;
      continue;
    }

    // TODO: Think about if we need to consider CPU mapped adj_node
    if (!adj_node->device().empty() /* &&
            (pinned_devices.find(adj_node->device()) == pinned_devices.end()) */) {
      if (adj_node->device() == node.device()) {
        node_comm_cost->ic += cost_node->max_memory_size();
      } else {
        node_comm_cost->ec += cost_node->max_memory_size();
      }
    }
  }

  auto it = name_to_cost.find(node.name());
  const CostGraphDef::Node* cost_node = NULL;
  if (it != name_to_cost.end()) {
    cost_node = it->second;
    node_comm_cost->compute_cost = cost_node->compute_cost();
  }
}

void PlacementOptimizer::FreeLocallyAllocatedMemory(
    std::unordered_map<const NodeDef*, struct NodeCommCost*>&
        node_to_commcost) {
  for (auto i : node_to_commcost) {
    free(i.second);
  }
}

bool PlacementOptimizer::IsEligibleForRelocation(const NodeDef* node,
                                                 set<string>& pinned_devices,
                                                 set<string>& whitelisted_ops) {
  const OpDef* op_def = nullptr;
  OpRegistry::Global()->LookUpOpDef(node->op(), &op_def);

  // TODO: Think about if we need to consider CPU mapped adj_node
  if (op_def != nullptr && !op_def->is_stateful() &&
      (whitelisted_ops.find(node->op()) != whitelisted_ops.end()) &&
      !node->device().empty() /* &&
      (pinned_devices.find(node->device()) == pinned_devices.end()) */) {
    return true;
  } else {
    return false;
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