#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_PLACEMENT_OPTIMIZER_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_PLACEMENT_OPTIMIZER_H_

#include "tensorflow/core/framework/cost_graph.pb.h"
#include "tensorflow/core/grappler/optimizers/graph_optimizer.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
using namespace std;

namespace tensorflow {
namespace grappler {

struct NodeCommCost {
  int64 compute_cost;
  int64 ec;  // external cost
  int64 ic;  // internal cost

  NodeCommCost() : compute_cost(0), ec(0), ic(0) {}
};

struct PlacementOptimizerOptions {
  bool usePlacementOptimizer;
  PlacementOptimizerOptions() : usePlacementOptimizer(true) {}
};

// Remap TensorFlow subgraphs onto alternative operations or collection of
// operations to make the overall graph more efficient.
class PlacementOptimizer : public GraphOptimizer {
 public:
  explicit PlacementOptimizer(RewriterConfig::Toggle opt_level)
      : opt_level_(opt_level) {}
  ~PlacementOptimizer() override = default;

  string name() const override { return "placement_optimizer"; };

  Status Optimize(Cluster* cluster, const GrapplerItem& item,
                  GraphDef* optimized_graph) override;

  void Feedback(Cluster* cluster, const GrapplerItem& item,
                const GraphDef& optimized_graph, double result) override;

 private:
  RewriterConfig::Toggle opt_level_;

  void PrintDeviceStats(Cluster* cluster);

  void PrintCostStats(const GrapplerItem& item, CostGraphDef& cost_graph);

  void CreateDefaultPlacement(Cluster* cluster, const GraphDef& graph_def,
                              GraphDef* optimized_graph);

  void MinCutPlacement(Cluster* cluster, const GraphDef& graph_def,
                       CostGraphDef& cost_graph, GraphDef* optimized_graph);

  set<string> GetWhitelistedOps();

  set<string> GetPinnedDeviceStrings(set<string>& devices);

  string GetDefaultDevice(const vector<string>& devices,
                          set<string>& pinned_devices);

  set<string> GetMappedDevices(const GraphDef& graph_def);

  set<string> GetDevices(Cluster* cluster);

  void PrintGrapplerItemStats(const GrapplerItem& item);

  void PrintGraphDefStats(GraphDef* graph_def);

  bool IsEligibleForRelocation(const NodeDef* node, set<string>& pinned_devices,
                               set<string>& whitelisted_ops);

  void ComputeNodeCommCosts(
      CostGraphDef& cost_graph, set<string>& pinned_devices,
      set<string>& whitelisted_ops,
      std::unordered_map<NodeDef*, struct NodeCommCost*>& node_to_commcost,
      std::unordered_map<string, const CostGraphDef::Node*>& name_to_cost,
      std::unordered_map<string, NodeDef*>& name_to_node);

  void PartitionTheGraph(
      Cluster* cluster,
      std::unordered_map<NodeDef*, struct NodeCommCost*>& node_to_commcost,
      std::unordered_map<string, const CostGraphDef::Node*>& name_to_cost,
      std::unordered_map<string, NodeDef*>& name_to_node, set<string>& devices,
      set<string>& pinned_devices);

  int ReassignNodes(
      set<string>& devices,
      std::unordered_map<NodeDef*, struct NodeCommCost*>& node_to_commcost,
      std::unordered_map<string, const CostGraphDef::Node*>& name_to_cost,
      std::unordered_map<string, NodeDef*>& name_to_node,
      set<string>& pinned_devices);

  NodeCommCost* ComputeNodeCommCost(
      NodeDef* node,
      std::unordered_map<string, const CostGraphDef::Node*>& name_to_cost,
      std::unordered_map<string, NodeDef*>& name_to_node);

  void FreeLocallyAllocatedMemory(
      std::unordered_map<NodeDef*, struct NodeCommCost*>& node_to_commcost);

  int64 ComputePerDeviceComputeCost(
      std::unordered_map<string, int64>& compute_costs,
      std::unordered_map<NodeDef*, struct NodeCommCost*>& node_to_commcost,
      set<string>& devices);

  bool IsBeneficialToMoveNode(double compute_margin, double idealPartitionShare,
                              std::unordered_map<string, int64>& compute_costs,
                              int64 current_compute_cost, int64 new_comm_cost,
                              int64 current_comm_cost, string orig_device,
                              string device, int64 total_compute_cost);

  void ParseOptions(struct PlacementOptimizerOptions& options);

  void CreateInitialPartition(
      CostGraphDef& cost_graph, set<string>& pinned_devices,
      set<string>& whitelisted_ops,
      std::unordered_map<NodeDef*, struct NodeCommCost*>& node_to_commcost,
      std::unordered_map<string, const CostGraphDef::Node*>& name_to_cost,
      std::unordered_map<string, NodeDef*>& name_to_node, set<string>& devices,
      const char* strategy);
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif