
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/cost_graph.pb.h"
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/grappler/costs/analytical_cost_estimator.h"
#include "tensorflow/core/grappler/optimizers/function_optimizer.h"

namespace tensorflow {
namespace grappler {

Status PlacementOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,
                                    GraphDef* optimized_graph) {
  VLOG(0) << "Optimize Grappler item: id=" << item.id;
  optimized_graph = item.graph;

  AnalyticalCostEstimator estimator(cluster, true);
  Status initStatus = estimator.Initialize(item));

  if (initStatus != Status::OK()) {
    return initStatus;
  }

  CostGraphDef cost_graph;
  Costs summary;
  Status predictStatus = estimator.PredictCosts(item.graph, &cost_graph, &summary));

  if (predictStatus != Status::OK) {
    return predictStatus;
  }

  VLOG(0) << "summary.execution_time: " << summary.execution_time << "\n";
  return Status::OK();
}

void PlacementOptimizer::Feedback(Cluster* cluster, const GrapplerItem& item,
                                  const GraphDef& optimized_graph,
                                  double result) {
  // nothing to be done
}

}  // namespace grappler
}  // namespace tensorflow