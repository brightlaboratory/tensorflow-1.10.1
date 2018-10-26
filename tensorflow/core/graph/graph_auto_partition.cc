
#include "tensorflow/core/graph/graph_auto_partition.h"

#include <deque>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/memory_types.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/control_flow.h"
#include "tensorflow/core/graph/costmodel.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
void PrintStats(Graph* g) {
  for (const Node* dst : g->op_nodes()) {
    // g->set_assigned_device_name(dst,
    //                            "/job:localhost/replica:0/task:0/device:GPU:0");
    VLOG(0) << "Name: " << dst->name() << " num_inputs: " << dst->num_inputs()
            << " num_outputs: " << dst->num_outputs()
            << "requested_device: " << dst->requested_device()
            << " assigned_device: " << dst->assigned_device_name() << "\n";
  }
}
}  // namespace tensorflow