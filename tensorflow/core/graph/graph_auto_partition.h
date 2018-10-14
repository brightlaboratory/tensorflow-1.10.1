#ifndef TENSORFLOW_GRAPH_GRAPH_AUTO_PARTITION_H_
#define TENSORFLOW_GRAPH_GRAPH_AUTO_PARTITION_H_

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/costmodel.h"
#include "tensorflow/core/graph/graph.h"

namespace tensorflow {

void PrintStats(Graph* graph);
}

#endif