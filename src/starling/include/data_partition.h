#pragma once

#include "aux_utils.h"
#include "partition_and_pq.h"

namespace diskann {

void knn_split(std::string data_type, std::string data_path, std::string prefix_path, float sampling_rate, size_t num_partitions, size_t k_index);

// Template declarations for generic data types
template<typename T>
void natural_split(std::string base_file, std::string output_prefix, int n);

} // diskann
