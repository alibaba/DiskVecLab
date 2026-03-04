#include "full_precision_quantizer.h"
#include "distance.h"
#include <iostream>
#include <fstream>
#include <cstring>

// Template helper function to load and convert data to float
template<typename T>
static void load_full_precision_data(const std::string& bin_file, size_t& num_points, size_t& dim,
                                     std::vector<float>& vectors) {
    size_t read_blk_size = 64 * 1024 * 1024;
    cached_ifstream base_reader(bin_file, read_blk_size);
    uint32_t npts32, basedim32;
    base_reader.read((char *)&npts32, sizeof(uint32_t));
    base_reader.read((char *)&basedim32, sizeof(uint32_t));
    num_points = npts32;
    dim = basedim32;

    size_t BLOCK_SIZE = 5000000;
    size_t block_size = num_points <= BLOCK_SIZE ? num_points : BLOCK_SIZE;
    size_t num_blocks = DIV_ROUND_UP(num_points, block_size);

    std::unique_ptr<T[]> block_data_T = std::make_unique<T[]>(block_size * dim);
    std::unique_ptr<float[]> block_data_tmp = std::make_unique<float[]>(block_size * dim);

    vectors.clear();
    vectors.reserve(num_points * dim);
    
    for (size_t block = 0; block < num_blocks; block++) {
        size_t start_id = block * block_size;
        size_t end_id = (std::min)((block + 1) * block_size, num_points);
        size_t cur_blk_size = end_id - start_id;

        base_reader.read((char *)(block_data_T.get()), sizeof(T) * (cur_blk_size * dim));
        diskann::convert_types<T, float>(block_data_T.get(), block_data_tmp.get(), cur_blk_size, dim);

        diskann::cout << "Processing points  [" << start_id << ", " << end_id << ").." << std::flush;
        vectors.insert(vectors.end(), block_data_tmp.get(), block_data_tmp.get() + cur_blk_size * dim);
    }
}

FullPrecisionQuantizer::FullPrecisionQuantizer(size_t d) : dim_(d), num_points_(0) {
}

void FullPrecisionQuantizer::train(size_t n, const float* x, const std::string data_file, const std::string index_file, const std::string& data_type) {
    // full precision doesn't need to train, but we note the data type for later use
    // The data_type parameter is available for future use if needed
}

DistanceComputer* FullPrecisionQuantizer::preprocess_query(float* query) {
    auto distanceComputer =  new FullPrecisionDistanceComputer(this);
    distanceComputer->set_query(query);
    return distanceComputer;
}

void FullPrecisionQuantizer::compute_dists(const uint32_t *ids, const uint64_t n_ids, float *dists_out,
                                           uint8_t *data, uint8_t *pq_coord_scratch, float* pq_dists,
                                           DistanceComputer* dc_computer) {
    for (uint64_t i = 0; i < n_ids; i++) {
        dists_out[i] = (*dc_computer)(ids[i]);
    }
}

void FullPrecisionQuantizer::load_pq_compressed_vectors(const std::string &bin_file, uint8_t* &data, const std::string& data_type) {
    // Dispatch to appropriate template function based on data type
    if (data_type == "float") {
        load_full_precision_data<float>(bin_file, num_points_, dim_, vectors_);
    } else if (data_type == "uint8") {
        load_full_precision_data<uint8_t>(bin_file, num_points_, dim_, vectors_);
    } else if (data_type == "int8") {
        load_full_precision_data<int8_t>(bin_file, num_points_, dim_, vectors_);
    } else {
        throw std::runtime_error("Unsupported data type: " + data_type + ". Supported types: float, uint8, int8");
    }
    
    std::cout << "Loaded " << num_points_ << " full precision vectors of dimension " << dim_ << std::endl;
    std::cout << "vectors_.size() = " << vectors_.size() << std::endl;
}

float FullPrecisionQuantizer::compute_distance_to_query(uint32_t row_id, const float* query) const {
    if (row_id >= num_points_) {
        diskann::cout << "row_id " << row_id << " >= num_points_ " << num_points_ << std::endl;
        return std::numeric_limits<float>::max();
    }
    
    if (vectors_.empty()) {
        diskann::cout << "vectors_ is empty!" << std::endl;
        return std::numeric_limits<float>::max();
    }
    
    if (dim_ == 0) {
        diskann::cout << "dim_ is 0!" << std::endl;
        return std::numeric_limits<float>::max();
    }
    
    // Check bounds
    size_t vector_start_idx = static_cast<size_t>(row_id) * dim_;
    if (vector_start_idx + dim_ > vectors_.size()) {
        diskann::cout << "vector bounds check failed: start=" << vector_start_idx << " dim=" << dim_ << " vectors_size=" << vectors_.size() << std::endl;
        exit(1);
    }
    
    // Compute L2 distance between query and stored vector
    const float* stored_vec = vectors_.data() + vector_start_idx;
    float dist = 0.0f;

#pragma omp parallel for reduction(+:dist)
    for (size_t i = 0; i < dim_; i++) {
        float diff = query[i] - stored_vec[i];
        dist += diff * diff;
    }
    
    return std::sqrt(dist);
}