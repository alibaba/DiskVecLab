#include "rabitq_quantizer.h"
#include "utils.h"
#include "math_utils.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <limits>
#include "rabitqlib/quantization/data_layout.hpp"
#include "rabitqlib/index/estimator.hpp"
#include "rabitqlib/utils/memory.hpp"
#include "rabitqlib/fastscan/fastscan.hpp"

// Helper: find the nearest centroid for a rotated vector
static uint32_t find_nearest_centroid(const float* rotated_vector, const float* rotated_centroids,
                                       size_t num_clusters, size_t padded_dim) {
    uint32_t best_cid = 0;
    if (num_clusters <= 1) return 0;

    float min_dist = std::numeric_limits<float>::max();
    for (size_t k = 0; k < num_clusters; k++) {
        const float* cent = rotated_centroids + k * padded_dim;
        float dist = 0;
        for (size_t d = 0; d < padded_dim; d++) {
            float diff = rotated_vector[d] - cent[d];
            dist += diff * diff;
        }
        if (dist < min_dist) {
            min_dist = dist;
            best_cid = static_cast<uint32_t>(k);
        }
    }
    return best_cid;
}

// Template helper function to handle different input data types
// Multi-cluster version: each vector is quantized relative to its nearest centroid.
template<typename T>
static void read_and_quantize_data(const std::string& data_file, size_t num_points, size_t dim, 
                                   size_t padded_dim, size_t ex_bits,
                                   const float* rotated_centroids,
                                   size_t num_clusters,
                                   rabitqlib::Rotator<float>* rotator,
                                   char* bin_data, char* ex_data,
                                   size_t bin_data_stride, size_t ex_data_stride,
                                   uint32_t* cluster_ids,
                                   const rabitqlib::quant::RabitqConfig& config) {
    size_t read_blk_size = 64 * 1024 * 1024;
    cached_ifstream base_reader(data_file, read_blk_size);
    uint32_t npts32, basedim32;
    base_reader.read((char *)&npts32, sizeof(uint32_t));
    base_reader.read((char *)&basedim32, sizeof(uint32_t));

    size_t BLOCK_SIZE = 5000000;
    size_t block_size = num_points <= BLOCK_SIZE ? num_points : BLOCK_SIZE;
    size_t num_blocks = DIV_ROUND_UP(num_points, block_size);

    std::unique_ptr<T[]> block_data_T = std::make_unique<T[]>(block_size * dim);
    std::unique_ptr<float[]> block_data_tmp = std::make_unique<float[]>(block_size * dim);
    std::unique_ptr<float[]> rotated_data = std::make_unique<float[]>(block_size * padded_dim);

    auto* bin_data_temp = bin_data;
    auto* ex_data_temp = ex_data;
    
    for (size_t block = 0; block < num_blocks; block++) {
        size_t start_id = block * block_size;
        size_t end_id = (std::min)((block + 1) * block_size, num_points);
        size_t cur_blk_size = end_id - start_id;

        base_reader.read((char *)(block_data_T.get()), sizeof(T) * (cur_blk_size * dim));
        diskann::convert_types<T, float>(block_data_T.get(), block_data_tmp.get(), cur_blk_size, dim);

        diskann::cout << "Processing points [" << start_id << ", " << end_id << ").." << std::flush;
        
        // Rotate each vector, find nearest centroid, and quantize
        for (size_t i = 0; i < cur_blk_size; i++) {
            float* vector = block_data_tmp.get() + i * dim;
            float* rotated_vector = rotated_data.get() + i * padded_dim;
            rotator->rotate(vector, rotated_vector);

            // Find nearest centroid for this vector
            uint32_t best_cid = find_nearest_centroid(rotated_vector, rotated_centroids,
                                                       num_clusters, padded_dim);
            cluster_ids[start_id + i] = best_cid;

            // Quantize relative to the assigned centroid
            const float* centroid = rotated_centroids + best_cid * padded_dim;
            rabitqlib::quant::quantize_split_single(
                rotated_vector,
                centroid,
                padded_dim,
                ex_bits,
                bin_data_temp,
                ex_data_temp,
                rabitqlib::METRIC_L2,
                config
            );

            bin_data_temp += bin_data_stride;
            ex_data_temp += ex_data_stride;
        }
        diskann::cout << "done" << std::endl;
    }
}

RabitqQuantizer::RabitqQuantizer(size_t d, size_t bits, size_t num_clusters)
    : dim_(d), bits_(bits), num_clusters_(num_clusters > 0 ? num_clusters : 1) {
    if (d > 0) {
        rotator_ = std::unique_ptr<rabitqlib::Rotator<float>>(rabitqlib::choose_rotator<float>(d, rabitqlib::RotatorType::FhtKacRotator));
        padded_dim_ = rotator_->size();
        ex_bits_ = bits_ - 1;
        if (faster) {
            this->config_ =
                rabitqlib::quant::faster_config(padded_dim_, ex_bits_ + 1);
        }

        bin_data_stride_ = rabitqlib::BinDataMap<float>::data_bytes(padded_dim_);
        ex_data_stride_ = rabitqlib::ExDataMap<float>::data_bytes(padded_dim_, ex_bits_);
        
        // Initialize scratch buffer sizes
        init_scratch_buffer_sizes();
    }
}

void RabitqQuantizer::train(size_t n, const float* x, const std::string data_file, const std::string index_file, const std::string& data_type) {
    size_t d = dim_;

    // ============================================================
    // Step 1: K-means clustering to get multiple centroids
    // ============================================================
    // Clamp num_clusters to not exceed training data size
    size_t effective_clusters = std::min(num_clusters_, n);
    if (effective_clusters < num_clusters_) {
        diskann::cout << "Warning: num_clusters (" << num_clusters_
                      << ") exceeds training data size (" << n
                      << "), clamping to " << effective_clusters << std::endl;
        num_clusters_ = effective_clusters;
    }

    diskann::cout << "RaBitQ multi-cluster training with " << num_clusters_
                  << " clusters on " << n << " training samples, dim=" << d << std::endl;

    std::vector<float> centroids(num_clusters_ * d, 0);

    if (num_clusters_ == 1) {
        // Single cluster: just compute global mean (original behavior)
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < d; j++) {
                centroids[j] += x[i * d + j];
            }
        }
        if (n != 0) {
            for (size_t j = 0; j < d; j++) {
                centroids[j] /= static_cast<float>(n);
            }
        }
        diskann::cout << "Single cluster (global centroid) computed." << std::endl;
    } else {
        // Multi-cluster: run k-means on training data
        // Need a non-const copy for k-means (it may modify data internally)
        std::vector<float> train_data_copy(n * d);
        std::memcpy(train_data_copy.data(), x, n * d * sizeof(float));

        constexpr uint32_t KMEANS_MAX_REPS = 15;

        diskann::cout << "Running k-means++ initialization..." << std::endl;
        kmeans::kmeanspp_selecting_pivots(train_data_copy.data(), n, d,
                                          centroids.data(), num_clusters_);

        diskann::cout << "Running Lloyd's iterations (max " << KMEANS_MAX_REPS << " reps)..." << std::endl;
        kmeans::run_lloyds(train_data_copy.data(), n, d,
                           centroids.data(), num_clusters_,
                           KMEANS_MAX_REPS, nullptr, nullptr);

        diskann::cout << "K-means clustering done: " << num_clusters_ << " centroids." << std::endl;
    }

    // ============================================================
    // Step 2: Rotate all centroids
    // ============================================================
    rotated_centroids_.resize(num_clusters_ * padded_dim_, 0);
    for (size_t k = 0; k < num_clusters_; k++) {
        rotator_->rotate(centroids.data() + k * d,
                         rotated_centroids_.data() + k * padded_dim_);
    }

    diskann::cout << "Rotated " << num_clusters_ << " centroids (padded_dim=" << padded_dim_ << ")." << std::endl;

    // ============================================================
    // Step 3: Read file header and allocate storage
    // ============================================================
    size_t read_blk_size = 64 * 1024 * 1024;
    cached_ifstream base_reader(data_file, read_blk_size);
    uint32_t npts32, basedim32;
    base_reader.read((char *)&npts32, sizeof(uint32_t));
    base_reader.read((char *)&basedim32, sizeof(uint32_t));
    
    size_t num_points = npts32;
    size_t dim = basedim32;
    num_points_ = num_points;

    // Allocate memory for quantized data
    this->bin_data_ = rabitqlib::memory::align_allocate<64, char, true>(bin_data_bytes());
    if (ex_bits_ > 0) {
        this->ex_data_ = rabitqlib::memory::align_allocate<64, char, true>(ex_data_bytes());
    }

    memset(this->bin_data_, 0, num_points_ * bin_data_stride_);
    memset(this->ex_data_, 0, num_points_ * ex_data_stride_);

    // Allocate cluster IDs
    cluster_ids_.resize(num_points_, 0);

    // ============================================================
    // Step 4: Quantize all vectors relative to their nearest centroid
    // ============================================================
    if (data_type == "float") {
        read_and_quantize_data<float>(data_file, num_points, dim, padded_dim_, ex_bits_,
                                     rotated_centroids_.data(), num_clusters_,
                                     rotator_.get(), bin_data_, ex_data_,
                                     bin_data_stride_, ex_data_stride_,
                                     cluster_ids_.data(), config_);
    } else if (data_type == "uint8") {
        read_and_quantize_data<uint8_t>(data_file, num_points, dim, padded_dim_, ex_bits_,
                                       rotated_centroids_.data(), num_clusters_,
                                       rotator_.get(), bin_data_, ex_data_,
                                       bin_data_stride_, ex_data_stride_,
                                       cluster_ids_.data(), config_);
    } else if (data_type == "int8") {
        read_and_quantize_data<int8_t>(data_file, num_points, dim, padded_dim_, ex_bits_,
                                      rotated_centroids_.data(), num_clusters_,
                                      rotator_.get(), bin_data_, ex_data_,
                                      bin_data_stride_, ex_data_stride_,
                                      cluster_ids_.data(), config_);
    } else {
        throw std::runtime_error("Unsupported data type: " + data_type + ". Supported types: float, uint8, int8");
    }

    // Print cluster distribution statistics
    {
        std::vector<size_t> cluster_counts(num_clusters_, 0);
        for (size_t i = 0; i < num_points_; i++) {
            cluster_counts[cluster_ids_[i]]++;
        }
        diskann::cout << "Cluster distribution:" << std::endl;
        for (size_t k = 0; k < num_clusters_; k++) {
            diskann::cout << "  cluster " << k << ": " << cluster_counts[k] << " vectors" << std::endl;
        }
    }

    // ============================================================
    // Step 5: Save to file
    // ============================================================
    std::ofstream out_file(index_file, std::ios::binary);
    if (!out_file.is_open()) {
        throw std::runtime_error("Cannot open output file: " + index_file);
    }
    
    // Write header
    out_file.write(reinterpret_cast<const char*>(&num_points_), sizeof(num_points_));
    out_file.write(reinterpret_cast<const char*>(&dim_), sizeof(dim_));
    out_file.write(reinterpret_cast<const char*>(&padded_dim_), sizeof(padded_dim_));
    out_file.write(reinterpret_cast<const char*>(&bits_), sizeof(bits_));
    out_file.write(reinterpret_cast<const char*>(&ex_bits_), sizeof(ex_bits_));
    out_file.write(reinterpret_cast<const char*>(&bin_data_stride_), sizeof(bin_data_stride_));
    out_file.write(reinterpret_cast<const char*>(&ex_data_stride_), sizeof(ex_data_stride_));
    out_file.write(reinterpret_cast<const char*>(&num_clusters_), sizeof(num_clusters_));

    // Write all rotated centroids (num_clusters * padded_dim floats)
    out_file.write(reinterpret_cast<const char*>(rotated_centroids_.data()), 
                   sizeof(float) * rotated_centroids_.size());

    // Write cluster IDs (num_points uint32_t)
    out_file.write(reinterpret_cast<const char*>(cluster_ids_.data()),
                   sizeof(uint32_t) * cluster_ids_.size());

    // Write quantized data
    out_file.write(reinterpret_cast<const char*>(bin_data_), static_cast<long>(bin_data_bytes()));
    out_file.write(reinterpret_cast<const char*>(ex_data_), static_cast<long>(ex_data_bytes()));

    // Write rotator
    rotator_->save(out_file);

    out_file.close();
    diskann::cout << "Saved RaBitQ multi-cluster index (" << num_clusters_
                  << " clusters) to " << index_file << std::endl;
}

DistanceComputer* RabitqQuantizer::preprocess_query(float* query) {
    std::lock_guard<std::mutex> guard(mutex_);

    auto dc = std::make_unique<RabitQDistanceComputer>(this);
    dc->set_query(query);
    return dc.release();
}

void RabitqQuantizer::load_pq_compressed_vectors(const std::string &bin_file, uint8_t* &data, const std::string& data_type) {
    if (bin_file.empty()) {
        diskann::cout << "Index file is empty when loading RaBitQ" << std::endl;
        exit(1);
    }
    
    std::ifstream in_file(bin_file, std::ios::binary);
    if (!in_file.is_open()) {
        throw std::runtime_error("Cannot open input file: " + bin_file);
    }
    
    // Read header
    in_file.read(reinterpret_cast<char*>(&num_points_), sizeof(num_points_));
    in_file.read(reinterpret_cast<char*>(&dim_), sizeof(dim_));
    in_file.read(reinterpret_cast<char*>(&padded_dim_), sizeof(padded_dim_));
    in_file.read(reinterpret_cast<char*>(&bits_), sizeof(bits_));
    in_file.read(reinterpret_cast<char*>(&ex_bits_), sizeof(ex_bits_));
    in_file.read(reinterpret_cast<char*>(&bin_data_stride_), sizeof(bin_data_stride_));
    in_file.read(reinterpret_cast<char*>(&ex_data_stride_), sizeof(ex_data_stride_));
    in_file.read(reinterpret_cast<char*>(&num_clusters_), sizeof(num_clusters_));
    
    diskann::cout << "Loading RaBitQ index from " << bin_file
                  << " num_points=" << num_points_
                  << " dim=" << dim_
                  << " padded_dim=" << padded_dim_
                  << " bits=" << bits_
                  << " num_clusters=" << num_clusters_
                  << std::endl;
    
    // Initialize rotator and config
    rotator_ = std::unique_ptr<rabitqlib::Rotator<float>>(rabitqlib::choose_rotator<float>(dim_, rabitqlib::RotatorType::FhtKacRotator));
    padded_dim_ = rotator_->size();
    if (faster) {
        this->query_config_ =
            rabitqlib::quant::faster_config(padded_dim_, rabitqlib::SplitSingleQuery<float>::kNumBits);
    }

    // Verify strides are consistent
    size_t expected_bin_stride = rabitqlib::BinDataMap<float>::data_bytes(padded_dim_);
    size_t expected_ex_stride = rabitqlib::ExDataMap<float>::data_bytes(padded_dim_, ex_bits_);
    
    if (bin_data_stride_ != expected_bin_stride) {
        diskann::cout << "Warning: bin_data_stride mismatch, using calculated value" << std::endl;
        bin_data_stride_ = expected_bin_stride;
    }
    if (ex_data_stride_ != expected_ex_stride) {
        diskann::cout << "Warning: ex_data_stride mismatch, using calculated value" << std::endl;
        ex_data_stride_ = expected_ex_stride;
    }

    // Read all rotated centroids (num_clusters * padded_dim floats)
    rotated_centroids_.resize(num_clusters_ * padded_dim_);
    in_file.read(reinterpret_cast<char*>(rotated_centroids_.data()), 
                 sizeof(float) * rotated_centroids_.size());

    diskann::cout << "Loaded " << num_clusters_ << " rotated centroids." << std::endl;

    // Read cluster IDs (num_points uint32_t)
    cluster_ids_.resize(num_points_);
    in_file.read(reinterpret_cast<char*>(cluster_ids_.data()),
                 sizeof(uint32_t) * cluster_ids_.size());

    diskann::cout << "Loaded " << num_points_ << " cluster IDs." << std::endl;
    
    // Read quantized data
    this->bin_data_ = rabitqlib::memory::align_allocate<64, char, true>(bin_data_bytes());
    if (ex_bits_ > 0) {
        this->ex_data_ = rabitqlib::memory::align_allocate<64, char, true>(ex_data_bytes());
    }
    
    in_file.read(reinterpret_cast<char*>(bin_data_), static_cast<long>(bin_data_bytes()));
    in_file.read(reinterpret_cast<char*>(ex_data_), static_cast<long>(ex_data_bytes()));

    rotator_->load(in_file);

    in_file.close();
    
    // Initialize scratch buffer sizes for compute_dists optimization
    init_scratch_buffer_sizes();
    
    diskann::cout << "Loaded RaBitQ multi-cluster index (" << num_clusters_ << " clusters)." << std::endl;
}

// Thread-local scratch buffers to avoid repeated allocation
// These are lazily initialized per-thread
struct RabitqScratchBuffers {
    std::vector<char> batch_data;
    std::vector<char> ex_data_batch;
    std::vector<uint8_t> unpacked_codes;
    std::vector<float> est_dist;
    std::vector<float> low_dist;
    std::vector<float> ip_x0_qr;
    size_t batch_data_capacity = 0;
    size_t ex_data_batch_capacity = 0;
    size_t unpacked_codes_capacity = 0;
    
    void ensure_capacity(size_t batch_data_size, size_t ex_data_batch_size, 
                         size_t unpacked_codes_size, size_t kBatchSize) {
        if (batch_data_capacity < batch_data_size) {
            batch_data.resize(batch_data_size);
            batch_data_capacity = batch_data_size;
        }
        if (ex_data_batch_capacity < ex_data_batch_size && ex_data_batch_size > 0) {
            ex_data_batch.resize(ex_data_batch_size);
            ex_data_batch_capacity = ex_data_batch_size;
        }
        if (unpacked_codes_capacity < unpacked_codes_size) {
            unpacked_codes.resize(unpacked_codes_size);
            unpacked_codes_capacity = unpacked_codes_size;
        }
        // These are always kBatchSize
        if (est_dist.size() < kBatchSize) {
            est_dist.resize(kBatchSize);
            low_dist.resize(kBatchSize);
            ip_x0_qr.resize(kBatchSize);
        }
    }
};

static thread_local RabitqScratchBuffers tls_scratch;

void RabitqQuantizer::init_scratch_buffer_sizes() {
    bin_code_size_per_vector_ = padded_dim_ / 8;
    batch_data_size_ = rabitqlib::BatchDataMap<float>::data_bytes(padded_dim_);
    ex_data_batch_size_ = ex_bits_ > 0 ? ex_data_stride_ * rabitqlib::fastscan::kBatchSize : 0;
    unpacked_codes_size_ = bin_code_size_per_vector_ * rabitqlib::fastscan::kBatchSize;
}

inline void RabitqQuantizer::swap_bytes_u64_to_u8(const uint64_t* src, uint8_t* dst, size_t num_u64) {
    for (size_t k = 0; k < num_u64; k++) {
        uint64_t val = src[k];
        // Use __builtin_bswap64 if available for better performance
#if defined(__GNUC__) || defined(__clang__)
        val = __builtin_bswap64(val);
        memcpy(dst + k * 8, &val, 8);
#else
        dst[k * 8 + 0] = (val >> 56) & 0xFF;
        dst[k * 8 + 1] = (val >> 48) & 0xFF;
        dst[k * 8 + 2] = (val >> 40) & 0xFF;
        dst[k * 8 + 3] = (val >> 32) & 0xFF;
        dst[k * 8 + 4] = (val >> 24) & 0xFF;
        dst[k * 8 + 5] = (val >> 16) & 0xFF;
        dst[k * 8 + 6] = (val >> 8) & 0xFF;
        dst[k * 8 + 7] = val & 0xFF;
#endif
    }
}

void RabitqQuantizer::prepare_batch_data(
    const uint32_t* ids, size_t start_idx, size_t batch_size, size_t n_ids,
    char* batch_data, uint8_t* unpacked_codes, char* ex_data_batch) const {
    
    constexpr size_t kBatchSize = rabitqlib::fastscan::kBatchSize;
    const size_t num_u64 = bin_code_size_per_vector_ / sizeof(uint64_t);
    
    // Get pointers to different sections of batch_data
    float* f_add = reinterpret_cast<float*>(batch_data + (bin_code_size_per_vector_ * kBatchSize));
    float* f_rescale = f_add + kBatchSize;
    float* f_error = f_rescale + kBatchSize;
    
    // Process valid vectors in the batch
    const size_t actual_count = std::min(batch_size, kBatchSize);
    for (size_t j = 0; j < actual_count; j++) {
        const size_t id = ids[start_idx + j];
        const char* src_bin_data = bin_data_ + id * bin_data_stride_;
        
        // Swap bytes and copy binary code
        swap_bytes_u64_to_u8(
            reinterpret_cast<const uint64_t*>(src_bin_data),
            unpacked_codes + j * bin_code_size_per_vector_,
            num_u64
        );
        
        // Copy f_add, f_rescale, f_error values
        const float* src_f_values = reinterpret_cast<const float*>(src_bin_data + bin_code_size_per_vector_);
        f_add[j] = src_f_values[0];
        f_rescale[j] = src_f_values[1];
        f_error[j] = src_f_values[2];
        
        // Copy ex_data if present
        if (ex_data_batch != nullptr) {
            memcpy(ex_data_batch + j * ex_data_stride_, 
                   ex_data_ + id * ex_data_stride_, 
                   ex_data_stride_);
        }
    }
    
    // Fill padding with neutral values (unpacked_codes already zeroed by caller)
    for (size_t j = actual_count; j < kBatchSize; j++) {
        f_add[j] = 0.0f;
        f_rescale[j] = 1.0f;  // Neutral value for rescaling
        f_error[j] = 0.0f;
    }
}

void RabitqQuantizer::compute_dists(const uint32_t *ids, const uint64_t n_ids, float *dists_out,
                                   uint8_t *data, uint8_t *pq_coord_scratch, float* pq_dists,
                                   DistanceComputer* dc_computer) {
    if (n_ids == 0) return;
    
    auto* rabitq_dc = dynamic_cast<RabitqQuantizer::RabitQDistanceComputer*>(dc_computer);

    if (rabitq_dc && use_rabitq_fastscan_) {
        constexpr size_t kBatchSize = rabitqlib::fastscan::kBatchSize;
        
        // Ensure scratch buffers are allocated (lazy init, no reallocation if already big enough)
        tls_scratch.ensure_capacity(batch_data_size_, ex_data_batch_size_, 
                                    unpacked_codes_size_, kBatchSize);
        
        // Get references to thread-local buffers
        char* batch_data = tls_scratch.batch_data.data();
        char* ex_data_batch = ex_bits_ > 0 ? tls_scratch.ex_data_batch.data() : nullptr;
        uint8_t* unpacked_codes = tls_scratch.unpacked_codes.data();
        float* est_dist = tls_scratch.est_dist.data();
        float* low_dist = tls_scratch.low_dist.data();
        float* ip_x0_qr = tls_scratch.ip_x0_qr.data();
        
        // Reuse cached batch-query object (created in RabitQDistanceComputer::set_query)
        // NOTE: g_add is 0 in the batch_query; per-vector correction applied below.
        if (!rabitq_dc->batch_query) {
            rabitq_dc->batch_query = std::make_unique<rabitqlib::SplitBatchQuery<float>>(
                rabitq_dc->query_rotated.data(),
                padded_dim_,
                ex_bits_,
                rabitqlib::METRIC_L2,
                rabit_use_hacc_);
            // g_add intentionally NOT set (remains 0)
        }
        auto& batch_query = *(rabitq_dc->batch_query);
        
        // Get pointer to packed bin codes (output of pack_codes goes here)
        uint8_t* batch_bin_code = reinterpret_cast<uint8_t*>(batch_data);
        
        // Pre-select the excode IP function once (only if ex_bits_ > 0)
        auto excode_ipfunc = (ex_bits_ > 0) ? (rabitq_dc->ex_ip_func ? rabitq_dc->ex_ip_func
                                         : rabitqlib::select_excode_ipfunc(ex_bits_))
                            : nullptr;

        // Process in batches
        for (size_t i = 0; i < n_ids; i += kBatchSize) {
            const size_t batch_size = std::min(static_cast<size_t>(kBatchSize), n_ids - i);
            
            // Only zero-init unpacked_codes for incomplete batches (padding needs zeros for pack_codes)
            if (batch_size < kBatchSize) {
                memset(unpacked_codes, 0, unpacked_codes_size_);
            }
            
            // Prepare batch data (unified for both ex_bits paths)
            prepare_batch_data(ids, i, batch_size, n_ids, batch_data, unpacked_codes, ex_data_batch);
            
            // Pack the codes for FastScan
            rabitqlib::fastscan::pack_codes(padded_dim_, unpacked_codes, kBatchSize, batch_bin_code);
            
            // Compute distance estimates using FastScan (with g_add=0)
            rabitqlib::split_batch_estdist(
                batch_data,
                batch_query,
                padded_dim_,
                est_dist,
                low_dist,
                ip_x0_qr,
                rabit_use_hacc_
            );

            if (ex_bits_ > 0) {
                // Apply boosting for extra bits (returns distance with g_add=0)
                for (size_t j = 0; j < batch_size; j++) {
                    dists_out[i + j] = rabitqlib::split_distance_boosting(
                        ex_data_batch + j * ex_data_stride_,
                        excode_ipfunc,
                        batch_query,
                        padded_dim_,
                        ex_bits_,
                        ip_x0_qr[j]
                    );
                }
            } else {
                // Copy results directly
                memcpy(dists_out + i, est_dist, batch_size * sizeof(float));
            }

            // ============================================================
            // Multi-cluster correction: add per-vector g_add
            // Since batch_query has g_add=0, we need to add the query-to-centroid
            // distance (squared) for each vector's assigned cluster.
            // For L2: g_add = ||query - centroid_k||^2
            // ============================================================
            if (!cluster_ids_.empty()) {
                for (size_t j = 0; j < batch_size; j++) {
                    const uint32_t vector_id = ids[i + j];
                    const uint32_t cid = cluster_ids_[vector_id];
                    const float q_norm = rabitq_dc->q_to_centroid_norms_[cid];
                    dists_out[i + j] += q_norm * q_norm;  // g_add for L2 = norm^2
                }
            }
        }
    } else {
        // Fall back to individual distance computation
        for (size_t i = 0; i < n_ids; i++) {
            dists_out[i] = (*dc_computer)(ids[i]);
        }
    }
}

float RabitqQuantizer::compute_distance_to_query(uint32_t row_id, const float* query_rotated, float query_norm) const {
    if (row_id >= num_points_) {
        return std::numeric_limits<float>::max();
    }
    
    // Create SplitSingleQuery object for this query
    rabitqlib::SplitSingleQuery<float> query_wrapper(
        query_rotated, padded_dim_, ex_bits_, query_config_, rabitqlib::METRIC_L2
    );

    float ip_x0_qr;
    float est_dist;
    float low_dist;

    // Calculate offset for this vector's data
    size_t bin_offset = row_id * bin_data_stride_;
    const char* vector_bin_data = bin_data_ + bin_offset;
    
    if (ex_bits_ > 0) {
        // Full split distance computation (includes ex_bits)
        size_t ex_offset = row_id * ex_data_stride_;
        const char* vector_ex_data = ex_data_ + ex_offset;

        rabitqlib::split_single_fulldist(
            vector_bin_data,
            vector_ex_data,
            rabitqlib::select_excode_ipfunc(ex_bits_),
            query_wrapper,
            padded_dim_,
            ex_bits_,
            est_dist,
            low_dist,
            ip_x0_qr,
            query_norm * query_norm,  // g_add for L2 = norm^2
            query_norm                // g_error for L2 = norm
        );
    } else {
        // 1-bit only quantization
        rabitqlib::split_single_estdist(
            vector_bin_data,
            query_wrapper,
            padded_dim_,
            ip_x0_qr,
            est_dist,
            low_dist,
            query_norm * query_norm,  // g_add for L2
            query_norm                // g_error for L2
        );
    }

    return est_dist;
}
