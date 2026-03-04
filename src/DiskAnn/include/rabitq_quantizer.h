// rabitq_quantizer.h
#pragma once
#include "pq_table_base.h"
#include "pq.h"
#include "distance.h"
#include "utils.h"
#include <memory>
#include <vector>
#include <mutex>
#include "rabitqlib/utils/rotator.hpp"
#include "rabitqlib/quantization/rabitq.hpp"
#include "rabitqlib/index/query.hpp"

// Default number of clusters for RaBitQ multi-cluster quantization.
// Using multiple clusters reduces quantization error by computing residuals
// relative to nearby centroids instead of a single global centroid.
// 16 is a good default matching RaBitQ-Library's HNSW configuration.
static constexpr size_t RABITQ_DEFAULT_NUM_CLUSTERS = 16;

class RabitqQuantizer : public PQTableBase {
  public:
    RabitqQuantizer(size_t d = 0, size_t bits = 8, size_t num_clusters = RABITQ_DEFAULT_NUM_CLUSTERS);

    void train(size_t n, const float* x, const std::string data_file, const std::string index_file, const std::string& data_type = "float") override;

    // For compatibility with existing interface, we'll use a wrapper struct
    struct RabitQDistanceComputer : public DistanceComputer {
        const RabitqQuantizer* quantizer;
        std::vector<float> query_rotated;
        float query_norm;  // kept for backward compat, unused in multi-cluster batch path
        std::unique_ptr<rabitqlib::SplitBatchQuery<float>> batch_query;
        rabitqlib::ex_ipfunc ex_ip_func = nullptr;

        // Precomputed ||query_rotated - centroid_k|| for each cluster k
        std::vector<float> q_to_centroid_norms_;
        
        RabitQDistanceComputer(const RabitqQuantizer* q) : quantizer(q) {}
        
        void set_query(const float* query) override {
            query_rotated.resize(quantizer->padded_dim_);
            quantizer->rotator_->rotate(query, query_rotated.data());

            // Precompute query-to-centroid L2 norms for all clusters
            q_to_centroid_norms_.resize(quantizer->num_clusters_);
            for (size_t k = 0; k < quantizer->num_clusters_; k++) {
                float norm_sq = 0;
                const float* cent = quantizer->rotated_centroids_.data() + k * quantizer->padded_dim_;
                for (size_t i = 0; i < quantizer->padded_dim_; i++) {
                    float diff = query_rotated[i] - cent[i];
                    norm_sq += diff * diff;
                }
                q_to_centroid_norms_[k] = std::sqrt(norm_sq);
            }

            // For backward compat: query_norm = distance to first centroid
            query_norm = q_to_centroid_norms_[0];

            // Create batch query with g_add=0.
            // Per-vector g_add correction (based on each vector's cluster) is applied
            // post-hoc in compute_dists(). This is necessary because different vectors
            // in the same batch may belong to different clusters.
            batch_query = std::make_unique<rabitqlib::SplitBatchQuery<float>>(
                query_rotated.data(),
                quantizer->padded_dim_,
                quantizer->ex_bits_,
                rabitqlib::METRIC_L2,
                quantizer->rabit_use_hacc_);
            // NOTE: g_add is intentionally left at 0 (default).
            // Per-vector correction: est_dist += q_to_centroid_norms_[cid]^2

            ex_ip_func = (quantizer->ex_bits_ > 0) ? rabitqlib::select_excode_ipfunc(quantizer->ex_bits_) : nullptr;
        }
        
        float operator()(uint32_t row_id) const override {
            // Use per-vector centroid norm for distance computation
            uint32_t cid = 0;
            if (!quantizer->cluster_ids_.empty() && row_id < quantizer->cluster_ids_.size()) {
                cid = quantizer->cluster_ids_[row_id];
            }
            float q_norm = q_to_centroid_norms_[cid];
            return quantizer->compute_distance_to_query(row_id, query_rotated.data(), q_norm);
        }
        
        float symmetric_dis(uint64_t i, uint64_t j) const override {
            return 0.0f; // Not implemented for this use case
        }
        
        float distance_to_code(const uint8_t* code) const override {
            return 0.0f; // Not implemented for this use case
        }
    };

    DistanceComputer* preprocess_query(float* query) override;

    void populate_chunk_distances(const float* query, float* out_dists) override {
        // do nothing
    }

    uint64_t get_num_chunks() override {
        return 0;
    }

    void apply_rotation(float* vec) const override {
        if (rotator_ && padded_dim_ > 0) {
            std::vector<float> rotated_vec(padded_dim_);
            rotator_->rotate(vec, rotated_vec.data());
            std::copy(rotated_vec.begin(), rotated_vec.end(), vec);
        }
    }

    void compute_dists (const uint32_t *ids, const uint64_t n_ids, float *dists_out,
                       uint8_t *data, uint8_t *pq_coord_scratch, float* pq_dists,
                       DistanceComputer* dc_computer) override;

    uint64_t get_num_points() override {
        return num_points_;
    }

    uint64_t get_dim() override {
        return dim_;
    }

    float get_distance_by_row(uint32_t row_id, DistanceComputer* dc_computer) override
    {
        // dc_computer supports the operator() interface directly
        return (*dc_computer)(row_id);
    }

    void load_pq_compressed_vectors(const std::string &bin_file, uint8_t* &data, const std::string& data_type = "float") override;

#ifdef EXEC_ENV_OLS
    void load_pq_centroid_bin(MemoryMappedFiles &files, const char *pq_table_file, size_t num_chunks) override
    {
    }
#else
    void load_pq_centroid_bin(const char *pq_table_file, size_t num_chunks) override
    {
    }
#endif

    // Set RabitQ parameters
    void set_rabitq_params(bool use_fastscan, bool use_hacc) {
        use_rabitq_fastscan_ = use_fastscan;
        rabit_use_hacc_ = use_hacc;
    }

    // Set number of clusters for multi-cluster quantization
    void set_num_clusters(size_t num_clusters) { num_clusters_ = num_clusters; }
    size_t get_num_clusters() const { return num_clusters_; }

    // get num of bytes used for 1-bit code and corresponding factors (single mode)
    [[nodiscard]] size_t bin_data_bytes() const {
        return rabitqlib::BinDataMap<float>::data_bytes(padded_dim_) * num_points_;
    }

    [[nodiscard]] size_t ex_data_bytes() const {
        return rabitqlib::ExDataMap<float>::data_bytes(padded_dim_, ex_bits_) * num_points_;
    }

    [[nodiscard]] size_t ids_bytes() const { return sizeof(rabitqlib::PID) * num_points_; }

  private:
    size_t dim_;
    size_t bits_;
    size_t padded_dim_;
    size_t num_points_;
    size_t ex_bits_;
    char* bin_data_ = nullptr;           // 1-bit code and factors (single mode)
    char* ex_data_ = nullptr;            // extra bits code
    size_t size_bin_data_{0};
    size_t size_ex_data_{0};
    size_t bin_data_stride_{0};
    size_t ex_data_stride_{0};
    std::unique_ptr<rabitqlib::Rotator<float>> rotator_;

    // Multi-cluster support: multiple rotated centroids and per-vector cluster assignments
    size_t num_clusters_{1};
    std::vector<float> rotated_centroids_;   // num_clusters_ * padded_dim_ (flattened)
    std::vector<uint32_t> cluster_ids_;      // one cluster_id per vector (size = num_points_)

    rabitqlib::quant::RabitqConfig config_;
    rabitqlib::quant::RabitqConfig query_config_;
    
    // RabitQ parameters
    bool use_rabitq_fastscan_ = true;
    bool rabit_use_hacc_ = true;
    
    mutable std::mutex mutex_;
    
    float compute_distance_to_query(uint32_t row_id, const float* query_rotated, float query_norm) const;
    bool faster = true;

    // Pre-allocated scratch buffers for compute_dists (per-thread via thread_local in .cpp)
    // These are computed once after load and cached
    size_t bin_code_size_per_vector_{0};  // padded_dim_ / 8
    size_t batch_data_size_{0};           // BatchDataMap<float>::data_bytes(padded_dim_)
    size_t ex_data_batch_size_{0};        // ex_data_stride_ * kBatchSize
    size_t unpacked_codes_size_{0};       // bin_code_size_per_vector_ * kBatchSize

    // Initialize scratch buffer sizes after loading
    void init_scratch_buffer_sizes();

    // Helper: swap bytes of uint64_t and write to uint8_t array
    static inline void swap_bytes_u64_to_u8(const uint64_t* src, uint8_t* dst, size_t num_u64);
    
    // Helper: prepare batch data for FastScan computation
    void prepare_batch_data(
        const uint32_t* ids, size_t start_idx, size_t batch_size, size_t n_ids,
        char* batch_data, uint8_t* unpacked_codes, char* ex_data_batch) const;
};
