#pragma once
#include "pq_table_base.h"
#include "pq.h"
#include "distance.h"
#include <memory>
#include <vector>

class FullPrecisionQuantizer : public PQTableBase {
  public:
    FullPrecisionQuantizer(size_t d = 0);
    
    void train(size_t n, const float* x, const std::string data_file, const std::string index_file, const std::string& data_type = "float") override;

    struct FullPrecisionDistanceComputer : public DistanceComputer {
        const FullPrecisionQuantizer* quantizer;
        std::vector<float> query_vec;
        
        FullPrecisionDistanceComputer(const FullPrecisionQuantizer* q) : quantizer(q) {}
        
        void set_query(const float* query) override {
            query_vec.assign(query, query + quantizer->dim_);
        }
        
        float operator()(uint32_t row_id) const override {
            return quantizer->compute_distance_to_query(row_id, query_vec.data());
        }
    };

    DistanceComputer* preprocess_query(float* query) override;

    void populate_chunk_distances(const float* query, float* out_dists) override {
        // do nothing - no chunks in full precision mode
    }

    uint64_t get_num_chunks() override {
        return 0; // No quantization chunks
    }

    void apply_rotation(float* vec) const override {
        // No rotation needed for full precision
    }

    void compute_dists(const uint32_t *ids, const uint64_t n_ids, float *dists_out,
                       uint8_t *data, uint8_t *pq_coord_scratch, float* pq_dists,
                       DistanceComputer* dc_computer) override;

    uint64_t get_num_points() override {
        return num_points_;
    }

    uint64_t get_dim() override {
        return dim_;
    }

    float get_distance_by_row(uint32_t row_id, DistanceComputer* dc_computer) override {
        return (*dc_computer)(row_id);
    }

    void load_pq_compressed_vectors(const std::string &bin_file, uint8_t* &data, const std::string& data_type = "float") override;

#ifdef EXEC_ENV_OLS
    void load_pq_centroid_bin(MemoryMappedFiles &files, const char *pq_table_file, size_t num_chunks) override {
        // No centroids needed for full precision
    }
#else
    void load_pq_centroid_bin(const char *pq_table_file, size_t num_chunks) override {
        // No centroids needed for full precision
    }
#endif

  private:
    size_t dim_;
    size_t num_points_;
    std::vector<float> vectors_; // Store all vectors in full precision
    
    float compute_distance_to_query(uint32_t row_id, const float* query) const;
};