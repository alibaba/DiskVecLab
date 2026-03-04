#pragma once
#include "pq_table_base.h"
#include "pq.h" // 包含原始FixedChunkPQTable定义
#include "distance.h"
#include <faiss/IndexAdditiveQuantizer.h>
#include <faiss/MetricType.h>

class LLocalSearchQuantizer : public PQTableBase
{
  public:
    LLocalSearchQuantizer(int d,
                         size_t M,     ///< number of subquantizers
                         size_t nbits);

    void train(size_t n, const float *x, const std::string data_file, const std::string index_file, const std::string& data_type = "float") override;

    // Wrapper class to encapsulate faiss::FlatCodesDistanceComputer
    struct LSQDistanceComputer : public DistanceComputer {
        faiss::FlatCodesDistanceComputer* faiss_dc;
        
        LSQDistanceComputer(faiss::FlatCodesDistanceComputer* dc) : faiss_dc(dc) {}
        
        void set_query(const float* query) override {
            faiss_dc->set_query(query);
        }
        
        float operator()(uint32_t row_id) const override {
            return (*faiss_dc)(row_id);
        }
        
        float symmetric_dis(uint64_t i, uint64_t j) const override {
            return faiss_dc->symmetric_dis(i, j);
        }
        
        float distance_to_code(const uint8_t* code) const override {
            return faiss_dc->distance_to_code(code);
        }
    };

    DistanceComputer* preprocess_query(float *x) override;

    void populate_chunk_distances(const float *query, float *out_dists) override
    {
        // do nothing
    }

    uint64_t get_num_chunks() override
    {
        return 0;
    }

    // 扩展功能实现
    void apply_rotation(float *vec) const override
    {
        //        if (pq_table.rotation_applied()) { // 假设存在此方法
        //            pq_table.rotate(vec);
        //        }
    }

    void compute_dists(const uint32_t *ids, const uint64_t n_ids, float *dists_out, uint8_t *data,
                       uint8_t *pq_coord_scratch, float *pq_dists,
                       DistanceComputer* dc_computer) override;

    uint64_t get_num_points() override
    {
        return index_lsq->ntotal;
    }

    uint64_t get_dim() override
    {
        return index_lsq->d;
    }

    float get_distance_by_row(uint32_t row_id, DistanceComputer* dc_computer) override
    {
        auto* lsq_dc = static_cast<LSQDistanceComputer*>(dc_computer);
        return (*lsq_dc)(row_id);
    }

    void load_pq_compressed_vectors(const std::string &bin_file, uint8_t *&data, const std::string& data_type = "float") override;

#ifdef EXEC_ENV_OLS
    void load_pq_centroid_bin(MemoryMappedFiles &files, const char *pq_table_file, size_t num_chunks) override
    {
    }
#else
    void load_pq_centroid_bin(const char *pq_table_file, size_t num_chunks) override
    {
    }
#endif
  private:
    std::unique_ptr<faiss::IndexLocalSearchQuantizer> index_lsq;
    mutable std::vector<std::unique_ptr<LSQDistanceComputer>> distance_computers_;
};
