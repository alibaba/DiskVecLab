// fixed_chunk_pq_table_adapter.h
#pragma once
#include "pq_table_base.h"
#include "pq.h" // 包含原始FixedChunkPQTable定义

class FixedChunkPQTableAdapter : public PQTableBase {
  public:
    FixedChunkPQTableAdapter() {} // 初始化参数可能需要调整

    void train(size_t n, const float* x, const std::string data_file, const std::string index_file, const std::string& data_type = "float") override;

    DistanceComputer* preprocess_query(float* query) override;

    void populate_chunk_distances(const float* query, float* out_dists) override;

    uint64_t get_num_chunks() override;

    uint64_t get_dim() override {
        return pq_table.get_num_dims();
    }

    // 扩展功能实现
    void apply_rotation(float* vec) const override {
//        if (pq_table.rotation_applied()) { // 假设存在此方法
//            pq_table.rotate(vec);
//        }
    }

    void compute_dists (const uint32_t *ids, const uint64_t n_ids, float *dists_out,
                       uint8_t *data, uint8_t *pq_coord_scratch, float* pq_dists,
                       DistanceComputer*) override;

    uint64_t get_num_points() override;

    void load_pq_compressed_vectors(const std::string &bin_file, uint8_t* &data, const std::string& data_type = "float") override;

#ifdef EXEC_ENV_OLS
    void load_pq_centroid_bin(MemoryMappedFiles &files, const char *pq_table_file, size_t num_chunks) override;
#else
    void load_pq_centroid_bin(const char *pq_table_file, size_t num_chunks) override;
#endif

  private:
    diskann::FixedChunkPQTable pq_table;
    size_t npt;
    size_t ndim;
};
