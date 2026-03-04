// rabitq_quantizer.h
#pragma once
#include "pq_table_base.h"
#include "pq_table.h"
#include "distance.h"
#include "utils.h"
#include <memory>
#include <vector>
#include <mutex>
#include "rabitqlib/utils/rotator.hpp"
#include "rabitqlib/quantization/rabitq.hpp"
class RabitqQuantizer : public PQTableBase {
 public:
  RabitqQuantizer(size_t d = 0, size_t bits = 8);
  void train(size_t n, const float* x, const std::string data_file, const std::string index_file, const std::string& data_type = "float") override;
  // For compatibility with existing interface, we'll use a wrapper struct
  struct RabitQDistanceComputer : public DistanceComputer {
    const RabitqQuantizer* quantizer;
    std::vector<float> query_rotated;
    float query_norm;

    RabitQDistanceComputer(const RabitqQuantizer* q) : quantizer(q) {}

    void set_query(const float* query) override {
      query_rotated.resize(quantizer->padded_dim_);
      quantizer->rotator_->rotate(query, query_rotated.data());
      query_norm = 0;
      for (size_t i = 0; i < quantizer->padded_dim_; i++) {
        float diff = query_rotated[i] * quantizer->rotated_centroid_[i];
        query_norm += diff * diff;
      }
      query_norm = std::sqrt(query_norm);
    }

    float operator()(uint32_t row_id) const override {
      return quantizer->compute_distance_to_query(row_id, query_rotated.data(), query_norm);
    }

    float symmetric_dis(uint64_t i, uint64_t j) const override {
      return 0.0f; // Not implemented for this use case
    }

    float distance_to_code(const uint8_t* code) const override {
      return 0.0f; // Not implemented for this use case
    }
  };
  DistanceComputer* preprocess_query(const float* query) override;
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
  void load_pq_compressed_vectors(const std::string &bin_file, uint8_t* &data) override;
#ifdef EXEC_ENV_OLS
  void load_pq_centroid_bin(MemoryMappedFiles &files, const char *pq_table_file, size_t num_chunks) override
  {
  }
#else
  void load_pq_centroid_bin(const char *pq_table_file, size_t num_chunks) override
  {
  }
#endif
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
  std::vector<float> rotated_centroid_;
  rabitqlib::quant::RabitqConfig query_config_;

  mutable std::mutex mutex_;

  float compute_distance_to_query(uint32_t row_id, const float* query_rotated, float query_norm) const;
};