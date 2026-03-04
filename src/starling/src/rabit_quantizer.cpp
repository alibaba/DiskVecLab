#include "rabitq_quantizer.h"
#include "utils.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include "rabitqlib/quantization/data_layout.hpp"
#include "rabitqlib/index/estimator.hpp"
#include "rabitqlib/utils/memory.hpp"

// Template helper function to handle different input data types
template<typename T>
static void read_and_quantize_data(const std::string& data_file, size_t num_points, size_t dim, 
                                   size_t padded_dim, size_t ex_bits,
                                   const std::vector<float>& rotated_centroid,
                                   rabitqlib::Rotator<float>* rotator,
                                   char* bin_data, char* ex_data,
                                   size_t bin_data_stride, size_t ex_data_stride,
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
        
        // Rotate each vector
        for (size_t i = 0; i < cur_blk_size; i++) {
            float* vector = block_data_tmp.get() + i * dim;
            float* rotated_vector = rotated_data.get() + i * padded_dim;
            rotator->rotate(vector, rotated_vector);

            // Use single mode quantization
            rabitqlib::quant::quantize_split_single(
                rotated_vector,
                rotated_centroid.data(),
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
RabitqQuantizer::RabitqQuantizer(size_t d, size_t bits) : dim_(d), bits_(bits) {
  if (d > 0) {
    rotator_ = std::unique_ptr<rabitqlib::Rotator<float>>(rabitqlib::choose_rotator<float>(d, rabitqlib::RotatorType::FhtKacRotator));
    padded_dim_ = rotator_->size();
    ex_bits_ = bits_ - 1;
    this->query_config_ =
        rabitqlib::quant::faster_config(padded_dim_, rabitqlib::SplitSingleQuery<float>::kNumBits);
    bin_data_stride_ = rabitqlib::BinDataMap<float>::data_bytes(padded_dim_);
    ex_data_stride_ = rabitqlib::ExDataMap<float>::data_bytes(padded_dim_, ex_bits_);
  }
}
void RabitqQuantizer::train(size_t n, const float* x, const std::string data_file, const std::string index_file, const std::string& data_type) {
  // Compute global centroid
  size_t d = dim_;
  std::vector<float> centroid(d, 0);
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < d; j++) {
      centroid[j] += x[i * d + j];
    }
  }
  if (n != 0) {
    for (size_t j = 0; j < d; j++) {
      centroid[j] /= (float)n;
    }
  }

  // Rotate centroid
  std::vector<float> rotated_centroid(padded_dim_, 0);
  rotator_->rotate(centroid.data(), rotated_centroid.data());
  rotated_centroid_ = std::move(rotated_centroid);
  for (int i = 0; i < dim_; i++) {
    diskann::cout << "rorated centroid " << i << " " << rotated_centroid_[i] << " " << std::endl;
  }

  // Read file header to get dimensions
  size_t read_blk_size = 64 * 1024 * 1024;
  cached_ifstream base_reader(data_file, read_blk_size);
  uint32_t npts32, basedim32;
  base_reader.read((char *)&npts32, sizeof(uint32_t));
  base_reader.read((char *)&basedim32, sizeof(uint32_t));
  
  size_t num_points = npts32;
  size_t dim = basedim32;
  num_points_ = num_points;

  // Allocate memory for quantized data (single mode)
  this->bin_data_ = rabitqlib::memory::align_allocate<64, char, true>(bin_data_bytes());
  if (ex_bits_ > 0) {
    this->ex_data_ = rabitqlib::memory::align_allocate<64, char, true>(ex_data_bytes());
  }
  memset(this->bin_data_, 0, bin_data_bytes());
  memset(this->ex_data_, 0, ex_data_bytes());
  
  // Dispatch to appropriate template function based on data type
  if (data_type == "float") {
      read_and_quantize_data<float>(data_file, num_points, dim, padded_dim_, ex_bits_,
                                   rotated_centroid_, rotator_.get(), bin_data_, ex_data_,
                                   bin_data_stride_, ex_data_stride_, query_config_);
  } else if (data_type == "uint8") {
      read_and_quantize_data<uint8_t>(data_file, num_points, dim, padded_dim_, ex_bits_,
                                     rotated_centroid_, rotator_.get(), bin_data_, ex_data_,
                                     bin_data_stride_, ex_data_stride_, query_config_);
  } else if (data_type == "int8") {
      read_and_quantize_data<int8_t>(data_file, num_points, dim, padded_dim_, ex_bits_,
                                    rotated_centroid_, rotator_.get(), bin_data_, ex_data_,
                                    bin_data_stride_, ex_data_stride_, query_config_);
  } else {
      throw std::runtime_error("Unsupported data type: " + data_type + ". Supported types: float, uint8, int8");
  }
  // Save quantized data to file
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
  // Write rotated centroid
  out_file.write(reinterpret_cast<const char*>(rotated_centroid_.data()),
                 sizeof(float) * rotated_centroid_.size());
  // Write quantized data
  out_file.write(reinterpret_cast<const char*>(bin_data_), static_cast<long>(bin_data_bytes()));
  out_file.write(reinterpret_cast<const char*>(ex_data_), static_cast<long>(ex_data_bytes()));
  // write roatator
  rotator_->save(out_file);
  out_file.close();
  diskann::cout << "Saved RaBitQ index to " << index_file << std::endl;
}
DistanceComputer* RabitqQuantizer::preprocess_query(const float* query) {
  std::lock_guard<std::mutex> guard(mutex_);
  auto dc = std::make_unique<RabitQDistanceComputer>(this);
  dc->set_query(query);
  return dc.release();
}
void RabitqQuantizer::load_pq_compressed_vectors(const std::string &bin_file, uint8_t* &data) {
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

  diskann::cout << "Loaded RaBitQ index from " << bin_file << " " <<
      num_points_ << " " << dim_ << " " << padded_dim_ << " " << bits_ << std::endl;

  // Initialize rotator and config
  rotator_ = std::unique_ptr<rabitqlib::Rotator<float>>(rabitqlib::choose_rotator<float>(dim_, rabitqlib::RotatorType::FhtKacRotator));
  padded_dim_ = rotator_->size();
  this->query_config_ =
      rabitqlib::quant::faster_config(padded_dim_, rabitqlib::SplitSingleQuery<float>::kNumBits);

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
  // Read rotated centroid
  rotated_centroid_.resize(padded_dim_);
  in_file.read(reinterpret_cast<char*>(rotated_centroid_.data()),
               sizeof(float) * rotated_centroid_.size());
  for (int i = 0; i < dim_; i++) {
    diskann::cout << "load rorated centroid " << i << " " << rotated_centroid_[i] << " " << std::endl;
  }

  // Read quantized data
  this->bin_data_ = rabitqlib::memory::align_allocate<64, char, true>(bin_data_bytes());
  if (ex_bits_ > 0) {
    this->ex_data_ = rabitqlib::memory::align_allocate<64, char, true>(ex_data_bytes());
  }

  in_file.read(reinterpret_cast<char*>(bin_data_), static_cast<long>(bin_data_bytes()));
  in_file.read(reinterpret_cast<char*>(ex_data_), static_cast<long>(ex_data_bytes()));
  rotator_->load(in_file);
  in_file.close();
  diskann::cout << "Loaded RaBitQ quantized data" << std::endl;
}
void RabitqQuantizer::compute_dists(const uint32_t *ids, const uint64_t n_ids, float *dists_out,
                                    uint8_t *data, uint8_t *pq_coord_scratch, float* pq_dists,
                                    DistanceComputer* dc_computer) {
  memset(dists_out, 0, n_ids * sizeof(float));
  for (size_t i = 0; i < n_ids; i++) {
    dists_out[i] = (*dc_computer)(ids[i]);
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
    // 使用完整的split距离计算 (包含ex_bits)
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
        query_norm * query_norm,  // g_add for L2
        query_norm                // g_error for L2
    );
  } else {
    // 只使用1-bit量化
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