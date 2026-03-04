#include "llocal_search_quantizer.h"
#include <faiss/index_io.h>

// Template helper function to load and add data to LSQ index
template<typename T>
static void load_and_add_to_lsq(faiss::IndexLocalSearchQuantizer* index_lsq, 
                                const std::string& data_file) {
    size_t read_blk_size = 64 * 1024 * 1024;
    cached_ifstream base_reader(data_file, read_blk_size);
    uint32_t npts32, basedim32;
    base_reader.read((char *)&npts32, sizeof(uint32_t));
    base_reader.read((char *)&basedim32, sizeof(uint32_t));
    size_t num_points = npts32;
    size_t dim = basedim32;

    size_t BLOCK_SIZE = 5000000;
    size_t block_size = num_points <= BLOCK_SIZE ? num_points : BLOCK_SIZE;
    size_t num_blocks = DIV_ROUND_UP(num_points, block_size);

    std::unique_ptr<T[]> block_data_T = std::make_unique<T[]>(block_size * dim);
    std::unique_ptr<float[]> block_data_tmp = std::make_unique<float[]>(block_size * dim);

    for (size_t block = 0; block < num_blocks; block++) {
        size_t start_id = block * block_size;
        size_t end_id = (std::min)((block + 1) * block_size, num_points);
        size_t cur_blk_size = end_id - start_id;

        base_reader.read((char *)(block_data_T.get()), sizeof(T) * (cur_blk_size * dim));
        diskann::convert_types<T, float>(block_data_T.get(), block_data_tmp.get(), cur_blk_size, dim);

        diskann::cout << "Processing points  [" << start_id << ", " << end_id << ").." << std::flush;
        index_lsq->add(cur_blk_size, block_data_tmp.get());
    }
}
LLocalSearchQuantizer::LLocalSearchQuantizer(int d,
                                             size_t M,     ///< number of subquantizers
                                             size_t nbits) {
  index_lsq = std::make_unique<faiss::IndexLocalSearchQuantizer>(d, M, nbits);
}
void LLocalSearchQuantizer::train(size_t n, const float* x, const std::string data_file, const std::string index_file, const std::string& data_type) {
  index_lsq->train(n, x);

  // Dispatch to appropriate template function based on data type
  if (data_type == "float") {
      load_and_add_to_lsq<float>(index_lsq.get(), data_file);
  } else if (data_type == "uint8") {
      load_and_add_to_lsq<uint8_t>(index_lsq.get(), data_file);
  } else if (data_type == "int8") {
      load_and_add_to_lsq<int8_t>(index_lsq.get(), data_file);
  } else {
      throw std::runtime_error("Unsupported data type: " + data_type + ". Supported types: float, uint8, int8");
  }

  faiss::write_index(index_lsq.get(), index_file.c_str());
  diskann::cout << "save index lsq to " << index_file << std::endl;
}
DistanceComputer* LLocalSearchQuantizer::preprocess_query(const float* x) {
  std::lock_guard<std::mutex> guard(mutex_);

  // Get the faiss distance computer
  auto* faiss_dc = index_lsq->get_FlatCodesDistanceComputer();
  faiss_dc->set_query(x);

  // Create our wrapper
  auto wrapper = std::make_unique<LSQDistanceComputer>(faiss_dc);
  LSQDistanceComputer* result = wrapper.get();

  // Store the wrapper for later cleanup
  distance_computers_.push_back(std::move(wrapper));

  return result;
}
void LLocalSearchQuantizer::load_pq_compressed_vectors(const std::string &bin_file, uint8_t* &data) {
  // TODO
  // load the PQ compressed vectors generated in `train`
  // for now, we just keep the vector in memory to avoid the step write and load.
  if (bin_file.empty()) {
    diskann::cout << "data file is empty when train lsq" << std::endl;
    exit(1);
  }
  index_lsq.reset(dynamic_cast<faiss::IndexLocalSearchQuantizer*>(faiss::read_index(bin_file.c_str())));
  diskann::cout << "load index lsq from " << bin_file << std::endl;
}
void LLocalSearchQuantizer::compute_dists (const uint32_t *ids, const uint64_t n_ids, float *dists_out,
                                          uint8_t *data, uint8_t *pq_coord_scratch, float* pq_dists,
                                          DistanceComputer* dc_computer) {
  memset(dists_out, 0, n_ids * sizeof(float));
  auto* lsq_dc = static_cast<LSQDistanceComputer*>(dc_computer);
  for (size_t i = 0; i < n_ids; i++) {
    dists_out[i] = (*lsq_dc)(ids[i]);
  }
}