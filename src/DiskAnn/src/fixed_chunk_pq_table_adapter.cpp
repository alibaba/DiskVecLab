#include "fixed_chunk_pq_table_adapter.h"

void FixedChunkPQTableAdapter::train(size_t n, const float* x, const std::string data_file, const std::string index_file, const std::string& data_type) {
    //FixedChunkPQTable write trained data to file and load into memory when search
    // The data_type parameter is available for future use if needed
}

DistanceComputer* FixedChunkPQTableAdapter::preprocess_query(float* query) {
    pq_table.preprocess_query(query); // 假设原接口兼容
    return nullptr;
}

void FixedChunkPQTableAdapter::populate_chunk_distances(const float* query, float* out_dists) {
    pq_table.populate_chunk_distances(query, out_dists);
}

uint64_t FixedChunkPQTableAdapter::get_num_chunks() {
    return pq_table.get_num_chunks();
}

void FixedChunkPQTableAdapter::compute_dists (const uint32_t *ids, const uint64_t n_ids, float *dists_out,
                   uint8_t *data, uint8_t *pq_coord_scratch, float* pq_dists,
                                             DistanceComputer* dc_computer) {
    // this->data 所有的编码的pq code [1,2,3,4] [1,4,2,4] ......
    const auto chunks = this->get_num_chunks();
    diskann::aggregate_coords(ids, n_ids, data, chunks, pq_coord_scratch);
    diskann::pq_dist_lookup(pq_coord_scratch, n_ids, chunks, pq_dists, dists_out);
}

uint64_t FixedChunkPQTableAdapter::get_num_points() {
    return npt;
}

void FixedChunkPQTableAdapter::load_pq_compressed_vectors(const std::string &bin_file, uint8_t* &data, const std::string& data_type) {
//#ifdef EXEC_ENV_OLS
//    diskann::load_bin<uint8_t>(files, bin_file, data, npt, ndim); // load pq_compressed_vectors
//#else
//    diskann::load_bin<uint8_t>(bin_file, data, npt, ndim);
//#endif
}

#ifdef EXEC_ENV_OLS
void FixedChunkPQTableAdapter::load_pq_centroid_bin(MemoryMappedFiles &files, const char *pq_table_file, size_t num_chunks)
{
    pq_table.load_pq_centroid_bin(files, pq_table_file, num_chunks);
}
#else
void FixedChunkPQTableAdapter::load_pq_centroid_bin(const char *pq_table_file, size_t num_chunks)
{
    pq_table.load_pq_centroid_bin(pq_table_file, num_chunks);
}
#endif


