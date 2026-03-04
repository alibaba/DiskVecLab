#include "data_partition.h"
#include <type_traits>

namespace diskann {
    // Template implementation for generic data types
    template<typename T>
    void block_convert_copy(std::ifstream& reader, std::ofstream& writer, 
                                    T* read_buf, T* write_buf, _u64 npts, _u64 ndims) {
        if (std::is_same<T, uint8_t>::value) {
            // For bvecs files, we need to read data differently
            // Each vector has 1 uint32_t for dimension followed by ndims uint8_t values
            for (_u64 i = 0; i < npts; i++) {
                // Read the dimension (should match ndims)
                uint32_t dim;
                reader.read((char*)&dim, sizeof(uint32_t));
                if (dim != ndims) {
                    std::cerr << "Dimension mismatch in bvecs file: expected " << ndims 
                              << ", got " << dim << std::endl;
                    exit(-1);
                }
                // Read the actual vector data
                reader.read((char*)(write_buf + i * ndims), ndims * sizeof(uint8_t));
            }
            writer.write((char*)write_buf, npts * ndims * sizeof(uint8_t));
        } else {
            reader.read((char*)read_buf, npts * (ndims + 1) * sizeof(T));
            for (_u64 i = 0; i < npts; i++) {
                memcpy(write_buf + i * ndims, (read_buf + i * (ndims + 1)) + 1,
                       ndims * sizeof(T));
            }
            writer.write((char*)write_buf, npts * ndims * sizeof(T));
        }
    }

    template<typename T>
    void natural_split(std::string base_file, std::string output_prefix, int n) {
        std::ifstream reader(base_file, std::ios::binary | std::ios::ate);
        _u64 fsize = reader.tellg();
        reader.seekg(0, std::ios::beg);

        unsigned ndims_u32;
        reader.read((char*)&ndims_u32, sizeof(unsigned));
        reader.seekg(0, std::ios::beg);
        _u64 ndims = (_u64)ndims_u32;
        
        // For uint8_t type (bvecs files), each vector consists of one int for dimension and ndims uint8_t values
        // So we need to calculate npts differently for uint8_t
        _u64 npts;
        if (std::is_same<T, uint8_t>::value) {
            npts = fsize / (sizeof(unsigned) + ndims);
        } else {
            npts = fsize / ((ndims + 1) * sizeof(T));
        }
        std::cout << "Dataset: #pts = " << npts << ", #dims = " << ndims << std::endl;

        _u64 count_per_file = npts / n;
        _u64 remainder = npts % n;

        _u64 blk_size = 131072;
        
        // For uint8_t type, we need to allocate buffers differently
        T* read_buf;
        T* write_buf;
        if (std::is_same<T, uint8_t>::value) {
            // For bvecs files, each block needs space for blk_size vectors, each with (1 int + ndims uint8_t)
            read_buf = new T[blk_size * (sizeof(unsigned) + ndims)];
            write_buf = new T[blk_size * ndims];
        } else {
            read_buf = new T[blk_size * (ndims + 1)];
            write_buf = new T[blk_size * ndims];
        }
        uint32_t* id_buf = new uint32_t[blk_size];

        for (int i = 0; i < n; ++i) {
            _u64 file_npts = count_per_file + (i < remainder ? 1 : 0);
            if (file_npts == 0) continue;

            std::ostringstream out_name;
            out_name << output_prefix << "_subshard-" << i << ".bin";
            std::cout << out_name.str() << " has " << file_npts << " points." << std::endl;
            std::ofstream writer(out_name.str(), std::ios::binary);
            if (!writer) {
                std::cerr << "Failed to create file: " << out_name.str() << std::endl;
                exit(-1);
            }

            // Write header for vector file
            uint32_t npts_u32 = static_cast<uint32_t>(file_npts);
            uint32_t ndims_u32 = static_cast<uint32_t>(ndims);
            writer.write((char*)&npts_u32, sizeof(uint32_t));
            writer.write((char*)&ndims_u32, sizeof(uint32_t));

            // Create and write ID file
            std::ostringstream out_name_id;
            out_name_id << output_prefix << "_subshard-" << i << "_ids_uint32.bin";
            std::ofstream id_writer(out_name_id.str(), std::ios::binary);
            if (!id_writer) {
                std::cerr << "Failed to create ID file: " << out_name_id.str() << std::endl;
                exit(-1);
            }

            uint32_t dim_id_u32 = static_cast<uint32_t>(1);
            // Write header for ID file
            id_writer.write((char*)&npts_u32, sizeof(uint32_t));
            id_writer.write((char*)&dim_id_u32, sizeof(uint32_t));

            _u64 start_point = 0;
            for (int j = 0; j < i; ++j) {
                start_point += count_per_file + (j < remainder ? 1 : 0);
            }

            // For uint8_t type, seek position is calculated differently
            if (std::is_same<T, uint8_t>::value) {
                // For bvecs files, each vector takes (sizeof(unsigned) + ndims) bytes
                reader.seekg(start_point * (sizeof(unsigned) + ndims), std::ios::beg);
            } else {
                reader.seekg(start_point * (ndims + 1) * sizeof(T), std::ios::beg);
            }

            _u64 remaining = file_npts;
            _u64 current_id = start_point;

            while (remaining > 0) {
                _u64 current_blk_size = std::min(blk_size, remaining);
                block_convert_copy<T>(reader, writer, read_buf, write_buf, current_blk_size, ndims);

                // Generate and write IDs for this block
                for (_u64 j = 0; j < current_blk_size; ++j) {
                    id_buf[j] = static_cast<uint32_t>(current_id + j);
                }
                id_writer.write((char*)id_buf, current_blk_size * sizeof(uint32_t));

                current_id += current_blk_size;
                remaining -= current_blk_size;
            }

            writer.close();
            id_writer.close();
        }

        delete[] read_buf;
        delete[] write_buf;
        delete[] id_buf;

        reader.close();
    }

    // Explicit template instantiations
    template void natural_split<float>(std::string base_file, std::string output_prefix, int n);
    template void natural_split<uint8_t>(std::string base_file, std::string output_prefix, int n);

    void knn_split(std::string data_type, std::string data_path, std::string prefix_path, float sampling_rate, size_t num_partitions, size_t k_index) {
        const size_t max_reps = 15;
        if (data_type == std::string("float"))
            partition<float>(data_path, sampling_rate, num_partitions, max_reps,
                             prefix_path, k_index);
        else if (data_type == std::string("int8"))
            partition<int8_t>(data_path, sampling_rate, num_partitions, max_reps,
                              prefix_path, k_index);
        else if (data_type == std::string("uint8"))
            partition<uint8_t>(data_path, sampling_rate, num_partitions, max_reps,
                               prefix_path, k_index);
        else
            std::cout << "unsupported data format. use float/int8/uint8" << std::endl;
    }


} // diskann
