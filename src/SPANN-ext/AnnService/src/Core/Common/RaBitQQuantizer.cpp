#include "inc/Core/Common/RaBitQQuantizer.h"
#include <algorithm>
#include <limits>
#include <cstring>
#include <sstream>
#include <fstream>
#include <execinfo.h>
#include <cstdlib>
#include <dlfcn.h>

// Include RaBitQ library headers
#include "inc/Core/Common/rabitqlib/quantization/data_layout.hpp"
#include "inc/Core/Common/rabitqlib/index/estimator.hpp"
#include "inc/Core/Common/rabitqlib/utils/memory.hpp"

namespace SPTAG {
    namespace COMMON {
        RaBitQQuantizer::RaBitQQuantizer(DimensionType dim, SizeType bits, int64_t num_points)
        : m_dim(dim), m_bits(bits), m_num_points(num_points), m_quantization_counter(0) {
            if (dim > 0) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "point %d, m_bits %d!\n", num_points, m_bits);
                InitializeRotator();
                m_ex_bits = m_bits - 1;
                m_query_config = rabitqlib::quant::faster_config(m_padded_dim, rabitqlib::SplitSingleQuery<float>::kNumBits);
                
                // Initialize DiskAnn format strides
                m_diskann_bin_data_stride = rabitqlib::BinDataMap<float>::data_bytes(m_padded_dim);
                m_diskann_ex_data_stride = rabitqlib::ExDataMap<float>::data_bytes(m_padded_dim, m_ex_bits);

                // Initialize debug storage for original vectors
                m_debug_enabled = false;
                m_original_vectors_map.clear();

                //InitializeDualFormatStorage(m_num_points);
            }
        }

        void RaBitQQuantizer::InitializeRotator() {
            // Always use float type with FhtKacRotator for best performance
            m_rotator = std::unique_ptr<rabitqlib::Rotator<float>>(
                rabitqlib::choose_rotator<float>(m_dim, rabitqlib::RotatorType::FhtKacRotator));
            m_padded_dim = m_rotator->size();
        }


        float RaBitQQuantizer::L2Distance(const std::uint8_t* pX, const std::uint8_t* pY) const {
            // pX is the query vector (quantized with embedded cache)
            // pY is the posting list vector (quantized) - not used, we use row_id to access DiskAnn format data
            // row_id is the actual vector ID from the posting list

            // Create SplitSingleQuery object for this query
            const float* query_vec = reinterpret_cast<const float*>(pX);
            rabitqlib::SplitSingleQuery<float> query_wrapper(
                query_vec, m_padded_dim, m_ex_bits, m_query_config, rabitqlib::METRIC_L2
            );


            float query_norm = *(query_vec + m_padded_dim);
            //SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "query norm  in l2 distance %f\n", query_norm);

            float ip_x0_qr;
            float est_dist;
            float low_dist;

            if (m_ex_bits > 0) {
                // Use full split distance calculation (including ex_bits)

                rabitqlib::split_single_fulldist(
                    reinterpret_cast<const char*>(pY),
                    reinterpret_cast<const char*>(pY + m_diskann_bin_data_stride),
                    rabitqlib::select_excode_ipfunc(m_ex_bits),
                    query_wrapper,
                    m_padded_dim,
                    m_ex_bits,
                    est_dist,
                    low_dist,
                    ip_x0_qr,
                    query_norm * query_norm,  // g_add for L2
                    query_norm                // g_error for L2
                );
            } else {
                // Use 1-bit quantization only
                rabitqlib::split_single_estdist(
                    reinterpret_cast<const char*>(pY),
                    query_wrapper,
                    m_padded_dim,
                    ip_x0_qr,
                    est_dist,
                    low_dist,
                    query_norm * query_norm,  // g_add for L2
                    query_norm                // g_error for L2
                );
            }

            return est_dist;
        }

        float RaBitQQuantizer::CosineDistance(const std::uint8_t* pX, const std::uint8_t* pY) const {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "RaBitQ quantizer does not support CosineDistance!\n");
            return 0.0f;
        }

        DimensionType RaBitQQuantizer::GetNumSubvectors() const {
          SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "RaBitQ quantizer does not support GetNumSubvectors\n");
          return QuantizeSize();
        }

        void RaBitQQuantizer::QuantizeVector(const void* vec, std::uint8_t* vecout, bool ADC) const {
            // Call the row ID version with internal counter
            //std::lock_guard<std::mutex> lock(m_mutex);
            QuantizeVector(vec, vecout, m_quantization_counter, ADC);
            //m_quantization_counter++;
        }

        // adc = true means this the vec is query and the function is called in the process when search
        void RaBitQQuantizer::QuantizeVector(const void* vec, std::uint8_t* vecout, SizeType row_id, bool ADC) const {
            //SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "quantize vector! rowid %d\n", row_id);
            // Convert input to float if needed and perform rotation
            const float* input_vec = static_cast<const float*>(vec);
            //std::lock_guard<std::mutex> lock(m_mutex);
            std::vector<float> float_vec(m_dim);
            std::copy(input_vec, input_vec + m_dim, float_vec.data());

            // Store original vector for debugging
            if (m_debug_enabled) {  // Only store for non-query vectors
                std::lock_guard<std::mutex> lock(m_debug_mutex);
                // Explicitly create a copy to ensure memory safety
                std::vector<float> vector_copy(float_vec);  // Explicit copy constructor
                m_original_vectors_map[row_id] = std::move(vector_copy);  // Move the copy
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "DEBUG: Stored original vector copy for row_id %d\n", row_id);
            }

            ByteArray rotated_data = ByteArray::Alloc(sizeof(float) * m_padded_dim);
            m_rotator->rotate(float_vec.data(), reinterpret_cast<float*>(rotated_data.Data()));
            const float* query_rotated = reinterpret_cast<const float*>(rotated_data.Data());
            
            // Store in SPTAG format for reconstruction compatibility
            //StoreSPTAGFormat(reinterpret_cast<const float*>(rotated_data.Data()), vecout, row_id);
            if (is_load) {
                // means the vector is query
                //std::memcpy(query_rotated, query_rotated + m_padded_dim, vecout);
                // float* vecout_float = reinterpret_cast<float*>(vecout);
                // std::copy(query_rotated, query_rotated + m_padded_dim, vecout_float);
                std::memcpy(vecout, query_rotated, sizeof(float) * m_padded_dim);
                // calcuate norm
                float query_norm = 0;
                for (size_t i = 0; i < m_padded_dim; i++) {
                     float diff = query_rotated[i] - m_rotated_centroid[i];
                     query_norm += diff * diff;
                }
                query_norm = std::sqrt(query_norm);
                float* temp = reinterpret_cast<float*>(vecout + sizeof(float) * m_padded_dim);
                *temp = query_norm;
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "norm %f\n", query_norm);
                return;
            } else {
                // Use DiskAnn's quantization method
                rabitqlib::quant::quantize_split_single(
                    query_rotated,
                    m_rotated_centroid.data(),
                    m_padded_dim,
                    m_ex_bits,
                    reinterpret_cast<char*>(vecout),
                    reinterpret_cast<char*>(vecout + m_diskann_bin_data_stride),
                    rabitqlib::METRIC_L2,
                    m_query_config
                );
            }

            // 保存SPTAG格式数据到内存中用于比较
            //if (!is_load && row_id < m_sptag_quantized_data.size()) {
             //   std::copy(vecout, vecout + QuantizeSize(), m_sptag_quantized_data[row_id].begin());
            //}

            //if (!is_load) {
            //  StoreDiskAnnFormat(row_id, reinterpret_cast<const float*>(rotated_data.Data()));
            //}

        }

        SizeType RaBitQQuantizer::QuantizeSize() const {
            //if(m_EnableADC) {
            //  return m_padded_dim * sizeof(float) + sizeof(float);
            //}
            if (is_load) {
                // quantize query
                // origin query vector + query_norm
                return m_padded_dim * sizeof(float) + sizeof(float);
            }
            return m_diskann_bin_data_stride + m_diskann_ex_data_stride;
        }

        void RaBitQQuantizer::ReconstructVector(const std::uint8_t* qvec, void* vecout) const {
            // Extract delta and vl values (ignore row ID for reconstruction)
            const float* delta = reinterpret_cast<const float*>(qvec + m_padded_dim);
            const float* vl = delta + 1;
            
            // Reconstruct the vector using the scalar quantization method
            rabitqlib::quant::reconstruct_vec(qvec, *delta, *vl, m_padded_dim, (float*)vecout);
        }

        SizeType RaBitQQuantizer::ReconstructSize() const {
            return sizeof(float) * ReconstructDim();
        }

        DimensionType RaBitQQuantizer::ReconstructDim() const {
            return m_dim;
        }

        std::uint64_t RaBitQQuantizer::BufferSize() const {
            // Estimate rotator size - this is a rough estimate as the actual size 
            // depends on the rotator implementation
            SizeType estimated_rotator_size = m_padded_dim * sizeof(float); // Rough estimate
            
            return sizeof(DimensionType) * 2 + sizeof(SizeType) * 7 +  // Added extra strides for dual format + rotator size
                   sizeof(float) * m_rotated_centroid.size() + 
                   GetDiskAnnBinDataBytes() + GetDiskAnnExDataBytes() +  // DiskAnn format data
                   estimated_rotator_size +  // Estimated rotator data size
                   sizeof(VectorValueType) + sizeof(QuantizerType);
        }

        ErrorCode RaBitQQuantizer::SaveQuantizer(std::shared_ptr<Helper::DiskIO> p_out) const {
            QuantizerType qtype = QuantizerType::RaBitQ;
            VectorValueType rtype = VectorValueType::Float;
            
            IOBINARY(p_out, WriteBinary, sizeof(QuantizerType), (char*)&qtype);
            IOBINARY(p_out, WriteBinary, sizeof(VectorValueType), (char*)&rtype);
            IOBINARY(p_out, WriteBinary, sizeof(DimensionType), (char*)&m_dim);
            IOBINARY(p_out, WriteBinary, sizeof(DimensionType), (char*)&m_padded_dim);
            IOBINARY(p_out, WriteBinary, sizeof(SizeType), (char*)&m_bits);
            IOBINARY(p_out, WriteBinary, sizeof(SizeType), (char*)&m_ex_bits);
            IOBINARY(p_out, WriteBinary, sizeof(SizeType), (char*)&m_num_points);
            
            // Save DiskAnn format strides
            IOBINARY(p_out, WriteBinary, sizeof(SizeType), (char*)&m_diskann_bin_data_stride);
            IOBINARY(p_out, WriteBinary, sizeof(SizeType), (char*)&m_diskann_ex_data_stride);
            
            // Save rotated centroid
            IOBINARY(p_out, WriteBinary, sizeof(float) * m_rotated_centroid.size(), (char*)m_rotated_centroid.data());

            // Save rotator
            if (m_rotator) {
                // Create a temporary file to save the rotator
                std::string temp_filename = "temp_rotator.bin";
                std::ofstream temp_file(temp_filename, std::ios::binary);
                m_rotator->save(temp_file);
                temp_file.close();
                
                // Read the file content
                std::ifstream temp_read(temp_filename, std::ios::binary | std::ios::ate);
                SizeType rotator_size = static_cast<SizeType>(temp_read.tellg());
                temp_read.seekg(0, std::ios::beg);
                std::vector<char> rotator_data(rotator_size);
                temp_read.read(rotator_data.data(), rotator_size);
                temp_read.close();
                
                // Delete the temporary file
                std::remove(temp_filename.c_str());
                
                IOBINARY(p_out, WriteBinary, sizeof(SizeType), (char*)&rotator_size);
                IOBINARY(p_out, WriteBinary, rotator_size, (char*)rotator_data.data());
            } else {
                SizeType rotator_size = 0;
                IOBINARY(p_out, WriteBinary, sizeof(SizeType), (char*)&rotator_size);
            }
            
            // Save original vectors for debugging
            //SaveDebugVectors();
            
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Saving RaBitQ quantizer: dim:%d padded_dim:%d bits:%d\n", m_dim, m_padded_dim, m_bits);
            return ErrorCode::Success;
        }

        ErrorCode RaBitQQuantizer::LoadQuantizer(std::shared_ptr<Helper::DiskIO> p_in) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Loading RaBitQ Quantizer.\n");
            is_load = true;
            IOBINARY(p_in, ReadBinary, sizeof(DimensionType), (char*)&m_dim);
            IOBINARY(p_in, ReadBinary, sizeof(DimensionType), (char*)&m_padded_dim);
            IOBINARY(p_in, ReadBinary, sizeof(SizeType), (char*)&m_bits);
            IOBINARY(p_in, ReadBinary, sizeof(SizeType), (char*)&m_ex_bits);
            IOBINARY(p_in, ReadBinary, sizeof(SizeType), (char*)&m_num_points);
            
            // Load DiskAnn format strides
            IOBINARY(p_in, ReadBinary, sizeof(SizeType), (char*)&m_diskann_bin_data_stride);
            IOBINARY(p_in, ReadBinary, sizeof(SizeType), (char*)&m_diskann_ex_data_stride);

            // Initialize rotator and query config
            InitializeRotator();
            m_query_config = rabitqlib::quant::faster_config(m_padded_dim, rabitqlib::SplitSingleQuery<float>::kNumBits);

            // Load rotated centroid
            m_rotated_centroid.resize(m_padded_dim);
            IOBINARY(p_in, ReadBinary, sizeof(float) * m_rotated_centroid.size(), (char*)m_rotated_centroid.data());
            
            // Load rotator
            SizeType rotator_size = 0;
            IOBINARY(p_in, ReadBinary, sizeof(SizeType), (char*)&rotator_size);
            if (rotator_size > 0) {
                std::vector<char> rotator_data(rotator_size);
                IOBINARY(p_in, ReadBinary, rotator_size, (char*)rotator_data.data());
                
                // Create a temporary file to load the rotator
                std::string temp_filename = "temp_rotator.bin";
                std::ofstream temp_file(temp_filename, std::ios::binary);
                temp_file.write(rotator_data.data(), rotator_size);
                temp_file.close();
                
                // Load from the temporary file
                std::ifstream temp_read(temp_filename, std::ios::binary);
                m_rotator->load(temp_read);
                temp_read.close();
                
                // Delete the temporary file
                std::remove(temp_filename.c_str());
            }



            // Initialize debug mode and try to load debug vectors
            m_debug_enabled = false;

            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Loaded RaBitQ quantizer: dim:%d padded_dim:%d bits:%d\n", m_dim, m_padded_dim, m_bits);
            return ErrorCode::Success;
        }

        ErrorCode RaBitQQuantizer::LoadQuantizer(std::uint8_t* raw_bytes) {
            // Implementation for loading from raw bytes
            return ErrorCode::Success;
        }

        int RaBitQQuantizer::GetBase() const {
            return COMMON::Utils::GetBase<float>();
        }

        void RaBitQQuantizer::Train(SizeType n, const float* x) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Training RaBitQ quantizer with %d points\n", n);
            
            // Reset quantization counter for new training session
            m_quantization_counter = 0;
            
            // Compute global centroid (similar to DiskAnn approach)
            std::vector<float> centroid(m_dim, 0.0f);
            for (SizeType i = 0; i < n; i++) {
                for (DimensionType j = 0; j < m_dim; j++) {
                    centroid[j] += x[i * m_dim + j];
                }
            }
            
            if (n > 0) {
                for (DimensionType j = 0; j < m_dim; j++) {
                    centroid[j] /= static_cast<float>(n);
                }
            }
            
            // Rotate centroid
            m_rotated_centroid.resize(m_padded_dim, 0.0f);
            m_rotator->rotate(centroid.data(), m_rotated_centroid.data());

            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "RaBitQ training completed with dual format storage\n");
        }

        float RaBitQQuantizer::ComputeDistanceToQuery(SizeType row_id, const float* query_rotated, float query_norm) const {
            // Use fast distance calculation with DiskAnn format
            return ComputeFastDistanceToQuery(row_id, query_rotated, query_norm);
        }

        void RaBitQQuantizer::InitializeDualFormatStorage(SizeType num_points) {
            m_num_points = num_points;

            // Initialize DiskAnn format storage
            SizeType diskann_bin_total = GetDiskAnnBinDataBytes();
            SizeType diskann_ex_total = GetDiskAnnExDataBytes();

            if (diskann_bin_total > 0) {
                m_diskann_bin_data = std::unique_ptr<char[]>(new char[diskann_bin_total]);
                std::memset(m_diskann_bin_data.get(), 0, diskann_bin_total);
            }

            if (diskann_ex_total > 0 && m_ex_bits > 0) {
                m_diskann_ex_data = std::unique_ptr<char[]>(new char[diskann_ex_total]);
                std::memset(m_diskann_ex_data.get(), 0, diskann_ex_total);
            }

            // Initialize SPTAG format storage for memory comparison
            m_sptag_quantized_data.resize(num_points);
            for (SizeType i = 0; i < num_points; i++) {
                m_sptag_quantized_data[i].resize(QuantizeSize());
            }
        }

        SizeType RaBitQQuantizer::GetDiskAnnBinDataBytes() const {
            return m_diskann_bin_data_stride * m_num_points;
        }

        SizeType RaBitQQuantizer::GetDiskAnnExDataBytes() const {
            return m_diskann_ex_data_stride * m_num_points;
        }

        void RaBitQQuantizer::StoreDiskAnnFormat(SizeType point_id, const float* rotated_vec) const {
            if (!m_diskann_bin_data || point_id >= m_num_points) return;
            
            // Calculate offsets for this point
            char* bin_data_ptr = m_diskann_bin_data.get() + point_id * m_diskann_bin_data_stride;
            char* ex_data_ptr = (m_diskann_ex_data && m_ex_bits > 0) ? 
                                m_diskann_ex_data.get() + point_id * m_diskann_ex_data_stride : nullptr;
            
            // Use DiskAnn's quantization method
            rabitqlib::quant::quantize_split_single(
                rotated_vec,
                m_rotated_centroid.data(),
                m_padded_dim,
                m_ex_bits,
                bin_data_ptr,
                ex_data_ptr,
                rabitqlib::METRIC_L2,
                m_query_config
            );
        }

        void RaBitQQuantizer::StoreSPTAGFormat(const float* rotated_vec, std::uint8_t* vecout, SizeType row_id) const {
            // Use original SPTAG quantization method (scalar quantization)
            float delta, vl;
            for(int i=0;i<m_padded_dim;i++) {
              //SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "rowid %d before rotated_vec[%d]=%f\n", row_id, i, rotated_vec[i]);
            }
            rabitqlib::quant::quantize_scalar<float, std::uint8_t>(
                rotated_vec, 
                (size_t)m_padded_dim, 
                (size_t)m_bits, 
                vecout, 
                delta, 
                vl, 
                m_query_config
            );
//            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "RaBitQ quantizer quantize rowid %d, delta %f, vl %f!\n", row_id, delta, vl);
            
            // Store delta and vl values at the end
            float* temp = reinterpret_cast<float*>(vecout + m_padded_dim);
            *temp = delta;
            *(temp + 1) = vl;
            
            // Store row ID for later reference
            SizeType* row_id_ptr = reinterpret_cast<SizeType*>(temp + 2);
            *row_id_ptr = row_id;

            // Test reconstruction
            std::unique_ptr<float[]> rotated_vec_test(new float[m_padded_dim]);
            rabitqlib::quant::reconstruct_vec(vecout, delta, vl, m_padded_dim, rotated_vec_test.get());
            for(int i=0; i< m_padded_dim; i++) {
              //SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "rowid %d after reconstruct rotated_vec[%d]=%f\n", row_id, i, rotated_vec_test[i]);
            }
        }

        float RaBitQQuantizer::ComputeFastDistanceToQuery(SizeType row_id, const float* query_rotated, float query_norm) const {
            if (!m_diskann_bin_data || row_id >= m_num_points) {
                return std::numeric_limits<float>::max();
            }

            // Create SplitSingleQuery object for this query
            rabitqlib::SplitSingleQuery<float> query_wrapper(
                query_rotated, m_padded_dim, m_ex_bits, m_query_config, rabitqlib::METRIC_L2
            );

            float ip_x0_qr;
            float est_dist;
            float low_dist;

            // Calculate offset for this vector's data
            size_t bin_offset = row_id * m_diskann_bin_data_stride;
            const char* vector_bin_data = m_diskann_bin_data.get() + bin_offset;

            if (m_ex_bits > 0 && m_diskann_ex_data) {
                // Use full split distance calculation (including ex_bits)
                size_t ex_offset = row_id * m_diskann_ex_data_stride;
                const char* vector_ex_data = m_diskann_ex_data.get() + ex_offset;

                rabitqlib::split_single_fulldist(
                    vector_bin_data,
                    vector_ex_data,
                    rabitqlib::select_excode_ipfunc(m_ex_bits),
                    query_wrapper,
                    m_padded_dim,
                    m_ex_bits,
                    est_dist,
                    low_dist,
                    ip_x0_qr,
                    query_norm * query_norm,  // g_add for L2
                    query_norm                // g_error for L2
                );
            } else {
                // Use only 1-bit quantization
                rabitqlib::split_single_estdist(
                    vector_bin_data,
                    query_wrapper,
                    m_padded_dim,
                    ip_x0_qr,
                    est_dist,
                    low_dist,
                    query_norm * query_norm,  // g_add for L2
                    query_norm                // g_error for L2
                );
            }

            return est_dist;
        }

        void RaBitQQuantizer::SaveDebugVectors() const {
            if (!m_debug_enabled || m_original_vectors_map.empty()) {
                return;
            }

            std::string debug_filename = "rabitq_debug_vectors.bin";
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Saving debug vectors to %s (%zu vectors)\n", 
                debug_filename.c_str(), m_original_vectors_map.size());

            std::ofstream debug_file(debug_filename, std::ios::binary);
            if (!debug_file.is_open()) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to create debug vectors file: %s\n", debug_filename.c_str());
                return;
            }

            // Write header information
            SizeType num_vectors = static_cast<SizeType>(m_original_vectors_map.size());
            debug_file.write(reinterpret_cast<const char*>(&num_vectors), sizeof(SizeType));
            debug_file.write(reinterpret_cast<const char*>(&m_dim), sizeof(DimensionType));

            // Write vectors with their row IDs
            std::lock_guard<std::mutex> lock(m_debug_mutex);
            for (const auto& pair : m_original_vectors_map) {
                SizeType row_id = pair.first;
                const std::vector<float>& vector = pair.second;
                
                // Write row ID
                debug_file.write(reinterpret_cast<const char*>(&row_id), sizeof(SizeType));
                
                // Write vector data
                debug_file.write(reinterpret_cast<const char*>(vector.data()), sizeof(float) * m_dim);
            }

            debug_file.close();
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Debug vectors saved successfully\n");
        }

        void RaBitQQuantizer::LoadDebugVectors(const std::string& filename) {
            if (!m_debug_enabled) {
                return;
            }

            std::ifstream debug_file(filename, std::ios::binary);
            if (!debug_file.is_open()) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Debug vectors file not found: %s\n", filename.c_str());
                return;
            }

            // Read header information
            SizeType num_vectors;
            DimensionType file_dim;
            debug_file.read(reinterpret_cast<char*>(&num_vectors), sizeof(SizeType));
            debug_file.read(reinterpret_cast<char*>(&file_dim), sizeof(DimensionType));

            if (file_dim != m_dim) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Debug vectors dimension mismatch: file=%d, quantizer=%d\n", 
                    file_dim, m_dim);
                debug_file.close();
                return;
            }

            // Read vectors
            std::lock_guard<std::mutex> lock(m_debug_mutex);
            m_original_vectors_map.clear();
            
            for (SizeType i = 0; i < num_vectors; i++) {
                SizeType row_id;
                std::vector<float> vector(m_dim);
                
                // Read row ID
                debug_file.read(reinterpret_cast<char*>(&row_id), sizeof(SizeType));
                
                // Read vector data
                debug_file.read(reinterpret_cast<char*>(vector.data()), sizeof(float) * m_dim);
                
                m_original_vectors_map[row_id] = std::move(vector);
            }

            debug_file.close();
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Loaded %zu debug vectors from %s\n", 
                m_original_vectors_map.size(), filename.c_str());
        }
      }
}