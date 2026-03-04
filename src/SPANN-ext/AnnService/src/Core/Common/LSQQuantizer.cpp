#include "inc/Core/Common/LSQQuantizer.h"
#include <faiss/index_io.h>
#include <faiss/utils/distances.h>
#include <cstring>

namespace SPTAG {
    namespace COMMON {
        LSQQuantizer::LSQQuantizer(DimensionType dim, SizeType num_subvectors, SizeType nbits) {
            m_index_lsq = std::make_unique<faiss::IndexLocalSearchQuantizer>(dim, num_subvectors, nbits);
            _num_subvectors = num_subvectors;
        }

        float LSQQuantizer::L2Distance(const std::uint8_t* pX, const std::uint8_t* pY) const {
            // Use FAISS distance computer - disabled for now
            float originX[m_index_lsq->d];
            float originy[m_index_lsq->d];
            m_index_lsq->sa_decode(1, pX, originX);
            m_index_lsq->sa_decode(1, pY, originy);

			float res = faiss::fvec_L2sqr(originX, originy, m_index_lsq->d);
            return res;
        }

        float LSQQuantizer::CosineDistance(const std::uint8_t* pX, const std::uint8_t* pY) const {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "LSQ quantizer does not support CosineDistance!\n");
            return 0.0f;
        }

        void LSQQuantizer::QuantizeVector(const void* vec, std::uint8_t* vecout, bool ADC) const {
            // Convert input to float and quantize
            // Add mutex lock to prevent concurrent access
            std::lock_guard<std::mutex> lock(m_mutex);

            const float* input_vec = static_cast<const float*>(vec);
            m_index_lsq->sa_encode(1, input_vec, vecout);
        }

        SizeType LSQQuantizer::QuantizeSize() const {
            //SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "lsq code size %d\n", m_index_lsq->code_size);
            return m_index_lsq->code_size;
        }

        void LSQQuantizer::ReconstructVector(const std::uint8_t* qvec, void* vecout) const {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_index_lsq->sa_decode(1, qvec, static_cast<float*>(vecout));
        }

        SizeType LSQQuantizer::ReconstructSize() const {
            return sizeof(float) * ReconstructDim();
        }

        DimensionType LSQQuantizer::ReconstructDim() const {
            return m_index_lsq->d;
        }

        std::uint64_t LSQQuantizer::BufferSize() const {
            // Estimate buffer size needed - FAISS doesn't provide direct method
            return 0;
        }

        ErrorCode LSQQuantizer::SaveQuantizer(std::shared_ptr<Helper::DiskIO> p_out) const {
            QuantizerType qtype = QuantizerType::LSQ;
            VectorValueType rtype = VectorValueType::Float;
            
            IOBINARY(p_out, WriteBinary, sizeof(QuantizerType), (char*)&qtype);
            IOBINARY(p_out, WriteBinary, sizeof(VectorValueType), (char*)&rtype);

            // Save FAISS index to a temporary file and then read it back
            std::string tmp_file = "/tmp/lsq_quantizer_tmp.idx";
            faiss::write_index(m_index_lsq.get(), tmp_file.c_str());

            // Read the temporary file and write to output
            std::ifstream file(tmp_file, std::ios::binary | std::ios::ate);
            if (!file.is_open()) {
                return ErrorCode::Fail;
            }

            std::streamsize size = file.tellg();
            file.seekg(0, std::ios::beg);

            std::vector<char> buffer(size);
            if (!file.read(buffer.data(), size)) {
                return ErrorCode::Fail;
            }
            file.close();

            // Write the buffer size first, then the buffer
            std::uint64_t buffer_size = static_cast<std::uint64_t>(size);
            IOBINARY(p_out, WriteBinary, sizeof(std::uint64_t), (char*)&buffer_size);
            IOBINARY(p_out, WriteBinary, size, buffer.data());

            // Clean up temporary file
            std::remove(tmp_file.c_str());

            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Saving LSQ quantizer completed\n");
            return ErrorCode::Success;
        }

        ErrorCode LSQQuantizer::LoadQuantizer(std::shared_ptr<Helper::DiskIO> p_in) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Loading LSQ Quantizer.\n");

            // Read the buffer size
            std::uint64_t buffer_size;
            IOBINARY(p_in, ReadBinary, sizeof(std::uint64_t), (char*)&buffer_size);

            // Read the FAISS index data into a temporary file
            std::vector<char> buffer(buffer_size);
            IOBINARY(p_in, ReadBinary, buffer_size, buffer.data());

            std::string tmp_file = "/tmp/lsq_quantizer_load_tmp.idx";
            std::ofstream file(tmp_file, std::ios::binary);
            if (!file.is_open()) {
                return ErrorCode::Fail;
            }

            file.write(buffer.data(), buffer_size);
            file.close();

            // Load FAISS index from file
            m_index_lsq.reset(dynamic_cast<faiss::IndexLocalSearchQuantizer*>(faiss::read_index(tmp_file.c_str())));

            // Clean up temporary file
            std::remove(tmp_file.c_str());

            if (!m_index_lsq) {
                return ErrorCode::Fail;
            }

            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Loading LSQ quantizer completed\n");
            return ErrorCode::Success;
        }

        ErrorCode LSQQuantizer::LoadQuantizer(std::uint8_t* raw_bytes) {
            // Implementation for loading from raw bytes
            return ErrorCode::Success;
        }

        QuantizerType LSQQuantizer::GetQuantizerType() const {
            return QuantizerType::LSQ;
        }

        VectorValueType LSQQuantizer::GetReconstructType() const {
            return VectorValueType::Float;
        }

        DimensionType LSQQuantizer::GetNumSubvectors() const {
            return _num_subvectors;
        }

        int LSQQuantizer::GetBase() const {
            return COMMON::Utils::GetBase<float>();
        }

        void LSQQuantizer::Train(SizeType n, const float* x) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Training LSQ quantizer with %d points\n", n);

            // Train the FAISS index
            m_index_lsq->train(n, x);
            
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "LSQ training completed\n");
        }
    }
}