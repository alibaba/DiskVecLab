#include "inc/Core/Common/FlatQuantizer.h"
#include "inc/Core/Common/CommonUtils.h"
#include <cstring>
#include <cmath>

namespace SPTAG {
    namespace COMMON {

        FlatQuantizer::FlatQuantizer() : m_dim(0), m_enableADC(false) {
        }

        FlatQuantizer::FlatQuantizer(DimensionType dim) : m_dim(dim), m_enableADC(false) {
        }

        FlatQuantizer::~FlatQuantizer() {
        }

        float FlatQuantizer::L2Distance(const std::uint8_t* pX, const std::uint8_t* pY) const {
            // Cast back to float and compute L2 distance directly
            const float* x = reinterpret_cast<const float*>(pX);
            const float* y = reinterpret_cast<const float*>(pY);

            float dist = 0.0f;
            for (DimensionType i = 0; i < m_dim; i++) {
                float diff = x[i] - y[i];
                dist += diff * diff;
            }
            return dist;
        }

        float FlatQuantizer::CosineDistance(const std::uint8_t* pX, const std::uint8_t* pY) const {
            return 0;  // Convert similarity to distance
        }

        void FlatQuantizer::QuantizeVector(const void* vec, std::uint8_t* vecout, bool ADC) const {
            // No quantization - just copy the original float vector
            const float* input_vec = static_cast<const float*>(vec);
            float* output_vec = reinterpret_cast<float*>(vecout);

            std::memcpy(output_vec, input_vec, m_dim * sizeof(float));
        }

        SizeType FlatQuantizer::QuantizeSize() const {
            // Size is just the original vector size in bytes
            return m_dim * sizeof(float);
        }

        void FlatQuantizer::ReconstructVector(const std::uint8_t* qvec, void* vecout) const {
            // No reconstruction needed - just copy the data
            const float* quantized_vec = reinterpret_cast<const float*>(qvec);
            float* output_vec = static_cast<float*>(vecout);

            std::memcpy(output_vec, quantized_vec, m_dim * sizeof(float));
        }

        SizeType FlatQuantizer::ReconstructSize() const {
            return m_dim * sizeof(float);
        }

        DimensionType FlatQuantizer::ReconstructDim() const {
            return m_dim;
        }

        std::uint64_t FlatQuantizer::BufferSize() const {
            return sizeof(QuantizerType) + sizeof(VectorValueType) + sizeof(DimensionType) + sizeof(bool);
        }

        ErrorCode FlatQuantizer::SaveQuantizer(std::shared_ptr<Helper::DiskIO> p_out) const {
            QuantizerType qtype = QuantizerType::Flat;
            VectorValueType rtype = VectorValueType::Float;

            IOBINARY(p_out, WriteBinary, sizeof(QuantizerType), (char*)&qtype);
            IOBINARY(p_out, WriteBinary, sizeof(VectorValueType), (char*)&rtype);
            IOBINARY(p_out, WriteBinary, sizeof(DimensionType), (char*)&m_dim);
            IOBINARY(p_out, WriteBinary, sizeof(bool), (char*)&m_enableADC);

            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Saving Flat quantizer: dim:%d\n", m_dim);
            return ErrorCode::Success;
        }

        ErrorCode FlatQuantizer::LoadQuantizer(std::shared_ptr<Helper::DiskIO> p_in) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Loading Flat Quantizer.\n");

            IOBINARY(p_in, ReadBinary, sizeof(DimensionType), (char*)&m_dim);
            IOBINARY(p_in, ReadBinary, sizeof(bool), (char*)&m_enableADC);

            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Loaded Flat quantizer: dim:%d\n", m_dim);
            return ErrorCode::Success;
        }

        ErrorCode FlatQuantizer::LoadQuantizer(std::uint8_t* raw_bytes) {
            // Implementation for loading from raw bytes
            size_t offset = 0;

            // Skip quantizer type and vector value type (already read by caller)
            offset += sizeof(QuantizerType) + sizeof(VectorValueType);

            std::memcpy(&m_dim, raw_bytes + offset, sizeof(DimensionType));
            offset += sizeof(DimensionType);

            std::memcpy(&m_enableADC, raw_bytes + offset, sizeof(bool));

            return ErrorCode::Success;
        }

        bool FlatQuantizer::GetEnableADC() const {
            return m_enableADC;
        }

        void FlatQuantizer::SetEnableADC(bool enableADC) {
            m_enableADC = enableADC;
        }

        QuantizerType FlatQuantizer::GetQuantizerType() const {
            return QuantizerType::Flat;
        }

        VectorValueType FlatQuantizer::GetReconstructType() const {
            return VectorValueType::Float;
        }

        DimensionType FlatQuantizer::GetNumSubvectors() const {
            return m_dim;
        }

        int FlatQuantizer::GetBase() const {
            return COMMON::Utils::GetBase<float>();
        }

        float* FlatQuantizer::GetL2DistanceTables() {
            // No distance tables needed for flat quantizer
            return nullptr;
        }

        bool FlatQuantizer::isPQ() {
            return false;
        }

        bool FlatQuantizer::isRaBitQ() {
            return false;
        }
    }
}