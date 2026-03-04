// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_COMMON_RABITQQUANTIZER_H_
#define _SPTAG_COMMON_RABITQQUANTIZER_H_

#include "CommonUtils.h"
#include "DistanceUtils.h"
#include "IQuantizer.h"
#include <memory>
#include <vector>
#include <mutex>
#include <unordered_map>
#include <string>
#include "inc/Core/Common/rabitqlib/utils/rotator.hpp"
#include "inc/Core/Common/rabitqlib/quantization/rabitq.hpp"

// Forward declarations for RaBitQ library types
namespace rabitqlib {
    template<typename T>
    class Rotator;
    
    namespace quant {
        struct RabitqConfig;
    }
}

namespace SPTAG {
    namespace COMMON {
        class RaBitQQuantizer : public IQuantizer {
        private:
            DimensionType m_dim;
            SizeType m_bits;
            DimensionType m_padded_dim;
            SizeType m_num_points;
            SizeType m_ex_bits;
            
            // SPTAG format storage (for compatibility)
            std::unique_ptr<char[]> m_bin_data;
            std::unique_ptr<char[]> m_ex_data;
            SizeType m_bin_data_stride;
            SizeType m_ex_data_stride;
            
            // DiskAnn format storage (for fast distance calculation)
            std::unique_ptr<char[]> m_diskann_bin_data;
            std::unique_ptr<char[]> m_diskann_ex_data;
            SizeType m_diskann_bin_data_stride;
            SizeType m_diskann_ex_data_stride;

            // SPTAG format storage (for memory comparison)
            mutable std::vector<std::vector<uint8_t>> m_sptag_quantized_data;

            std::unique_ptr<rabitqlib::Rotator<float>> m_rotator;
            std::vector<float> m_rotated_centroid;
            rabitqlib::quant::RabitqConfig m_query_config;
            
            // Quantization counter for sequential processing
            mutable SizeType m_quantization_counter;  // Internal counter for quantization

            mutable std::mutex m_mutex; // Mutex for thread safety during quantization and reconstruction

            // Debug: storage for original vectors for distance comparison
            mutable std::unordered_map<SizeType, std::vector<float>> m_original_vectors_map;
            mutable std::mutex m_debug_mutex;
            bool m_debug_enabled = false;

        public:
            RaBitQQuantizer(DimensionType dim = 0, SizeType bits = 8, int64_t num_points = 0);
            
            virtual float L2Distance(const std::uint8_t* pX, const std::uint8_t* pY) const override;
            
            virtual float CosineDistance(const std::uint8_t* pX, const std::uint8_t* pY) const override;
            
            virtual void QuantizeVector(const void* vec, std::uint8_t* vecout, bool ADC = true) const override;
            
            // Internal method for quantization with explicit row_id (used internally)
            void QuantizeVector(const void* vec, std::uint8_t* vecout, SizeType row_id, bool ADC = true) const;
            
            virtual SizeType QuantizeSize() const override;
            
            virtual void ReconstructVector(const std::uint8_t* qvec, void* vecout) const override;
            
            virtual SizeType ReconstructSize() const override;
            
            virtual DimensionType ReconstructDim() const override;
            
            virtual std::uint64_t BufferSize() const override;
            
            virtual ErrorCode SaveQuantizer(std::shared_ptr<Helper::DiskIO> p_out) const override;
            
            virtual ErrorCode LoadQuantizer(std::shared_ptr<Helper::DiskIO> p_in) override;
            
            virtual ErrorCode LoadQuantizer(std::uint8_t* raw_bytes) override;
            
            virtual bool GetEnableADC() const override { return m_EnableADC; }
            
            virtual void SetEnableADC(bool enableADC) override {
                m_EnableADC = enableADC;
            }

            
            virtual QuantizerType GetQuantizerType() const override { return QuantizerType::RaBitQ; }
            
            virtual VectorValueType GetReconstructType() const override { return VectorValueType::Float; }
            
            virtual DimensionType GetNumSubvectors() const override;
            
            virtual int GetBase() const override;
            
            virtual float* GetL2DistanceTables() override { return nullptr; }

            virtual bool isPQ() override { return false; }
            
            template<typename U>
            U* GetCodebooks() { return nullptr; }
            
            // Additional methods specific to RaBitQ
            void Train(SizeType n, const float* x);

            bool isRaBitQ() override { return true; }
            
        protected:
            void InitializeRotator();
            float ComputeDistanceToQuery(SizeType row_id, const float* query_rotated, float query_norm) const;
            
            // Dual format storage methods
            void InitializeDualFormatStorage(SizeType num_points);
            SizeType GetDiskAnnBinDataBytes() const;
            SizeType GetDiskAnnExDataBytes() const;
            void StoreDiskAnnFormat(SizeType point_id, const float* rotated_vec) const;
            void StoreSPTAGFormat(const float* rotated_vec, std::uint8_t* vecout, SizeType row_id = static_cast<SizeType>(-1)) const;
            float ComputeFastDistanceToQuery(SizeType row_id, const float* query_rotated, float query_norm) const;
            float ComputeDiskAnnDistance(const char* query_bin_data, const char* query_ex_data,
                                       const char* vector_bin_data, const char* vector_ex_data) const;
            
            // Debug functions for distance comparison
            void SaveDebugVectors() const;
            void LoadDebugVectors(const std::string& filename);
            
            bool m_EnableADC = false;
            bool is_load = false;
        };
    }
}

#endif // _SPTAG_COMMON_RABITQQUANTIZER_H_