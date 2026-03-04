// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_COMMON_LSQQUANTIZER_H_
#define _SPTAG_COMMON_LSQQUANTIZER_H_

#include "CommonUtils.h"
#include "DistanceUtils.h"
#include "IQuantizer.h"
#include <memory>
#include <mutex>
#include <faiss/IndexAdditiveQuantizer.h>

namespace SPTAG {
    namespace COMMON {
        class LSQQuantizer : public IQuantizer {
        private:
            std::unique_ptr<faiss::IndexLocalSearchQuantizer> m_index_lsq;
            mutable std::vector<std::unique_ptr<faiss::FlatCodesDistanceComputer>> m_distance_computers;
            mutable std::mutex m_mutex; // Mutex for thread safety during quantization and reconstruction
            int _num_subvectors;
        public:
            LSQQuantizer(DimensionType dim, SizeType num_subvectors, SizeType nbits);
            
            // Default constructor used in loading quantizer
            LSQQuantizer() = default;
            
            virtual float L2Distance(const std::uint8_t* pX, const std::uint8_t* pY) const override;
            
            virtual float CosineDistance(const std::uint8_t* pX, const std::uint8_t* pY) const override;
            
            virtual void QuantizeVector(const void* vec, std::uint8_t* vecout, bool ADC = true) const override;
            
            virtual SizeType QuantizeSize() const override;
            
            virtual void ReconstructVector(const std::uint8_t* qvec, void* vecout) const override;
            
            virtual SizeType ReconstructSize() const override;
            
            virtual DimensionType ReconstructDim() const override;
            
            virtual std::uint64_t BufferSize() const override;
            
            virtual ErrorCode SaveQuantizer(std::shared_ptr<Helper::DiskIO> p_out) const override;
            
            virtual ErrorCode LoadQuantizer(std::shared_ptr<Helper::DiskIO> p_in) override;
            
            virtual ErrorCode LoadQuantizer(std::uint8_t* raw_bytes) override;
            
            virtual VectorValueType GetReconstructType() const override;

            virtual QuantizerType GetQuantizerType() const override;
            
            virtual DimensionType GetNumSubvectors() const override;
            
            virtual int GetBase() const override;

            virtual bool GetEnableADC() const override {
                return false;
            }

            virtual void SetEnableADC(bool enableADC) override {
              // Not supported
            }
            
            virtual float* GetL2DistanceTables() override { return nullptr; }

            virtual bool isPQ() override { return false; }
            
            template<typename U>
            U* GetCodebooks() { return nullptr; }
            
            // Additional methods specific to LSQ
            void Train(SizeType n, const float* x);
        };
    }
}

#endif // _SPTAG_COMMON_LSQQUANTIZER_H_