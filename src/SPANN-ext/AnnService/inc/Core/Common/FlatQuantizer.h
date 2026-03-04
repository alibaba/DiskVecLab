// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_COMMON_FLATQUANTIZER_H_
#define _SPTAG_COMMON_FLATQUANTIZER_H_

#include "IQuantizer.h"
#include "inc/Core/Common.h"
#include "inc/Helper/DiskIO.h"
#include <memory>

namespace SPTAG
{
    namespace COMMON
    {
        class FlatQuantizer : public IQuantizer
        {
        public:
            FlatQuantizer();
            FlatQuantizer(DimensionType dim);
            virtual ~FlatQuantizer();

            // Distance calculations
            virtual float L2Distance(const std::uint8_t* pX, const std::uint8_t* pY) const override;
            virtual float CosineDistance(const std::uint8_t* pX, const std::uint8_t* pY) const override;

            // Quantization (no-op, just copy the data)
            virtual void QuantizeVector(const void* vec, std::uint8_t* vecout, bool ADC = true) const override;
            virtual SizeType QuantizeSize() const override;

            // Reconstruction (no-op, just copy the data)
            virtual void ReconstructVector(const std::uint8_t* qvec, void* vecout) const override;
            virtual SizeType ReconstructSize() const override;
            virtual DimensionType ReconstructDim() const override;

            // Serialization
            virtual std::uint64_t BufferSize() const override;
            virtual ErrorCode SaveQuantizer(std::shared_ptr<Helper::DiskIO> p_out) const override;
            virtual ErrorCode LoadQuantizer(std::shared_ptr<Helper::DiskIO> p_in) override;
            virtual ErrorCode LoadQuantizer(std::uint8_t* raw_bytes) override;

            // Properties
            virtual bool GetEnableADC() const override;
            virtual void SetEnableADC(bool enableADC) override;
            virtual QuantizerType GetQuantizerType() const override;
            virtual VectorValueType GetReconstructType() const override;
            virtual DimensionType GetNumSubvectors() const override;
            virtual int GetBase() const override;
            virtual float* GetL2DistanceTables() override;

            // Type identification
            virtual bool isPQ() override;
            virtual bool isRaBitQ() override;

        private:
            DimensionType m_dim;
            bool m_enableADC;
        };
    }
}

#endif // _SPTAG_COMMON_FLATQUANTIZER_H_