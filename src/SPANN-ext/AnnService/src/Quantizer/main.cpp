// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Helper/VectorSetReader.h"
#include "inc/Core/VectorIndex.h"
#include "inc/Core/Common.h"
#include "inc/Helper/SimpleIniReader.h"
#include <inc/Core/Common/DistanceUtils.h>
#include "inc/Quantizer/Training.h"
#include "inc/Core/Common/FlatQuantizer.h"

#include <memory>

using namespace SPTAG;

void QuantizeAndSave(std::shared_ptr<SPTAG::Helper::VectorSetReader>& vectorReader, std::shared_ptr<QuantizerOptions>& options, std::shared_ptr<SPTAG::COMMON::IQuantizer>& quantizer)
{
    std::shared_ptr<SPTAG::VectorSet> set;
    
    for (int i = 0; (set = vectorReader->GetVectorSet(i, i + options->m_trainingSamples))->Count() > 0; i += options->m_trainingSamples)
    {
        if (i % (options->m_trainingSamples *10) == 0 || i % options->m_trainingSamples != 0)
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Saving vector batch starting at %d\n", i);
        }
        std::shared_ptr<VectorSet> quantized_vectors;
        if (options->m_normalized)
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Normalizing vectors.\n");
            set->Normalize(options->m_threadNum);
        }

        size_t quantized_vector_size = options->m_quantizedDim * set->Count();
        if (!(quantizer->isPQ()))
        {
          quantized_vector_size = quantizer->QuantizeSize() * set->Count();
        }

        //SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "alloc size %d\n", quantized_vector_size);
        ByteArray PQ_vector_array = ByteArray::Alloc(sizeof(std::uint8_t) * quantized_vector_size);
        quantized_vectors = std::make_shared<BasicVectorSet>(PQ_vector_array, VectorValueType::UInt8, quantizer->QuantizeSize(), set->Count());
        //SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "llllllllll\n", quantized_vector_size);
if (!(quantizer->isPQ()) && options->m_inputValueType == SPTAG::VectorValueType::UInt8) {
// Convert VectorSet to float array
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "convert uint8 to float\n");
    std::unique_ptr<float[]> float_data = std::make_unique<float[]>(set->Count() * set->Dimension());

    for (SizeType i = 0; i < set->Count(); i++) {
        auto vec = reinterpret_cast<const std::uint8_t*>(set->GetVector(i));
        // Convert to float
        for (DimensionType j = 0; j < set->Dimension(); j++) {
            float_data[i * set->Dimension() + j] = static_cast<float>(vec[j]);
        }
    }
#pragma omp parallel for
        for (int j = 0; j < set->Count(); j++)
        {
            quantizer->QuantizeVector((void*)(float_data.get() + j * set->Dimension()), (uint8_t*)quantized_vectors->GetVector(j));
        }
        //SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "mmmmmmmmmmmm\n", quantized_vector_size);
} else {
#pragma omp parallel for
     for (int j = 0; j < set->Count(); j++)
        {
            quantizer->QuantizeVector(set->GetVector(j), (uint8_t*)quantized_vectors->GetVector(j));
        }
}


        ErrorCode code;
        if ((code = quantized_vectors->AppendSave(options->m_outputFile)) != ErrorCode::Success)
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to save quantized vectors, ErrorCode: %s.\n", SPTAG::Helper::Convert::ConvertToString(code).c_str());
            exit(1);
        }
        //SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "nnnnnnnnnnnnnnnnnn\n", quantized_vector_size);
        if (!options->m_outputFullVecFile.empty())
        {
            if (ErrorCode::Success != set->AppendSave(options->m_outputFullVecFile))
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to save uncompressed vectors.\n");
                exit(1);
            }
        }
        if (!options->m_outputReconstructVecFile.empty())
        {
#pragma omp parallel for
            for (int j = 0; j < set->Count(); j++)
            {
                quantizer->ReconstructVector((uint8_t*)quantized_vectors->GetVector(j), set->GetVector(j));
            }
            if (ErrorCode::Success != set->AppendSave(options->m_outputReconstructVecFile))
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to save uncompressed vectors.\n");
                exit(1);
            }
        }
        //break;
    }
}

int main(int argc, char* argv[])
{
    std::shared_ptr<QuantizerOptions> options = std::make_shared<QuantizerOptions>(10000, true, 0.0f, SPTAG::QuantizerType::None, std::string(), -1, std::string(), std::string());

    if (!options->Parse(argc - 1, argv + 1))
    {
        exit(1);
    }
    auto vectorReader = Helper::VectorSetReader::CreateInstance(options);
    if (ErrorCode::Success != vectorReader->LoadFile(options->m_inputFiles))
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read input file.\n");
        exit(1);
    }
    switch (options->m_quantizerType)
    {
    case QuantizerType::None:
    {
        std::shared_ptr<SPTAG::VectorSet> set;
        for (int i = 0; (set = vectorReader->GetVectorSet(i, i + options->m_trainingSamples))->Count() > 0; i += options-> m_trainingSamples)
        {
            set->AppendSave(options->m_outputFile);
        }
        
        if (!options->m_outputMetadataFile.empty() && !options->m_outputMetadataIndexFile.empty())
        {
            auto metadataSet = vectorReader->GetMetadataSet();
            if (metadataSet)
            {
                metadataSet->SaveMetadata(options->m_outputMetadataFile, options->m_outputMetadataIndexFile);
            }
        }

        break;
    }
    case QuantizerType::PQQuantizer:
    {
        std::shared_ptr<COMMON::IQuantizer> quantizer;
        auto fp_load = SPTAG::f_createIO();
        if (fp_load == nullptr || !fp_load->Initialize(options->m_outputQuantizerFile.c_str(), std::ios::binary | std::ios::in))
        {
            auto set = vectorReader->GetVectorSet(0, options->m_trainingSamples);
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "set count. %d %d", set->Count(), sizeof(std::uint8_t) * options->m_quantizedDim * set->Count());
            ByteArray PQ_vector_array = ByteArray::Alloc(sizeof(std::uint8_t) * options->m_quantizedDim * set->Count());
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "there\n");
            std::shared_ptr<VectorSet> quantized_vectors = std::make_shared<BasicVectorSet>(PQ_vector_array, VectorValueType::UInt8, options->m_quantizedDim, set->Count());
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Quantizer Does not exist. Training a new one.\n");

            switch (options->m_inputValueType)
            {
#define DefineVectorValueType(Name, Type) \
                    case VectorValueType::Name: \
                        quantizer.reset(new COMMON::PQQuantizer<Type>(options->m_quantizedDim, 256, (DimensionType)(options->m_dimension/options->m_quantizedDim), false, TrainPQQuantizer<Type>(options, set, quantized_vectors))); \
                        break;

#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType
            }

            auto ptr = SPTAG::f_createIO();
            if (ptr != nullptr && ptr->Initialize(options->m_outputQuantizerFile.c_str(), std::ios::binary | std::ios::out))
            {
                if (ErrorCode::Success != quantizer->SaveQuantizer(ptr))
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write quantizer file.\n");
                    exit(1);
                }
            }
        }
        else
        {
            quantizer = SPTAG::COMMON::IQuantizer::LoadIQuantizer(fp_load);
            if (!quantizer)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to open existing quantizer file.\n");
                exit(1);
            }
            quantizer->SetEnableADC(false);
        }

        QuantizeAndSave(vectorReader, options, quantizer);

        auto metadataSet = vectorReader->GetMetadataSet();
        if (metadataSet)
        {
            metadataSet->SaveMetadata(options->m_outputMetadataFile, options->m_outputMetadataIndexFile);
        }
        
        break;
    }
    case QuantizerType::OPQQuantizer:
    {
        std::shared_ptr<COMMON::IQuantizer> quantizer;
        auto fp_load = SPTAG::f_createIO();
        if (fp_load == nullptr || !fp_load->Initialize(options->m_outputQuantizerFile.c_str(), std::ios::binary | std::ios::in))
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Quantizer Does not exist. Not supported for OPQ.\n");
            exit(1);
        }
        else
        {
            quantizer = SPTAG::COMMON::IQuantizer::LoadIQuantizer(fp_load);
            if (!quantizer)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to open existing quantizer file.\n");
                exit(1);
            }
            quantizer->SetEnableADC(false);
        }

        QuantizeAndSave(vectorReader, options, quantizer);


        auto metadataSet = vectorReader->GetMetadataSet();
        if (metadataSet)
        {
            metadataSet->SaveMetadata(options->m_outputMetadataFile, options->m_outputMetadataIndexFile);
        }

        break;
    }
    case QuantizerType::RaBitQ:
    {
        if (options->m_rabitqBits <= 0 || options->m_rabitqBits > 8)
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "rabitqBits must be > 0 and < 8.\n");
            exit(1);
        }
        std::shared_ptr<COMMON::IQuantizer> quantizer;
        auto fp_load = SPTAG::f_createIO();
        if (fp_load == nullptr || !fp_load->Initialize(options->m_outputQuantizerFile.c_str(), std::ios::binary | std::ios::in))
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Quantizer Does not exist. Training a new one.\n");
            auto set = vectorReader->GetVectorSet(0, options->m_trainingSamples);
            
            switch (options->m_inputValueType)
            {
#define DefineVectorValueType(Name, Type) \
                case VectorValueType::Name: \
                    quantizer = TrainRaBitQQuantizer<Type>(options, set, set->Count()); \
                    break;

#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType
            }

            auto ptr = SPTAG::f_createIO();
            if (ptr != nullptr && ptr->Initialize(options->m_outputQuantizerFile.c_str(), std::ios::binary | std::ios::out))
            {
               if (ErrorCode::Success != quantizer->SaveQuantizer(ptr))
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write quantizer file.\n");
                    exit(1);
                }
            }
        }
        else
        {
            quantizer = SPTAG::COMMON::IQuantizer::LoadIQuantizer(fp_load);
            if (!quantizer)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to open existing quantizer file.\n");
                exit(1);
            }
            quantizer->SetEnableADC(false);
        }

        QuantizeAndSave(vectorReader, options, quantizer);

        auto metadataSet = vectorReader->GetMetadataSet();
        if (metadataSet)
        {
            metadataSet->SaveMetadata(options->m_outputMetadataFile, options->m_outputMetadataIndexFile);
        }
        
        break;
    }
    case QuantizerType::LSQ:
    {
        std::shared_ptr<COMMON::IQuantizer> quantizer;
        auto fp_load = SPTAG::f_createIO();
        if (fp_load == nullptr || !fp_load->Initialize(options->m_outputQuantizerFile.c_str(), std::ios::binary | std::ios::in))
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Quantizer Does not exist. Training a new one.\n");
            auto set = vectorReader->GetVectorSet(0, options->m_trainingSamples);
            
            switch (options->m_inputValueType)
            {
#define DefineVectorValueType(Name, Type) \
                case VectorValueType::Name: \
                    quantizer = TrainLSQQuantizer<Type>(options, set); \
                    break;

#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType
            }

            auto ptr = SPTAG::f_createIO();
            if (ptr != nullptr && ptr->Initialize(options->m_outputQuantizerFile.c_str(), std::ios::binary | std::ios::out))
            {
                if (ErrorCode::Success != quantizer->SaveQuantizer(ptr))
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write quantizer file.\n");
                    exit(1);
                }
            }
        }
        else
        {
            quantizer = SPTAG::COMMON::IQuantizer::LoadIQuantizer(fp_load);
            if (!quantizer)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to open existing quantizer file.\n");
                exit(1);
            }
            quantizer->SetEnableADC(false);
        }

        QuantizeAndSave(vectorReader, options, quantizer);

        auto metadataSet = vectorReader->GetMetadataSet();
        if (metadataSet)
        {
            metadataSet->SaveMetadata(options->m_outputMetadataFile, options->m_outputMetadataIndexFile);
        }

        break;
    }
    case QuantizerType::Flat:
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Using Flat quantizer (no quantization).\n");
        std::shared_ptr<COMMON::IQuantizer> quantizer;

        // For Flat quantizer, we need to know the dimension
        auto set = vectorReader->GetVectorSet(0, 1); // Get first vector to determine dimension
        if (set->Count() == 0) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "No vectors found to determine dimension.\n");
            exit(1);
        }

        quantizer.reset(new COMMON::FlatQuantizer(set->Dimension()));

        // Save the quantizer file
        auto ptr = SPTAG::f_createIO();
        if (ptr != nullptr && ptr->Initialize(options->m_outputQuantizerFile.c_str(), std::ios::binary | std::ios::out))
        {
            if (ErrorCode::Success != quantizer->SaveQuantizer(ptr))
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write quantizer file.\n");
                exit(1);
            }
        }

        QuantizeAndSave(vectorReader, options, quantizer);

        auto metadataSet = vectorReader->GetMetadataSet();
        if (metadataSet)
        {
            metadataSet->SaveMetadata(options->m_outputMetadataFile, options->m_outputMetadataIndexFile);
        }

        break;
    }

    default:
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read quantizer type.\n");
        exit(1);
    }
    }
}