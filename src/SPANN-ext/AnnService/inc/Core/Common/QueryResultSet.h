// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_COMMON_QUERYRESULTSET_H_
#define _SPTAG_COMMON_QUERYRESULTSET_H_

#include "inc/Core/SearchQuery.h"
#include "DistanceUtils.h"
#include <algorithm>
#include <cstring>
#include <typeinfo>
#include <type_traits>
#include <vector>
#include "IQuantizer.h"

namespace SPTAG
{
namespace COMMON
{

inline bool operator < (const BasicResult& lhs, const BasicResult& rhs)
{
    return ((lhs.Dist < rhs.Dist) || ((lhs.Dist == rhs.Dist) && (lhs.VID < rhs.VID)));
}


inline bool Compare(const BasicResult& lhs, const BasicResult& rhs)
{
    return ((lhs.Dist < rhs.Dist) || ((lhs.Dist == rhs.Dist) && (lhs.VID < rhs.VID)));
}


// Space to save temporary answer, similar with TopKCache
template<typename T>
class QueryResultSet : public QueryResult
{
public:
    QueryResultSet(const T* _target, int _K) : QueryResult(_target, _K, false)
    {
    }

    QueryResultSet(const QueryResultSet& other) : QueryResult(other)
    {
    }

    ~QueryResultSet()
    {
    }

    inline void SetTarget(const T* p_target, const std::shared_ptr<IQuantizer>& quantizer)
    {
        //SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "set target!\n");
        if (quantizer == nullptr) {
            QueryResult::SetTarget((const void*)p_target);
            // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "setTarget: quantizer null");
        }
        else
        {
            // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "setTarget: quantizer not null");

            // Allocate quantized target buffer
            if (m_target == m_quantizedTarget || (m_quantizedSize != quantizer->QuantizeSize()))
            {
                if (m_target != m_quantizedTarget) ALIGN_FREE(m_quantizedTarget);
                m_quantizedTarget = ALIGN_ALLOC(quantizer->QuantizeSize());
                m_quantizedSize = quantizer->QuantizeSize();
            }
            m_target = p_target;

            // For RaBitQ, convert to float if T is not float
            if (!(quantizer->isPQ()) && !std::is_same<T, float>::value) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "RaBitQ: Converting input from type %s to float\n", 
                    typeid(T).name());

                // Allocate temporary float buffer, the dim shoudld be passed from outside
                // for now, wo just hard code
                auto dim = 128;
                std::vector<float> float_vector(dim);
                
                // Convert T* to float*
                for (DimensionType i = 0; i < dim; i++) {
                    float_vector[i] = static_cast<float>(p_target[i]);
                }
                
                // Quantize using float vector
                quantizer->QuantizeVector((void*)float_vector.data(), (uint8_t*)m_quantizedTarget);
                
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "RaBitQ: Converted %d elements to float for quantization\n", dim);
            }
            else {
                // Normal path: directly quantize
                quantizer->QuantizeVector((void*)p_target, (uint8_t*)m_quantizedTarget);
            }
        }
    }


    // Set target with pre-quantized data (no re-quantization needed)
    inline void SetTargetWithQuantized(const T* p_target, const uint8_t* p_quantized_target, const std::shared_ptr<IQuantizer>& quantizer)
    {
        auto quantized_size = quantizer == nullptr ? 0 : quantizer->QuantizeSize();

        if (quantized_size == 0) {
            QueryResult::SetTarget((const void*)p_target);
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SetTargetWithQuantized: no quantization needed");
        }
        else
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SetTargetWithQuantized: using pre-quantized target\n");
            // Allocate memory for quantized target if needed
            if (m_target == m_quantizedTarget || (m_quantizedSize != quantized_size))
            {
                if (m_target != m_quantizedTarget) ALIGN_FREE(m_quantizedTarget);
                m_quantizedTarget = ALIGN_ALLOC(quantized_size);
                m_quantizedSize = quantized_size;
            }

            // Set the original target
            m_target = p_target;

            // Copy the pre-quantized data
            std::memcpy(m_quantizedTarget, p_quantized_target, quantized_size);
        }
    }

    inline const T* GetTarget() const
    {
        return reinterpret_cast<const T*>(m_target);
    }

    T* GetQuantizedTarget()
    {
        return reinterpret_cast<T*>(m_quantizedTarget);
    }

    inline float worstDist() const
    {
        return m_results[0].Dist;
    }

    bool AddPoint(const SizeType index, float dist)
    {
        if (dist < m_results[0].Dist || (dist == m_results[0].Dist && index < m_results[0].VID))
        {
            m_results[0].VID = index;
            m_results[0].Dist = dist;
            Heapify(m_resultNum);
            return true;
        }
        return false;
    }

    inline void SortResult()
    {
        for (int i = m_resultNum - 1; i >= 0; i--)
        {
            std::swap(m_results[0], m_results[i]);
            Heapify(i);
        }
    }

    void Reverse()
    {
        std::reverse(m_results.Data(), m_results.Data() + m_resultNum);
    }

private:
    void Heapify(int count)
    {
        int parent = 0, next = 1, maxidx = count - 1;
        while (next < maxidx)
        {
            if (m_results[next] < m_results[next + 1]) next++;
            if (m_results[parent] < m_results[next])
            {
                std::swap(m_results[next], m_results[parent]);
                parent = next;
                next = (parent << 1) + 1;
            }
            else break;
        }
        if (next == maxidx && m_results[parent] < m_results[next]) std::swap(m_results[parent], m_results[next]);
    }
};
}
}

#endif // _SPTAG_COMMON_QUERYRESULTSET_H_
