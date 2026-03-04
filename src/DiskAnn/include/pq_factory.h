// pq_factory.h
#pragma once
#include "pq_table_base.h"
#include "fixed_chunk_pq_table_adapter.h"
#include "rabitq_quantizer.h"
#include "llocal_search_quantizer.h"
#include "full_precision_quantizer.h"

class PQFactory {
  public:
    static std::unique_ptr<PQTableBase> create_pq_table(PQType type, uint64_t dim = 128, uint64_t num_pq_chunks = 32,
                                                         size_t rabitq_clusters = RABITQ_DEFAULT_NUM_CLUSTERS) {
        switch(type) {
        case PQType::PQ:
        case PQType::OPQ:
            return std::make_unique<FixedChunkPQTableAdapter>();
        case PQType::RABITQ:
            diskann::cout << "Creating RabitQ with dim=" << dim << " nbit=" << num_pq_chunks/16
                          << " num_clusters=" << rabitq_clusters << std::endl;
            return std::make_unique<RabitqQuantizer>(dim, num_pq_chunks/16, rabitq_clusters);
        case PQType::LSQ:
            diskann::cout << "Creating LSQ with dim=" << dim << " " << num_pq_chunks * 2 << "x" << "4-bit chunks." << std::endl;
            return std::make_unique<LLocalSearchQuantizer>(dim, num_pq_chunks * 2, 4);
        case PQType::FULL_PRECISION:
            return std::make_unique<FullPrecisionQuantizer>(dim);
        default:
            return nullptr;
        }
    }
};
