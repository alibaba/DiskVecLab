// pq_factory.h
#pragma once
#include "pq_table_base.h"
#include "fixed_chunk_pq_table_adapter.h"
#include "rabitq_quantizer.h"
#include "llocal_search_quantizer.h"
class PQFactory {
 public:
  static std::unique_ptr<PQTableBase> create_pq_table(PQType type) {
    switch(type) {
      case PQType::PQ:
      case PQType::OPQ:
        return std::make_unique<FixedChunkPQTableAdapter>();
      case PQType::RABITQ:
        return std::make_unique<RabitqQuantizer>(128);
      case PQType::LSQ:
        return std::make_unique<LLocalSearchQuantizer>(128, 64, 4);
      default:
        return nullptr;
    }
  }
};