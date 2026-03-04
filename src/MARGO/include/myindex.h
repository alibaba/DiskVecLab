#pragma once

#include "index.h"

namespace diskann {

    template<typename T, typename TagT = uint32_t>
    class myIndex : public Index<T, TagT> {
        public:
        DISKANN_DLLEXPORT myIndex(Metric m, const size_t dim,
                            const size_t max_points = 1,
                            const bool   dynamic_index = false,
                            const bool   enable_tags = false,
                            const bool   support_eager_delete = false,
                            const bool   concurrent_consolidate = false);
        
        DISKANN_DLLEXPORT void mybuild(
        const char *filename, const size_t num_points_to_load,
        Parameters &             parameters,
        const std::vector<TagT> &tags = std::vector<TagT>());

        void my_build_with_data_populated(Parameters &parameters, 
                                          const std::vector<TagT> &tags);

        void mylink(Parameters &parameters);

        void my_prune_neighbors(const unsigned location, std::vector<Neighbor> &pool,
                                std::vector<unsigned> &pruned_list,
                                std::vector<float> &weight_list);

        void my_prune_neighbors(const unsigned location, std::vector<Neighbor> &pool,
                                const _u32 range, const _u32 max_candidate_size,
                                const float alpha, std::vector<unsigned> &pruned_list,
                                std::vector<float> &weight_list);
        
        void my_occlude_list(std::vector<Neighbor> &pool, const float alpha,
                             const unsigned degree, const unsigned maxc,
                             std::vector<std::pair<Neighbor, float>> &result, 
                             const unsigned location);
        
        void my_batch_inter_insert(unsigned                     n,
                                   const std::vector<unsigned> &pruned_list,
                                   const _u32                   range,
                                   std::vector<unsigned> &      need_to_sync);

        void my_batch_inter_insert(unsigned                     n,
                                   const std::vector<unsigned> &pruned_list,
                                   std::vector<unsigned> &      need_to_sync);
        
        DISKANN_DLLEXPORT void mysave(const char *filename,
                                       bool        compact_before_save = false);
        
        DISKANN_DLLEXPORT _u64 my_save_weights(std::string filename);

        std::vector<std::vector<float>> weights_;
        std::vector<float> wws_;
    };

}