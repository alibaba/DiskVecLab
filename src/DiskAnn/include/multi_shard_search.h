// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <vector>
#include <memory>
#include <algorithm>
#include <set>
#include <omp.h>
#include "search_state.h"
#include "pq_flash_index.h"
#include "windows_customizations.h"

namespace diskann
{

/**
 * MultiShardSearchContext manages adaptive multi-shard search with dynamic
 * L allocation based on each shard's contribution to the global top-k.
 * 
 * Algorithm Overview:
 * 1. Initialize all shards with L_init
 * 2. Iteratively:
 *    a. Merge current results from all shards
 *    b. Identify which shards contribute to global top-k
 *    c. Expand L by L_big for contributing shards, L_small for others
 *    d. Resume search on all shards
 * 3. Stop when total L across shards >= L_end
 * 
 * Termination Condition:
 * The search terminates when the sum of L values across all shards
 * reaches or exceeds L_end. This ensures sufficient exploration
 * while avoiding excessive IO on low-value shards.
 */
template <typename T, typename LabelT = uint32_t>
class MultiShardSearchContext
{
  public:
    MultiShardSearchContext(const MultiShardSearchParams &params)
        : _params(params), _global_L_sum(0), _iteration_count(0)
    {
    }

    /**
     * Add a shard to the search context.
     * @param shard_id Unique identifier for this shard
     * @param index Pointer to the PQFlashIndex for this shard
     */
    void add_shard(size_t shard_id, PQFlashIndex<T, LabelT> *index)
    {
        _shard_indices.push_back(index);
        _shard_states.emplace_back(shard_id);
    }

    /**
     * Execute adaptive multi-shard search.
     * 
     * @param query Query vector
     * @param k Number of results to return
     * @param beam_width Beam width for each shard search
     * @param res_ids Output array for result IDs
     * @param res_dists Output array for result distances
     * @param res_shard_ids Optional output array for shard IDs (can be nullptr)
     * @param debug Enable debug logging for this search
     */
    void search(const T *query, const uint64_t k, const uint64_t beam_width,
                uint64_t *res_ids, float *res_dists, size_t *res_shard_ids = nullptr,
                bool debug = false)
    {
        if (_shard_indices.empty())
        {
            throw ANNException("No shards added to MultiShardSearchContext", -1, __FUNCSIG__, __FILE__, __LINE__);
        }

        // Initialize L_end if not set
        if (_params.L_end == 0)
        {
            _params.set_default_L_end(_shard_indices.size());
        }

        _debug = debug;

        // Step 1: Initialize all shards with L_init
        initialize_all_shards(query, k, beam_width);

        if (_debug)
        {
            diskann::cout << "\n[MultiShard] Initialized " << _shard_indices.size() << " shards with L_init=" 
                          << _params.L_init << ", L_end=" << _params.L_end << std::endl;
        }

        // Step 2: Iterative refinement until global L sum >= L_end
        while (_global_L_sum < _params.L_end)
        {
            _iteration_count++;

            // 2a: Merge current results and identify contributing shards
            std::vector<GlobalSearchResult> merged_results;
            merge_current_results(k, merged_results);

            // 2b: Identify which shards contribute to global top-k
            std::set<size_t> contributing_shards;
            for (size_t i = 0; i < std::min(k, (uint64_t)merged_results.size()); i++)
            {
                contributing_shards.insert(merged_results[i].shard_id);
            }

            if (_debug)
            {
                diskann::cout << "[Iter " << _iteration_count << "] Contributing shards: {";
                bool first = true;
                for (auto shard_id : contributing_shards)
                {
                    if (!first) diskann::cout << ", ";
                    diskann::cout << shard_id;
                    first = false;
                }
                diskann::cout << "}" << std::endl;
            }

            // 2c: Update shard contribution status and calculate new L values
            update_shard_contributions(contributing_shards);

            // Check if we've reached L_end after this iteration
            if (_global_L_sum >= _params.L_end)
            {
                break;
            }

            // 2d: Resume search on all shards with new L values
            resume_all_shards(query, k, beam_width);
        }

        // Step 3: Final merge and return results
        get_final_results(k, res_ids, res_dists, res_shard_ids);
    }

    /**
     * Get the total number of IOs across all shards.
     */
    uint64_t get_total_ios() const
    {
        uint64_t total = 0;
        for (const auto &shard_state : _shard_states)
        {
            total += shard_state.state.get_num_ios();
        }
        return total;
    }

    /**
     * Get the current global L sum.
     */
    uint64_t get_global_L_sum() const
    {
        return _global_L_sum;
    }

    /**
     * Get the number of iterations performed.
     */
    uint32_t get_iteration_count() const
    {
        return _iteration_count;
    }

    /**
     * Get per-shard statistics.
     */
    void get_shard_stats(std::vector<uint64_t> &shard_L_values,
                         std::vector<uint32_t> &shard_ios,
                         std::vector<bool> &shard_contributed) const
    {
        shard_L_values.clear();
        shard_ios.clear();
        shard_contributed.clear();

        for (const auto &shard_state : _shard_states)
        {
            shard_L_values.push_back(shard_state.state.get_current_L());
            shard_ios.push_back(shard_state.state.get_num_ios());
            shard_contributed.push_back(shard_state.contributes_to_topk);
        }
    }

    /**
     * Reset the context for a new search.
     */
    void reset()
    {
        for (auto &shard_state : _shard_states)
        {
            shard_state.state.reset();
            shard_state.contributes_to_topk = false;
            shard_state.consecutive_non_contributing = 0;
        }
        _global_L_sum = 0;
        _iteration_count = 0;
    }

  private:
    /**
     * Initialize search on all shards with L_init.
     */
    void initialize_all_shards(const T *query, const uint64_t k, const uint64_t beam_width)
    {
        _global_L_sum = 0;

#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < _shard_indices.size(); i++)
        {
            _shard_indices[i]->init_search(query, k, _params.L_init, beam_width, _shard_states[i].state);
        }

        // Update global L sum
        for (auto &shard_state : _shard_states)
        {
            _global_L_sum += shard_state.state.get_current_L();
        }
    }

    /**
     * Merge current results from all shards.
     * Uses full_retset (full-precision distances) for accurate ordering.
     */
    void merge_current_results(const uint64_t k, std::vector<GlobalSearchResult> &merged_results)
    {
        merged_results.clear();

        // Collect results from each shard's full_retset (full-precision distances)
        for (size_t shard_idx = 0; shard_idx < _shard_states.size(); shard_idx++)
        {
            auto &shard_state = _shard_states[shard_idx];
            auto &full_retset = shard_state.state.full_retset();

            // Sort full_retset by distance (full-precision)
            std::sort(full_retset.begin(), full_retset.end());

            // Get top-k from this shard
            size_t count = std::min(k, (uint64_t)full_retset.size());
            for (size_t i = 0; i < count; i++)
            {
                merged_results.emplace_back(full_retset[i].id, full_retset[i].distance, shard_state.shard_id);
            }
        }

        // Sort merged results by distance
        std::sort(merged_results.begin(), merged_results.end());
    }

    /**
     * Update shard contribution status and calculate new L values.
     */
    void update_shard_contributions(const std::set<size_t> &contributing_shards)
    {
        for (auto &shard_state : _shard_states)
        {
            bool contributes = contributing_shards.find(shard_state.shard_id) != contributing_shards.end();
            shard_state.contributes_to_topk = contributes;

            if (contributes)
            {
                shard_state.consecutive_non_contributing = 0;
            }
            else
            {
                shard_state.consecutive_non_contributing++;
            }
        }
    }

    /**
     * Resume search on all shards with updated L values.
     */
    void resume_all_shards(const T *query, const uint64_t k, const uint64_t beam_width)
    {
        // Calculate new L values for each shard
        std::vector<uint64_t> new_L_values(_shard_states.size());
        uint64_t new_global_L_sum = 0;

        if (_debug)
        {
            diskann::cout << "[Iter " << _iteration_count << "] L adjustments:" << std::endl;
        }

        for (size_t i = 0; i < _shard_states.size(); i++)
        {
            auto &shard_state = _shard_states[i];
            uint64_t current_L = shard_state.state.get_current_L();
            uint64_t L_increment = shard_state.contributes_to_topk ? _params.L_big : _params.L_small;
            new_L_values[i] = current_L + L_increment;
            new_global_L_sum += new_L_values[i];

            if (_debug)
            {
                diskann::cout << "  Shard " << shard_state.shard_id 
                              << ": L " << current_L << " -> " << new_L_values[i]
                              << " (+" << L_increment << ", " 
                              << (shard_state.contributes_to_topk ? "contributing" : "non-contributing") << ")"
                              << std::endl;
            }
        }

        // Update global L sum
        _global_L_sum = new_global_L_sum;

        if (_debug)
        {
            diskann::cout << "  Global L sum: " << _global_L_sum << " / " << _params.L_end << std::endl;
        }

        // Resume search on all shards in parallel
#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < _shard_indices.size(); i++)
        {
            if (new_L_values[i] > _shard_states[i].state.get_current_L())
            {
                _shard_indices[i]->resume_search(query, k, new_L_values[i], beam_width, _shard_states[i].state);
            }
        }
    }

    /**
     * Get final merged results.
     * Uses get_search_results from each index to properly handle dummy points.
     */
    void get_final_results(const uint64_t k, uint64_t *res_ids, float *res_dists, size_t *res_shard_ids)
    {
        // Get results from each shard using proper get_search_results (handles dummy points)
        std::vector<GlobalSearchResult> merged_results;
        merged_results.reserve(_shard_states.size() * k);

        for (size_t shard_idx = 0; shard_idx < _shard_indices.size(); shard_idx++)
        {
            std::vector<uint64_t> shard_ids(k);
            std::vector<float> shard_dists(k);

            // Use get_search_results which handles dummy point remapping
            _shard_indices[shard_idx]->get_search_results(k, _shard_states[shard_idx].state,
                                                          shard_ids.data(), shard_dists.data(), false);

            auto &full_retset = _shard_states[shard_idx].state.full_retset();
            size_t count = std::min(k, (uint64_t)full_retset.size());
            for (size_t i = 0; i < count; i++)
            {
                merged_results.emplace_back(shard_ids[i], shard_dists[i], _shard_states[shard_idx].shard_id);
            }
        }

        // Sort merged results by distance
        std::sort(merged_results.begin(), merged_results.end());

        // Copy to output
        size_t result_count = std::min(k, (uint64_t)merged_results.size());
        for (size_t i = 0; i < result_count; i++)
        {
            res_ids[i] = merged_results[i].id;
            if (res_dists != nullptr)
            {
                res_dists[i] = merged_results[i].distance;
            }
            if (res_shard_ids != nullptr)
            {
                res_shard_ids[i] = merged_results[i].shard_id;
            }
        }
    }

    // Hyperparameters
    MultiShardSearchParams _params;

    // Shard indices and states
    std::vector<PQFlashIndex<T, LabelT> *> _shard_indices;
    std::vector<ShardSearchState<T>> _shard_states;

    // Global statistics
    uint64_t _global_L_sum;
    uint32_t _iteration_count;

    // Debug flag
    bool _debug = false;
};

} // namespace diskann
