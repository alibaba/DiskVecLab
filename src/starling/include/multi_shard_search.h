// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <set>
#include <string>
#include <unordered_map>
#include <omp.h>
#include "search_state.h"
#include "pq_flash_index.h"
#include "windows_customizations.h"

namespace diskann {

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
template <typename T>
class MultiShardSearchContext {
  public:
    MultiShardSearchContext(const MultiShardSearchParams &params)
        : _params(params), _global_L_sum(0), _iteration_count(0) {
    }

    /**
     * Add a shard to the search context.
     * @param shard_id Unique identifier for this shard
     * @param index Pointer to the PQFlashIndex for this shard
     */
    void add_shard(size_t shard_id, PQFlashIndex<T> *index) {
        _shard_indices.push_back(index);
        _shard_states.emplace_back(shard_id);
    }

    /**
     * Execute adaptive multi-shard search using page_search.
     * 
     * @param query Query vector
     * @param k Number of results to return
     * @param mem_L Memory graph L parameter
     * @param beam_width Beam width for each shard search
     * @param io_limit IO limit per search
     * @param use_ratio Page search use ratio
     * @param res_ids Output array for result IDs (global IDs via id_maps)
     * @param res_dists Output array for result distances
     * @param res_shard_ids Optional output array for shard IDs
     * @param id_maps ID mapping vectors for each shard (local->global)
     * @param debug Enable debug logging for this search
     */
    void search(const T *query, const uint64_t k, const uint32_t mem_L,
                const uint64_t beam_width, const uint32_t io_limit,
                const float use_ratio,
                uint64_t *res_ids, float *res_dists, 
                size_t *res_shard_ids,
                const std::vector<std::vector<uint32_t>> &id_maps,
                bool debug = false) {
        if (_shard_indices.empty()) {
            throw std::runtime_error("No shards added to MultiShardSearchContext");
        }

        // Initialize L_end if not set
        if (_params.L_end == 0) {
            _params.set_default_L_end(_shard_indices.size());
        }

        // Auto-compute L_capacity if not set
        _params.set_default_L_capacity(_shard_indices.size());

        _debug = debug;

        // Step 1: Initialize all shards with L_init search
        initialize_all_shards(query, k, mem_L, beam_width, io_limit, use_ratio);

        if (_debug) {
            std::cout << "\n[MultiShard] Initialized " << _shard_indices.size() 
                      << " shards with L_capacity=" << _params.L_capacity
                      << ", L_init=" << _params.L_init 
                      << ", L_end=" << _params.L_end << std::endl;
        }

        // Step 2: Iterative refinement until global L sum >= L_end
        // Scheduler can be selected via params.adaptive_shard_scheduler.
        if (_params.adaptive_shard_scheduler == "radius_topL") {
            while (_global_L_sum < _params.L_end) {
                _iteration_count++;
                uint64_t prev_global_L_sum = _global_L_sum;

                // Pick the shard with smallest per-shard top-L radius, where
                // L is the shard's current L_stop. Radius is computed from the
                // candidate retset within [0, min(cur_list_size, L_stop)).
                size_t best_idx = std::numeric_limits<size_t>::max();
                float best_radius = std::numeric_limits<float>::infinity();
                for (size_t i = 0; i < _shard_states.size(); i++) {
                    auto &st = _shard_states[i].state;
                    const uint64_t l_stop = st.get_current_L();
                    if (l_stop >= _params.L_capacity) {
                        continue;
                    }
                    const uint64_t bound = std::min<uint64_t>(st.cur_list_size(), l_stop);
                    if (bound == 0) {
                        continue;
                    }

                    float radius = 0.0f;
                    auto &retset = st.retset();
                    for (uint64_t j = 0; j < bound; j++) {
                        radius = std::max(radius, retset[j].distance);
                    }

                    if (radius < best_radius) {
                        best_radius = radius;
                        best_idx = i;
                    }
                }

                if (best_idx == std::numeric_limits<size_t>::max()) {
                    if (_debug) {
                        std::cout << "[Iter " << _iteration_count
                                  << "] No shard has a valid radius. Stopping." << std::endl;
                    }
                    break;
                }

                auto &best_state = _shard_states[best_idx];
                const uint64_t current_l_stop = best_state.state.get_current_L();
                const uint64_t new_l_stop = std::min(current_l_stop + _params.L_big, _params.L_capacity);

                if (_debug) {
                    std::cout << "[Iter " << _iteration_count << "] radius_topL choose shard "
                              << best_state.shard_id << " (idx=" << best_idx << ")"
                              << ": radius=" << best_radius
                              << ", L_stop " << current_l_stop << " -> " << new_l_stop
                              << " (capacity=" << _params.L_capacity << ")" << std::endl;
                }

                // Update global L sum and resume only the chosen shard.
                if (new_l_stop > current_l_stop) {
                    _global_L_sum += (new_l_stop - current_l_stop);
                    _shard_indices[best_idx]->resume_page_search(query, k, mem_L, new_l_stop,
                                                                 beam_width, io_limit, use_ratio,
                                                                 best_state.state);
                }

                // Progress check
                if (_global_L_sum <= prev_global_L_sum) {
                    if (_debug) {
                        std::cout << "[Iter " << _iteration_count
                                  << "] No progress (L_sum=" << _global_L_sum
                                  << "), chosen shard at capacity. Stopping." << std::endl;
                    }
                    break;
                }

                if (_global_L_sum >= _params.L_end) {
                    break;
                }
            }
        } else if (_params.adaptive_shard_scheduler == "global_topL_pages") {
            // Allocate exactly `budget` (= min(L_big, remaining)) L_stop increments per iteration,
            // by globally selecting the closest (PQ distance) unvisited pages across all shards.
            // Each selected page contributes +1 to its shard's L_stop.

            struct PageCandidate {
                float dist;
                size_t shard_idx;
                unsigned page_id;
                unsigned node_id;
            };
            struct PageCandidateGreater {
                bool operator()(const PageCandidate &a, const PageCandidate &b) const {
                    // priority_queue is max-heap by default; invert to get min-heap.
                    if (a.dist != b.dist) return a.dist > b.dist;
                    if (a.shard_idx != b.shard_idx) return a.shard_idx > b.shard_idx;
                    if (a.page_id != b.page_id) return a.page_id > b.page_id;
                    return a.node_id > b.node_id;
                }
            };

            while (_global_L_sum < _params.L_end) {
                _iteration_count++;

                const uint64_t remaining = _params.L_end - _global_L_sum;
                const uint64_t budget = std::min<uint64_t>(_params.L_big, remaining);
                if (budget == 0) {
                    break;
                }

                const uint64_t scan_cap = std::max<uint64_t>(64, 4 * budget);

                std::priority_queue<PageCandidate, std::vector<PageCandidate>, PageCandidateGreater> heap;
                std::vector<PageCandidate> all_candidates;
                if (_debug) {
                    all_candidates.reserve((size_t)(_shard_states.size() * scan_cap));
                }

                // Collect per-shard candidate pages from retset[pos>=cursor], with per-shard page de-dup.
                for (size_t shard_idx = 0; shard_idx < _shard_states.size(); shard_idx++) {
                    auto &st = _shard_states[shard_idx].state;
                    const uint64_t cursor = st.get_search_cursor();
                    const uint64_t l_capacity = st.get_l_capacity();
                    const uint64_t bound = std::min<uint64_t>(st.cur_list_size(), l_capacity);
                    if (cursor >= bound) {
                        continue;
                    }

                    auto &retset = st.retset();
                    auto &page_visited = st.page_visited();

                    // page_id -> (best_dist, best_node)
                    std::unordered_map<unsigned, std::pair<float, unsigned>> page_best;
                    page_best.reserve((size_t) scan_cap);

                    uint64_t scanned = 0;
                    for (uint64_t pos = cursor; pos < bound && scanned < scan_cap; pos++) {
                        scanned++;

                        if (!retset[pos].flag) {
                            continue;
                        }

                        const unsigned node_id = retset[pos].id;
                        const unsigned pid = _shard_indices[shard_idx]->get_page_id(node_id);
                        if (page_visited.find(pid) != page_visited.end()) {
                            continue;
                        }

                        auto it = page_best.find(pid);
                        if (it == page_best.end() || retset[pos].distance < it->second.first) {
                            page_best[pid] = std::make_pair(retset[pos].distance, node_id);
                        }

                        // Early stop: scanning further usually yields diminishing returns.
                        if (page_best.size() >= scan_cap) {
                            break;
                        }
                    }

                    for (const auto &kv : page_best) {
                        PageCandidate cand{kv.second.first, shard_idx, kv.first, kv.second.second};
                        heap.push(cand);
                        if (_debug) {
                            all_candidates.push_back(cand);
                        }
                    }
                }

                if (_debug) {
                    std::sort(all_candidates.begin(), all_candidates.end(),
                              [](const PageCandidate &a, const PageCandidate &b) {
                                  if (a.dist != b.dist) return a.dist < b.dist;
                                  if (a.shard_idx != b.shard_idx) return a.shard_idx < b.shard_idx;
                                  if (a.page_id != b.page_id) return a.page_id < b.page_id;
                                  return a.node_id < b.node_id;
                              });

                    std::cout << "[Iter " << _iteration_count << "] candidates: total="
                              << all_candidates.size() << std::endl;
                    for (size_t si = 0; si < _shard_states.size(); si++) {
                        size_t cnt = 0;
                        for (const auto &c : all_candidates) {
                            if (c.shard_idx == si) cnt++;
                        }
                        auto &st = _shard_states[si].state;
                        std::cout << "    shard_idx=" << si
                                  << " shard_id=" << _shard_states[si].shard_id
                                  << " cand_pages=" << cnt
                                  << " cursor=" << st.get_search_cursor()
                                  << " cur_list_size=" << st.cur_list_size()
                                  << " L_stop=" << st.get_current_L()
                                  << " L_capacity=" << st.get_l_capacity()
                                  << std::endl;
                    }

                    std::cout << "  candidates(sorted by pq_dist):" << std::endl;
                    for (size_t ci = 0; ci < all_candidates.size(); ci++) {
                        const auto &c = all_candidates[ci];
                        std::cout << "    cand[" << ci << "]: shard_idx=" << c.shard_idx
                                  << " shard_id=" << _shard_states[c.shard_idx].shard_id
                                  << " page_id=" << c.page_id
                                  << " node_id=" << c.node_id
                                  << " pq_dist=" << c.dist
                                  << std::endl;
                    }
                }

                // Allocate +1 L_stop per selected page (total picked <= budget).
                std::vector<uint64_t> alloc(_shard_states.size(), 0);
                uint64_t picked = 0;
                std::vector<PageCandidate> picked_pages;
                if (_debug) {
                    picked_pages.reserve((size_t) budget);
                }
                while (picked < budget && !heap.empty()) {
                    const PageCandidate cand = heap.top();
                    heap.pop();

                    auto &st = _shard_states[cand.shard_idx].state;
                    // Re-check: page may have become in-flight earlier in this iteration.
                    if (st.page_visited().find(cand.page_id) != st.page_visited().end()) {
                        continue;
                    }

                    alloc[cand.shard_idx] += 1;
                    picked++;
                    if (_debug) {
                        picked_pages.push_back(cand);
                    }
                }

                // Update per-shard L_stop and global sum.
                std::vector<uint64_t> new_l_stop_values(_shard_states.size());
                uint64_t new_global_L_sum = 0;
                for (size_t i = 0; i < _shard_states.size(); i++) {
                    auto &st = _shard_states[i].state;
                    const uint64_t current_l_stop = st.get_current_L();
                    const uint64_t l_capacity = st.get_l_capacity();
                    new_l_stop_values[i] = std::min<uint64_t>(current_l_stop + alloc[i], l_capacity);
                    new_global_L_sum += new_l_stop_values[i];
                }
                _global_L_sum = new_global_L_sum;

                if (_debug) {
                    std::cout << "[Iter " << _iteration_count << "] global_topL_pages: "
                              << "budget=" << budget << ", picked=" << picked
                              << ", global_L_sum=" << _global_L_sum << " / " << _params.L_end
                              << std::endl;

                    const size_t show = std::min<size_t>(picked_pages.size(), 16);
                    for (size_t j = 0; j < show; j++) {
                        const auto &c = picked_pages[j];
                        std::cout << "  pick[" << j << "]: shard_idx=" << c.shard_idx
                                  << " shard_id=" << _shard_states[c.shard_idx].shard_id
                                  << " page_id=" << c.page_id << " node_id=" << c.node_id
                                  << " pq_dist=" << c.dist << std::endl;
                    }

                    std::cout << "  per-shard alloc / L_stop:" << std::endl;
                    for (size_t i = 0; i < _shard_states.size(); i++) {
                        auto &st = _shard_states[i].state;
                        const uint64_t cur = st.get_current_L();
                        std::cout << "    shard_idx=" << i
                                  << " shard_id=" << _shard_states[i].shard_id
                                  << " alloc=" << alloc[i]
                                  << " L_stop " << cur << " -> " << new_l_stop_values[i]
                                  << std::endl;
                    }
                }

                // Resume shards that got new budget, and always resume shards with deferred pages.
                bool any_resumed = false;
                for (size_t i = 0; i < _shard_indices.size(); i++) {
                    auto &st = _shard_states[i].state;
                    const uint64_t current_l_stop = st.get_current_L();
                    const bool has_deferred = !st.last_io_ids().empty();

                    if (new_l_stop_values[i] > current_l_stop || has_deferred) {
                        _shard_indices[i]->resume_page_search(query, k, mem_L, new_l_stop_values[i],
                                                              beam_width, io_limit, use_ratio,
                                                              st);
                        any_resumed = true;
                    } else {
                        // Keep the state consistent with computed L_stop.
                        st.set_l_stop(new_l_stop_values[i]);
                    }
                }

                if (!any_resumed) {
                    // No increments and no deferred work: nothing else can be done.
                    break;
                }

                if (_global_L_sum >= _params.L_end) {
                    break;
                }
            }
        } else {
            while (_global_L_sum < _params.L_end) {
                _iteration_count++;
                uint64_t prev_global_L_sum = _global_L_sum;

                // 2a: Merge current results and identify which shards contribute to the
                // global top-k (based on full-precision distances).
                std::vector<GlobalSearchResult> merged_results;
                merge_current_results(k, id_maps, merged_results);

                std::set<size_t> contributing_shards;
                const size_t consider = std::min((uint64_t) merged_results.size(), k);
                for (size_t i = 0; i < consider; i++) {
                    contributing_shards.insert(merged_results[i].shard_id);
                }

                // 2b: Update contribution flags, then resume all shards with
                // L_big / L_small increments (clamped to L_capacity).
                update_shard_contributions(contributing_shards);
                resume_all_shards(query, k, mem_L, beam_width, io_limit, use_ratio);

                // Progress check: if global_L_sum didn't increase, all shards have
                // reached their L_capacity ceiling — no further progress is possible.
                if (_global_L_sum <= prev_global_L_sum) {
                    if (_debug) {
                        std::cout << "[Iter " << _iteration_count 
                                  << "] No progress (L_sum=" << _global_L_sum 
                                  << "), all shards at capacity. Stopping." << std::endl;
                    }
                    break;
                }

                // Check if we've reached L_end after this iteration
                if (_global_L_sum >= _params.L_end) {
                    break;
                }
            }
        }

        // Step 3: Final merge and return results
        get_final_results(k, id_maps, res_ids, res_dists, res_shard_ids);
    }

    /**
     * Get the total number of IOs across all shards.
     */
    uint64_t get_total_ios() const {
        uint64_t total = 0;
        for (const auto &shard_state : _shard_states) {
            total += shard_state.state.get_num_ios();
        }
        return total;
    }

    /**
     * Get the current global L sum.
     */
    uint64_t get_global_L_sum() const {
        return _global_L_sum;
    }

    /**
     * Get the number of iterations performed.
     */
    uint32_t get_iteration_count() const {
        return _iteration_count;
    }

    /**
     * Get per-shard statistics.
     */
    void get_shard_stats(std::vector<uint64_t> &shard_L_values,
                         std::vector<uint32_t> &shard_ios,
                         std::vector<bool> &shard_contributed) const {
        shard_L_values.clear();
        shard_ios.clear();
        shard_contributed.clear();

        for (const auto &shard_state : _shard_states) {
            shard_L_values.push_back(shard_state.state.get_current_L());
            shard_ios.push_back(shard_state.state.get_num_ios());
            shard_contributed.push_back(shard_state.contributes_to_topk);
        }
    }

    /**
     * Reset the context for a new search.
     */
    void reset() {
        for (auto &shard_state : _shard_states) {
            shard_state.state.reset();
            shard_state.contributes_to_topk = false;
            shard_state.consecutive_non_contributing = 0;
        }
        _global_L_sum = 0;
        _iteration_count = 0;
    }

  private:
    /**
     * Initialize search on all shards.
     * retset is sized to L_capacity (fixed), search pauses at L_init.
     */
    void initialize_all_shards(const T *query, const uint64_t k, const uint32_t mem_L,
                               const uint64_t beam_width, const uint32_t io_limit,
                               const float use_ratio) {
        _global_L_sum = 0;

        // Initialize and search each shard with L_capacity and L_init
        for (size_t i = 0; i < _shard_indices.size(); i++) {
            _shard_indices[i]->init_page_search(query, k, mem_L, 
                                                 _params.L_capacity, _params.L_init, 
                                                 beam_width, io_limit, use_ratio,
                                                 _shard_states[i].state);
        }

        // Update global L sum (uses L_stop, not L_capacity)
        for (auto &shard_state : _shard_states) {
            _global_L_sum += shard_state.state.get_current_L();
        }
    }

    /**
     * Merge current results from all shards.
     * Uses full_retset (full-precision distances) for accurate ordering.
     */
    void merge_current_results(const uint64_t k, 
                               const std::vector<std::vector<uint32_t>> &id_maps,
                               std::vector<GlobalSearchResult> &merged_results) {
        merged_results.clear();

        // Collect results from each shard's full_retset (full-precision distances)
        for (size_t shard_idx = 0; shard_idx < _shard_states.size(); shard_idx++) {
            auto &shard_state = _shard_states[shard_idx];
            auto &full_retset = shard_state.state.full_retset();

            // Sort full_retset by distance (full-precision)
            std::sort(full_retset.begin(), full_retset.end());

            // Get top-k from this shard
            size_t count = std::min(k, (uint64_t)full_retset.size());
            for (size_t i = 0; i < count; i++) {
                uint32_t local_id = full_retset[i].id;
                uint32_t global_id = id_maps[shard_state.shard_id][local_id];
                merged_results.emplace_back(global_id, full_retset[i].distance, 
                                           shard_state.shard_id);
            }
        }

        // Sort merged results by distance
        std::sort(merged_results.begin(), merged_results.end());
    }

    /**
     * Update shard contribution status and calculate new L values.
     */
    void update_shard_contributions(const std::set<size_t> &contributing_shards) {
        for (auto &shard_state : _shard_states) {
            bool contributes = contributing_shards.find(shard_state.shard_id) != 
                               contributing_shards.end();
            shard_state.contributes_to_topk = contributes;

            if (contributes) {
                shard_state.consecutive_non_contributing = 0;
            } else {
                shard_state.consecutive_non_contributing++;
            }
        }
    }

    /**
     * Resume search on all shards with updated L_stop values.
     * No capacity expansion needed — retset is already sized to L_capacity.
     */
    void resume_all_shards(const T *query, const uint64_t k, const uint32_t mem_L,
                           const uint64_t beam_width, const uint32_t io_limit,
                           const float use_ratio) {
        // Calculate new L_stop values for each shard
        std::vector<uint64_t> new_l_stop_values(_shard_states.size());
        uint64_t new_global_L_sum = 0;

        if (_debug) {
            std::cout << "[Iter " << _iteration_count << "] L_stop adjustments:" << std::endl;
        }

        for (size_t i = 0; i < _shard_states.size(); i++) {
            auto &shard_state = _shard_states[i];
            uint64_t current_l_stop = shard_state.state.get_current_L();
            uint64_t L_increment = shard_state.contributes_to_topk ? 
                                   _params.L_big : _params.L_small;
            // Clamp to L_capacity
            new_l_stop_values[i] = std::min(current_l_stop + L_increment, 
                                            _params.L_capacity);
            new_global_L_sum += new_l_stop_values[i];

            if (_debug) {
                std::cout << "  Shard " << shard_state.shard_id 
                          << ": L_stop " << current_l_stop << " -> " << new_l_stop_values[i]
                          << " (+" << L_increment << ", " 
                          << (shard_state.contributes_to_topk ? "contributing" : "non-contributing") 
                          << ", capacity=" << _params.L_capacity << ")" << std::endl;
            }
        }

        // Update global L sum
        _global_L_sum = new_global_L_sum;

        if (_debug) {
            std::cout << "  Global L_stop sum: " << _global_L_sum << " / " 
                      << _params.L_end << std::endl;
        }

        // Resume search on all shards with new stopping points
        for (size_t i = 0; i < _shard_indices.size(); i++) {
            if (new_l_stop_values[i] > _shard_states[i].state.get_current_L()) {
                _shard_indices[i]->resume_page_search(query, k, mem_L, new_l_stop_values[i],
                                                       beam_width, io_limit, use_ratio,
                                                       _shard_states[i].state);
            }
        }
    }

    /**
     * Get final merged results with global IDs.
     */
    void get_final_results(const uint64_t k,
                           const std::vector<std::vector<uint32_t>> &id_maps,
                           uint64_t *res_ids, float *res_dists, size_t *res_shard_ids) {
        // Collect results from each shard
        std::vector<GlobalSearchResult> merged_results;
        merged_results.reserve(_shard_states.size() * k);

        for (size_t shard_idx = 0; shard_idx < _shard_indices.size(); shard_idx++) {
            auto &shard_state = _shard_states[shard_idx];
            auto &full_retset = shard_state.state.full_retset();

            // Sort by distance
            std::sort(full_retset.begin(), full_retset.end());

            size_t count = std::min(k, (uint64_t)full_retset.size());
            for (size_t i = 0; i < count; i++) {
                uint32_t local_id = full_retset[i].id;
                uint32_t global_id = id_maps[shard_state.shard_id][local_id];
                merged_results.emplace_back(global_id, full_retset[i].distance,
                                           shard_state.shard_id);
            }
        }

        // Sort merged results by distance
        std::sort(merged_results.begin(), merged_results.end());

        // Remove duplicates (keep best distance)
        std::vector<GlobalSearchResult> unique_results;
        std::set<uint64_t> seen_ids;
        for (const auto &result : merged_results) {
            if (seen_ids.find(result.id) == seen_ids.end()) {
                seen_ids.insert(result.id);
                unique_results.push_back(result);
            }
        }

        // Copy to output
        size_t result_count = std::min(k, (uint64_t)unique_results.size());
        for (size_t i = 0; i < result_count; i++) {
            res_ids[i] = unique_results[i].id;
            if (res_dists != nullptr) {
                res_dists[i] = unique_results[i].distance;
            }
            if (res_shard_ids != nullptr) {
                res_shard_ids[i] = unique_results[i].shard_id;
            }
        }
    }

    // Hyperparameters
    MultiShardSearchParams _params;

    // Shard indices and states
    std::vector<PQFlashIndex<T> *> _shard_indices;
    std::vector<ShardSearchState<T>> _shard_states;

    // Global statistics
    uint64_t _global_L_sum;
    uint32_t _iteration_count;

    // Debug flag
    bool _debug = false;
};

} // namespace diskann
