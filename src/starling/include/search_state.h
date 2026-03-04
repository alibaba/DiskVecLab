// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <vector>
#include <memory>
#include <atomic>
#include <algorithm>
#include <string>
#include <limits>
#include "neighbor.h"
#include "tsl/robin_set.h"
#include "windows_customizations.h"

namespace diskann {

/**
 * SearchState encapsulates the state of a page search operation,
 * enabling pause/resume functionality for iterative search.
 * 
 * Design: retset capacity (_l_capacity) and search stopping point (_l_stop)
 * are decoupled. The retset always has a large fixed capacity, while L_init /
 * L_big / L_small only control _l_stop — the position up to which the search
 * cursor (k) is allowed to advance before pausing.
 * 
 * On resume, _l_stop is increased and the search continues from the saved
 * cursor without needing to expand capacity or re-insert candidates.
 * 
 * Key state components:
 * - retset: Sorted candidate vector with fixed capacity = _l_capacity
 * - _l_stop: Current stopping boundary (only controls search termination)
 * - _search_cursor: Saved k position for resume
 * - visited: Set of all visited node IDs (prevents re-visiting)
 * - full_retset: Full precision results for reordering
 * - Statistics: IO count, comparisons, hops
 */
template <typename T>
class SearchState {
  public:
    SearchState() : _l_capacity(0), _l_stop(0), _search_cursor(0),
                    _num_ios(0), _num_cmps(0), _num_hops(0), 
                    _cur_list_size(0), _is_initialized(false) {
    }

    /**
     * Initialize search state with separate capacity and stopping point.
     * @param l_capacity  Fixed retset capacity (candidate queue size).
     *                    This determines how many candidates retset can hold.
     * @param l_stop      Initial stopping point for the search cursor.
     *                    The search pauses when k reaches this position.
     * @param visited_reserve  Reserved size for visited set.
     */
    void initialize(uint64_t l_capacity, uint64_t l_stop, uint64_t visited_reserve = 4096) {
        _l_capacity = l_capacity;
        _l_stop = std::min(l_stop, l_capacity);
        _search_cursor = 0;
        _retset.clear();
        _retset.resize(l_capacity + 1);
        _cur_list_size = 0;
        _visited.clear();
        _visited.reserve(visited_reserve);
        _page_visited.clear();
        _page_visited.reserve(visited_reserve);
        _full_retset.clear();
        _full_retset.reserve(4096);
        _kicked.set_cap(l_capacity);
        _last_io_ids.clear();
        _last_pages.clear();
        _num_ios = 0;
        _num_cmps = 0;
        _num_hops = 0;
        _is_initialized = true;
    }

    // Reset state for new search
    void reset() {
        _retset.clear();
        _visited.clear();
        _page_visited.clear();
        _full_retset.clear();
        _last_io_ids.clear();
        _last_pages.clear();
        _num_ios = 0;
        _num_cmps = 0;
        _num_hops = 0;
        _l_capacity = 0;
        _l_stop = 0;
        _search_cursor = 0;
        _cur_list_size = 0;
        _is_initialized = false;
    }

    /**
     * Update the stopping point for resume (no capacity change needed).
     * Clamped to _l_capacity.
     */
    void set_l_stop(uint64_t new_l_stop) {
        _l_stop = std::min(new_l_stop, _l_capacity);
    }

    // Get fixed retset capacity
    uint64_t get_l_capacity() const { return _l_capacity; }

    // Get current stopping point
    uint64_t get_l_stop() const { return _l_stop; }

    // Save/restore search cursor position (k)
    unsigned get_search_cursor() const { return _search_cursor; }
    void set_search_cursor(unsigned k) { _search_cursor = k; }

    /**
     * get_current_L returns the stopping point (_l_stop).
     * This is used by MultiShardSearchContext to track the "logical L"
     * for global L_sum calculation and adaptive L allocation.
     */
    uint64_t get_current_L() const {
        return _l_stop;
    }

    // Check if search can continue (has unexpanded nodes within l_stop)
    // Note: flag=true means unexpanded (to be processed),
    //       flag=false means already expanded
    bool has_unexpanded_node() const {
        unsigned bound = std::min(_cur_list_size, (unsigned)_l_stop);
        for (unsigned i = 0; i < bound; i++) {
            if (_retset[i].flag) return true;
        }
        return false;
    }

    // Check if node was already visited
    bool is_visited(uint64_t node_id) const {
        return _visited.find(node_id) != _visited.end();
    }

    // Mark node as visited
    bool mark_visited(uint64_t node_id) {
        return _visited.insert(node_id).second;
    }

    // Check if page was already visited
    bool is_page_visited(unsigned page_id) const {
        return _page_visited.find(page_id) != _page_visited.end();
    }

    // Mark page as visited
    bool mark_page_visited(unsigned page_id) {
        return _page_visited.insert(page_id).second;
    }

    // Check if state is initialized
    bool is_initialized() const {
        return _is_initialized;
    }

    // Accessors for internal state
    std::vector<Neighbor> &retset() { return _retset; }
    const std::vector<Neighbor> &retset() const { return _retset; }

    unsigned& cur_list_size() { return _cur_list_size; }
    unsigned cur_list_size() const { return _cur_list_size; }

    tsl::robin_set<uint64_t> &visited() { return _visited; }
    const tsl::robin_set<uint64_t> &visited() const { return _visited; }

    tsl::robin_set<unsigned> &page_visited() { return _page_visited; }
    const tsl::robin_set<unsigned> &page_visited() const { return _page_visited; }

    std::vector<Neighbor> &full_retset() { return _full_retset; }
    const std::vector<Neighbor> &full_retset() const { return _full_retset; }

    NeighborVec &kicked() { return _kicked; }
    const NeighborVec &kicked() const { return _kicked; }

    // Statistics accessors
    uint32_t get_num_ios() const { return _num_ios; }
    void set_num_ios(uint32_t ios) { _num_ios = ios; }
    void add_ios(uint32_t ios) { _num_ios += ios; }

    uint32_t get_num_cmps() const { return _num_cmps; }
    void set_num_cmps(uint32_t cmps) { _num_cmps = cmps; }
    void add_cmps(uint32_t cmps) { _num_cmps += cmps; }

    uint32_t get_num_hops() const { return _num_hops; }
    void set_num_hops(uint32_t hops) { _num_hops = hops; }
    void add_hops(uint32_t hops) { _num_hops += hops; }

    // Get top-k results from current state
    void get_topk_results(uint64_t k, std::vector<uint64_t> &ids, 
                          std::vector<float> &distances) const {
        size_t result_count = std::min(k, static_cast<uint64_t>(_cur_list_size));
        ids.resize(result_count);
        distances.resize(result_count);
        for (size_t i = 0; i < result_count; i++) {
            ids[i] = _retset[i].id;
            distances[i] = _retset[i].distance;
        }
    }

    // Copy results to output arrays
    void copy_results(uint64_t k, uint64_t *res_ids, float *res_dists) const {
        size_t result_count = std::min(k, static_cast<uint64_t>(_cur_list_size));
        for (size_t i = 0; i < result_count; i++) {
            res_ids[i] = _retset[i].id;
            res_dists[i] = _retset[i].distance;
        }
    }

    // Deferred page data: persisted across pause/resume so the next iteration
    // can process co-located nodes while new I/O is in flight (matching the
    // original page_search pipeline exactly).
    std::vector<unsigned> &last_io_ids() { return _last_io_ids; }
    const std::vector<unsigned> &last_io_ids() const { return _last_io_ids; }

    std::vector<char> &last_pages() { return _last_pages; }
    const std::vector<char> &last_pages() const { return _last_pages; }

  private:
    std::vector<Neighbor> _retset;           // Candidate vector with fixed capacity = _l_capacity
    unsigned _cur_list_size;                 // Current number of valid entries in retset
    tsl::robin_set<uint64_t> _visited;       // All visited node IDs
    tsl::robin_set<unsigned> _page_visited;  // All visited page IDs
    std::vector<Neighbor> _full_retset;      // Full precision results (expanded nodes)
    NeighborVec _kicked;                     // Kicked neighbors that can be re-inserted

    uint64_t _l_capacity;                    // Fixed retset capacity (candidate queue size)
    uint64_t _l_stop;                        // Current search stopping point
    unsigned _search_cursor;                 // Saved k position for resume
    uint32_t _num_ios;                       // IO operations count
    uint32_t _num_cmps;                      // Distance comparisons
    uint32_t _num_hops;                      // Number of hops

    // Deferred page processing state (persisted across pause/resume)
    std::vector<unsigned> _last_io_ids;      // IDs from last frontier whose pages need processing
    std::vector<char> _last_pages;           // Raw page data for deferred processing

    bool _is_initialized;
};

/**
 * ShardSearchState wraps SearchState with shard-specific metadata
 * for multi-shard search scenarios.
 */
template <typename T>
struct ShardSearchState {
    size_t shard_id;                         // Identifier for this shard
    SearchState<T> state;                    // The search state for this shard
    bool contributes_to_topk;                // Whether this shard contributed to global top-k
    uint32_t consecutive_non_contributing;   // Rounds without contributing to top-k

    ShardSearchState() : shard_id(0), contributes_to_topk(false), 
                         consecutive_non_contributing(0) {
    }

    explicit ShardSearchState(size_t id) : shard_id(id), contributes_to_topk(false), 
                                           consecutive_non_contributing(0) {
    }
};

/**
 * MultiShardSearchParams holds hyperparameters for adaptive multi-shard search.
 * 
 * L_capacity: Fixed retset capacity per shard. The candidate queue always has
 *             this size. If 0, auto-computed from L_end and num_shards.
 * L_init:     Initial stopping point for the search cursor (not retset size).
 * L_big:      L_stop increment for shards contributing to top-k.
 * L_small:    L_stop increment for shards not contributing to top-k.
 * L_end:      Termination condition: stop when sum of L_stop values >= L_end.
 */
struct MultiShardSearchParams {
    uint64_t L_capacity;  // Fixed retset capacity per shard (0 = auto)
    uint64_t L_init;      // Initial search stopping point for all shards
    uint64_t L_big;       // L_stop increment for contributing shards
    uint64_t L_small;     // L_stop increment for non-contributing shards
    uint64_t L_end;       // Termination condition: sum of all L_stop values

    // Which per-iteration shard selection policy to use when use_adaptive_search=1.
    // - "contrib"      : contribution-based (default)
    // - "radius_topL"  : advance only the shard(s) with smallest per-shard top-L radius
    //                    where L is the shard's current L_stop.
    std::string adaptive_shard_scheduler;

    // Default values
    MultiShardSearchParams()
                : L_capacity(0), L_init(20), L_big(20), L_small(10), L_end(0),
                    adaptive_shard_scheduler("contrib") {
    }

    MultiShardSearchParams(uint64_t capacity, uint64_t init, uint64_t big, 
                           uint64_t small, uint64_t end)
                : L_capacity(capacity), L_init(init), L_big(big), L_small(small), L_end(end),
                    adaptive_shard_scheduler("contrib") {
    }

    // Backward-compatible constructor (without L_capacity, auto-compute)
    MultiShardSearchParams(uint64_t init, uint64_t big, uint64_t small, uint64_t end)
                : L_capacity(0), L_init(init), L_big(big), L_small(small), L_end(end),
                    adaptive_shard_scheduler("contrib") {
    }

    // Calculate default L_end based on number of shards
    void set_default_L_end(size_t num_shards, uint64_t multiplier = 40) {
        L_end = num_shards * multiplier;
    }

    // Auto-compute L_capacity if not explicitly set.
    // L_capacity is the retset size (candidate queue), independent of L_stop.
    // It should be large enough to retain good candidates across all iterations.
    // Default: L_end (the total budget), so each shard can hold plenty of candidates.
    // This is generous but safe — only L_stop controls how much work is done.
    void set_default_L_capacity(size_t num_shards) {
        if (L_capacity == 0 && L_end > 0) {
            // Default: enough capacity to hold the max L any single shard could reach
            // Use L_end / num_shards as a reasonable estimate
            L_capacity = std::max(L_init + L_big, L_end);
            // diskann::cout << "Auto-computed L_capacity: " << L_capacity << std::endl;
        }
    }
};

/**
 * GlobalSearchResult holds a result with shard attribution for multi-shard merge.
 */
struct GlobalSearchResult {
    uint64_t id;           // Node ID (global)
    float distance;        // Distance to query
    size_t shard_id;       // Which shard this result came from

    GlobalSearchResult() : id(0), distance(std::numeric_limits<float>::max()), 
                           shard_id(0) {
    }

    GlobalSearchResult(uint64_t id, float dist, size_t shard)
        : id(id), distance(dist), shard_id(shard) {
    }

    bool operator<(const GlobalSearchResult &other) const {
        return distance < other.distance || 
               (distance == other.distance && id < other.id);
    }
};

} // namespace diskann
