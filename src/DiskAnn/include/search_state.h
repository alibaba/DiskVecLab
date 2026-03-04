// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <vector>
#include <memory>
#include <atomic>
#include "neighbor.h"
#include "tsl/robin_set.h"
#include "windows_customizations.h"

namespace diskann
{

// Forward declaration
template <typename T> class PQScratch;

/**
 * SearchState encapsulates the state of a beam search operation,
 * enabling pause/resume functionality for iterative search.
 * 
 * Key state components:
 * - retset: Priority queue of candidates with expansion status
 * - visited: Set of all visited node IDs (prevents re-visiting)
 * - full_retset: Full precision results for reordering
 * - Statistics: IO count, comparisons, hops
 */
template <typename T>
class SearchState
{
  public:
    SearchState() : _current_L(0), _num_ios(0), _num_cmps(0), _num_hops(0), _is_initialized(false)
    {
    }

    // Initialize search state with given L value
    void initialize(uint64_t l_search, uint64_t visited_reserve = 4096)
    {
        _current_L = l_search;
        _retset.clear();  // Clear any existing data first
        _retset.reserve(l_search);
        _visited.clear();
        _visited.reserve(visited_reserve);
        _full_retset.clear();
        _all_candidates.clear();
        _num_ios = 0;
        _num_cmps = 0;
        _num_hops = 0;
        _is_initialized = true;
    }

    // Reset state for new search
    void reset()
    {
        _retset.clear();
        _visited.clear();
        _full_retset.clear();
        _all_candidates.clear();
        _num_ios = 0;
        _num_cmps = 0;
        _num_hops = 0;
        _current_L = 0;
        _is_initialized = false;
    }

    // Expand search capacity to new_L
    // This allows the search to accept more candidates without re-visiting
    // already expanded nodes
    void expand_capacity(uint64_t new_L)
    {
        if (new_L <= _current_L)
        {
            return; // No expansion needed
        }

        // Expand retset capacity
        _retset.set_capacity(new_L);

        // Re-insert ALL discovered candidates that may have been expelled due to capacity
        // This includes both expanded nodes (full_retset) and unexpanded neighbors (all_candidates)
        // The key insight: neighbors discovered but expelled before being expanded can now fit
        for (const auto &neighbor : _all_candidates)
        {
            // Insert will be a no-op if already in retset (same ID check)
            // Newly inserted nodes will have expanded=false, allowing them to be explored
            _retset.insert(neighbor);
        }

        // CRITICAL: After inserting new candidates, we need to find the first unexpanded node
        // Scan through retset to find the first unexpanded position
        _retset.reset_cursor();

        _current_L = new_L;
    }

    // Add a discovered candidate (called when a neighbor is discovered)
    // This tracks ALL candidates regardless of whether they fit in retset
    void add_candidate(const Neighbor &candidate)
    {
        _all_candidates.push_back(candidate);
    }

    // Check if search can continue (has unexpanded nodes)
    bool has_unexpanded_node() const
    {
        return _retset.has_unexpanded_node();
    }

    // Get current L value
    uint64_t get_current_L() const
    {
        return _current_L;
    }

    // Check if node was already visited
    bool is_visited(uint64_t node_id) const
    {
        return _visited.find(node_id) != _visited.end();
    }

    // Mark node as visited
    bool mark_visited(uint64_t node_id)
    {
        return _visited.insert(node_id).second;
    }

    // Check if state is initialized
    bool is_initialized() const
    {
        return _is_initialized;
    }

    // Accessors for internal state
    NeighborPriorityQueue &retset()
    {
        return _retset;
    }
    const NeighborPriorityQueue &retset() const
    {
        return _retset;
    }

    tsl::robin_set<uint64_t> &visited()
    {
        return _visited;
    }
    const tsl::robin_set<uint64_t> &visited() const
    {
        return _visited;
    }

    std::vector<Neighbor> &full_retset()
    {
        return _full_retset;
    }
    const std::vector<Neighbor> &full_retset() const
    {
        return _full_retset;
    }

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
    void get_topk_results(uint64_t k, std::vector<uint64_t> &ids, std::vector<float> &distances) const
    {
        size_t result_count = std::min(k, static_cast<uint64_t>(_retset.size()));
        ids.resize(result_count);
        distances.resize(result_count);
        for (size_t i = 0; i < result_count; i++)
        {
            ids[i] = _retset[i].id;
            distances[i] = _retset[i].distance;
        }
    }

    // Copy results to output arrays
    void copy_results(uint64_t k, uint64_t *res_ids, float *res_dists) const
    {
        size_t result_count = std::min(k, static_cast<uint64_t>(_retset.size()));
        for (size_t i = 0; i < result_count; i++)
        {
            res_ids[i] = _retset[i].id;
            res_dists[i] = _retset[i].distance;
        }
    }

  private:
    NeighborPriorityQueue _retset;           // Result queue with capacity = L
    tsl::robin_set<uint64_t> _visited;       // All visited node IDs
    std::vector<Neighbor> _full_retset;      // Full precision results (expanded nodes)
    std::vector<Neighbor> _all_candidates;   // ALL discovered candidates (for capacity expansion)

    uint64_t _current_L;                     // Current search range
    uint32_t _num_ios;                       // IO operations count
    uint32_t _num_cmps;                      // Distance comparisons
    uint32_t _num_hops;                      // Number of hops

    bool _is_initialized;
};

/**
 * ShardSearchState wraps SearchState with shard-specific metadata
 * for multi-shard search scenarios.
 */
template <typename T>
struct ShardSearchState
{
    size_t shard_id;                         // Identifier for this shard
    SearchState<T> state;                    // The search state for this shard
    bool contributes_to_topk;                // Whether this shard contributed to global top-k
    uint32_t consecutive_non_contributing;   // Rounds without contributing to top-k

    ShardSearchState() : shard_id(0), contributes_to_topk(false), consecutive_non_contributing(0)
    {
    }

    explicit ShardSearchState(size_t id) : shard_id(id), contributes_to_topk(false), consecutive_non_contributing(0)
    {
    }
};

/**
 * MultiShardSearchParams holds hyperparameters for adaptive multi-shard search.
 */
struct MultiShardSearchParams
{
    uint64_t L_init;    // Initial search range for all shards
    uint64_t L_big;     // L increment for shards contributing to top-k
    uint64_t L_small;   // L increment for shards not contributing to top-k
    uint64_t L_end;     // Termination condition: sum of all L values

    // Default values for k=10
    MultiShardSearchParams()
        : L_init(20), L_big(20), L_small(10), L_end(0)
    {
    }

    MultiShardSearchParams(uint64_t init, uint64_t big, uint64_t small, uint64_t end)
        : L_init(init), L_big(big), L_small(small), L_end(end)
    {
    }

    // Calculate default L_end based on number of shards
    void set_default_L_end(size_t num_shards, uint64_t multiplier = 40)
    {
        L_end = num_shards * multiplier;
    }
};

/**
 * GlobalSearchResult holds a result with shard attribution for multi-shard merge.
 */
struct GlobalSearchResult
{
    uint64_t id;           // Node ID
    float distance;        // Distance to query
    size_t shard_id;       // Which shard this result came from

    GlobalSearchResult() : id(0), distance(std::numeric_limits<float>::max()), shard_id(0)
    {
    }

    GlobalSearchResult(uint64_t id, float dist, size_t shard)
        : id(id), distance(dist), shard_id(shard)
    {
    }

    bool operator<(const GlobalSearchResult &other) const
    {
        return distance < other.distance || (distance == other.distance && id < other.id);
    }
};

} // namespace diskann
