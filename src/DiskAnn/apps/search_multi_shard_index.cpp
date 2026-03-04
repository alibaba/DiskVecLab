// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "common_includes.h"
#include <boost/program_options.hpp>
#include <fstream>
#include <sstream>

#include "index.h"
#include "disk_utils.h"
#include "math_utils.h"
#include "memory_mapper.h"
#include "partition.h"
#include "pq_flash_index.h"
#include "timer.h"
#include "percentile_stats.h"
#include "program_options_utils.hpp"
#include "multi_shard_search.h"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "linux_aligned_file_reader.h"
#else
#ifdef USE_BING_INFRA
#include "bing_aligned_file_reader.h"
#else
#include "windows_aligned_file_reader.h"
#endif
#endif

namespace po = boost::program_options;

// Parse index paths from either a file or comma-separated string
std::vector<std::string> parse_index_paths(const std::string &index_paths_input)
{
    std::vector<std::string> paths;

    // Check if input is a file
    if (file_exists(index_paths_input))
    {
        std::ifstream file(index_paths_input);
        std::string line;
        while (std::getline(file, line))
        {
            // Trim whitespace
            line.erase(0, line.find_first_not_of(" \t\r\n"));
            line.erase(line.find_last_not_of(" \t\r\n") + 1);
            if (!line.empty())
            {
                paths.push_back(line);
            }
        }
    }
    else
    {
        // Parse as comma-separated list
        std::stringstream ss(index_paths_input);
        std::string path;
        while (std::getline(ss, path, ','))
        {
            // Trim whitespace
            path.erase(0, path.find_first_not_of(" \t\r\n"));
            path.erase(path.find_last_not_of(" \t\r\n") + 1);
            if (!path.empty())
            {
                paths.push_back(path);
            }
        }
    }

    return paths;
}

template <typename T, typename LabelT = uint32_t>
int search_multi_shard_index(diskann::Metric &metric, const std::vector<std::string> &index_path_prefixes,
                             const std::string &result_output_prefix, const std::string &query_file,
                             std::string &gt_file, const uint32_t num_threads, const uint32_t recall_at,
                             const uint32_t beamwidth, const uint32_t num_nodes_to_cache,
                             const uint64_t L_init, const uint64_t L_big, const uint64_t L_small,
                             const uint64_t L_end, const float fail_if_recall_below,
                             const std::string &quantification_type = "PQ",
                             const bool use_rabitq_fastscan = true, const bool rabit_use_hacc = true,
                             const bool debug = false)
{
    size_t num_shards = index_path_prefixes.size();
    diskann::cout << "Multi-Shard Search Parameters:" << std::endl;
    diskann::cout << "  - Number of shards: " << num_shards << std::endl;
    diskann::cout << "  - L_init: " << L_init << std::endl;
    diskann::cout << "  - L_big: " << L_big << std::endl;
    diskann::cout << "  - L_small: " << L_small << std::endl;
    diskann::cout << "  - L_end: " << L_end << std::endl;
    diskann::cout << "  - num_threads: " << num_threads << std::endl;
    diskann::cout << "  - beamwidth: " << beamwidth << std::endl;
    diskann::cout << "  - quantification_type: " << quantification_type << std::endl;

    // Parse PQ type
    PQType pq_type;
    if (quantification_type == "PQ")
    {
        pq_type = PQType::PQ;
    }
    else if (quantification_type == "OPQ")
    {
        pq_type = PQType::OPQ;
    }
    else if (quantification_type == "LSQ")
    {
        pq_type = PQType::LSQ;
    }
    else if (quantification_type == "RABITQ")
    {
        pq_type = PQType::RABITQ;
    }
    else if (quantification_type == "FLAT")
    {
        pq_type = PQType::FULL_PRECISION;
    }
    else
    {
        diskann::cout << "Invalid quantification_type: " << quantification_type << std::endl;
        return -1;
    }

    // Load query file
    T *query = nullptr;
    uint32_t *gt_ids = nullptr;
    float *gt_dists = nullptr;
    size_t query_num, query_dim, query_aligned_dim, gt_num, gt_dim;
    diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim, query_aligned_dim);

    bool calc_recall_flag = false;
    if (gt_file != std::string("null") && gt_file != std::string("NULL") && file_exists(gt_file))
    {
        diskann::load_truthset(gt_file, gt_ids, gt_dists, gt_num, gt_dim);
        if (gt_num != query_num)
        {
            diskann::cout << "Error. Mismatch in number of queries and ground truth data" << std::endl;
        }
        calc_recall_flag = true;
    }

    // Load all shard indices
    std::vector<std::shared_ptr<AlignedFileReader>> readers(num_shards);
    std::vector<std::unique_ptr<diskann::PQFlashIndex<T, LabelT>>> shard_indices(num_shards);

    diskann::cout << "Loading " << num_shards << " shard indices..." << std::endl;
    for (size_t i = 0; i < num_shards; i++)
    {
        diskann::cout << "  Loading shard " << i << ": " << index_path_prefixes[i] << std::endl;

#ifdef _WINDOWS
#ifndef USE_BING_INFRA
        readers[i].reset(new WindowsAlignedFileReader());
#else
        readers[i].reset(new diskann::BingAlignedFileReader());
#endif
#else
        readers[i].reset(new LinuxAlignedFileReader());
#endif

        shard_indices[i].reset(new diskann::PQFlashIndex<T, LabelT>(readers[i], metric));
        int res = shard_indices[i]->load(num_threads, index_path_prefixes[i].c_str(), pq_type);

        if (pq_type == PQType::RABITQ)
        {
            shard_indices[i]->set_rabitq_params(use_rabitq_fastscan, rabit_use_hacc);
        }

        if (res != 0)
        {
            diskann::cout << "Failed to load shard " << i << std::endl;
            return res;
        }

        // Cache nodes for this shard
        if (num_nodes_to_cache > 0)
        {
            std::vector<uint32_t> node_list;
            shard_indices[i]->cache_bfs_levels(num_nodes_to_cache, node_list);
            shard_indices[i]->load_cache_list(node_list);
        }
    }
    diskann::cout << "All shards loaded successfully." << std::endl;

    // Load ID mapping files for each shard (for converting local IDs to global IDs)
    std::vector<std::vector<uint32_t>> shard_id_maps(num_shards);
    bool id_maps_loaded = true;
    for (size_t i = 0; i < num_shards; i++)
    {
        // Try to find ID mapping file: {index_prefix}_ids_uint32.bin
        std::string idmap_file = index_path_prefixes[i] + "_ids_uint32.bin";
        if (file_exists(idmap_file))
        {
            diskann::read_idmap(idmap_file, shard_id_maps[i]);
            diskann::cout << "  Loaded ID map for shard " << i << ": " << shard_id_maps[i].size() << " entries" << std::endl;
        }
        else
        {
            diskann::cout << "  Warning: ID map file not found for shard " << i << ": " << idmap_file << std::endl;
            id_maps_loaded = false;
        }
    }
    if (!id_maps_loaded && calc_recall_flag)
    {
        diskann::cerr << "Error: ID map files required for recall calculation but not found." << std::endl;
        diskann::cerr << "Expected file pattern: {index_prefix}_ids_uint32.bin" << std::endl;
        return -1;
    }

    omp_set_num_threads(num_threads);

    // Setup multi-shard search parameters
    diskann::MultiShardSearchParams params;
    params.L_init = L_init;
    params.L_big = L_big;
    params.L_small = L_small;
    params.L_end = (L_end == 0) ? num_shards * 40 : L_end;

    // Prepare result storage
    std::vector<uint64_t> query_result_ids(recall_at * query_num);
    std::vector<float> query_result_dists(recall_at * query_num);
    std::vector<size_t> query_result_shards(recall_at * query_num);

    // Statistics
    std::vector<uint64_t> query_total_ios(query_num, 0);
    std::vector<uint32_t> query_iterations(query_num, 0);
    std::vector<uint64_t> query_final_L_sum(query_num, 0);

    diskann::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
    diskann::cout.precision(2);

    diskann::cout << "\nStarting multi-shard search..." << std::endl;
    diskann::cout << "=================================================================="
                     "=================================================================="
                  << std::endl;

    diskann::Timer query_timer;

    // Execute search for all queries
#pragma omp parallel for schedule(dynamic, 1)
    for (int64_t i = 0; i < (int64_t)query_num; i++)
    {
        // Create search context for this query
        diskann::MultiShardSearchContext<T, LabelT> context(params);

        // Add all shards
        for (size_t s = 0; s < num_shards; s++)
        {
            context.add_shard(s, shard_indices[s].get());
        }

        // Execute search (enable debug for first query only)
        bool debug_this_query = debug && (i == 0);
        context.search(query + (i * query_aligned_dim), recall_at, beamwidth,
                       query_result_ids.data() + (i * recall_at),
                       query_result_dists.data() + (i * recall_at),
                       query_result_shards.data() + (i * recall_at),
                       debug_this_query);

        // Collect statistics
        query_total_ios[i] = context.get_total_ios();
        query_iterations[i] = context.get_iteration_count();
        query_final_L_sum[i] = context.get_global_L_sum();
    }

    double total_time = query_timer.elapsed() / 1000000.0; // Convert to seconds
    double qps = query_num / total_time;

    // Calculate statistics
    double mean_ios = 0, mean_iterations = 0, mean_L_sum = 0;
    for (size_t i = 0; i < query_num; i++)
    {
        mean_ios += query_total_ios[i];
        mean_iterations += query_iterations[i];
        mean_L_sum += query_final_L_sum[i];
    }
    mean_ios /= query_num;
    mean_iterations /= query_num;
    mean_L_sum /= query_num;

    // Calculate recall
    double recall = 0;
    if (calc_recall_flag)
    {
        std::vector<uint32_t> query_result_ids_32(recall_at * query_num);
        
        // Convert local shard IDs to global IDs using ID maps
        for (size_t i = 0; i < recall_at * query_num; i++)
        {
            uint32_t local_id = static_cast<uint32_t>(query_result_ids[i]);
            size_t shard_id = query_result_shards[i];
            
            // Convert local ID to global ID using ID map
            if (shard_id >= shard_id_maps.size() || shard_id_maps[shard_id].empty())
            {
                diskann::cerr << "Error: ID map not loaded for shard " << shard_id << std::endl;
                return -1;
            }
            if (local_id >= shard_id_maps[shard_id].size())
            {
                diskann::cerr << "Error: Local ID " << local_id << " out of range for shard " 
                              << shard_id << " (max: " << shard_id_maps[shard_id].size() - 1 << ")" << std::endl;
                return -1;
            }
            query_result_ids_32[i] = shard_id_maps[shard_id][local_id];
        }
        
        recall = diskann::calculate_recall((uint32_t)query_num, gt_ids, gt_dists, (uint32_t)gt_dim,
                                           query_result_ids_32.data(), query_result_dists.data(),
                                           recall_at, recall_at, nullptr);
    }

    // Print results
    diskann::cout << "\nMulti-Shard Search Results:" << std::endl;
    diskann::cout << "  - Total queries: " << query_num << std::endl;
    diskann::cout << "  - Total time: " << total_time << " seconds" << std::endl;
    diskann::cout << "  - QPS: " << qps << std::endl;
    diskann::cout << "  - Mean latency: " << (total_time / query_num * 1000000) << " us" << std::endl;
    diskann::cout << "  - Mean IOs: " << mean_ios << std::endl;
    diskann::cout << "  - Mean iterations: " << mean_iterations << std::endl;
    diskann::cout << "  - Mean final L sum: " << mean_L_sum << std::endl;
    if (calc_recall_flag)
    {
        diskann::cout << "  - Recall@" << recall_at << ": " << recall << std::endl;
    }

    // Print shard distribution in top-k results
    std::vector<size_t> shard_contribution(num_shards, 0);
    for (size_t i = 0; i < query_num * recall_at; i++)
    {
        shard_contribution[query_result_shards[i]]++;
    }
    diskann::cout << "\nShard contribution to results:" << std::endl;
    for (size_t s = 0; s < num_shards; s++)
    {
        double percentage = 100.0 * shard_contribution[s] / (query_num * recall_at);
        diskann::cout << "  Shard " << s << ": " << shard_contribution[s] << " (" << percentage << "%)" << std::endl;
    }

    // Save results
    if (!result_output_prefix.empty())
    {
        std::string result_ids_file = result_output_prefix + "_ids_uint64.bin";
        std::string result_dists_file = result_output_prefix + "_dists_float.bin";
        std::string result_shards_file = result_output_prefix + "_shards.bin";

        diskann::save_bin<uint64_t>(result_ids_file, query_result_ids.data(), query_num, recall_at);
        diskann::save_bin<float>(result_dists_file, query_result_dists.data(), query_num, recall_at);

        diskann::cout << "\nResults saved to:" << std::endl;
        diskann::cout << "  - " << result_ids_file << std::endl;
        diskann::cout << "  - " << result_dists_file << std::endl;
    }

    diskann::aligned_free(query);
    return (recall >= fail_if_recall_below) ? 0 : -1;
}

int main(int argc, char **argv)
{
    std::string data_type, dist_fn, index_paths_input, result_path_prefix, query_file, gt_file,
        quantification_type;
    uint32_t num_threads, K, W, num_nodes_to_cache;
    uint64_t L_init, L_big, L_small, L_end;
    bool use_rabitq_fastscan = true;
    bool rabit_use_hacc = true;
    bool debug = false;
    float fail_if_recall_below = 0.0f;

    po::options_description desc{
        program_options_utils::make_program_description("search_multi_shard_index",
                                                        "Searches multiple DiskANN indexes with adaptive L allocation")};
    try
    {
        desc.add_options()("help,h", "Print information on arguments");

        // Required parameters
        po::options_description required_configs("Required");
        required_configs.add_options()("data_type", po::value<std::string>(&data_type)->required(),
                                       program_options_utils::DATA_TYPE_DESCRIPTION);
        required_configs.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                                       program_options_utils::DISTANCE_FUNCTION_DESCRIPTION);
        required_configs.add_options()("index_path_prefixes",
                                       po::value<std::string>(&index_paths_input)->required(),
                                       "Comma-separated list of index path prefixes, or path to a file containing "
                                       "one index path prefix per line");
        required_configs.add_options()("result_path", po::value<std::string>(&result_path_prefix)->required(),
                                       program_options_utils::RESULT_PATH_DESCRIPTION);
        required_configs.add_options()("query_file", po::value<std::string>(&query_file)->required(),
                                       program_options_utils::QUERY_FILE_DESCRIPTION);
        required_configs.add_options()("recall_at,K", po::value<uint32_t>(&K)->required(),
                                       program_options_utils::NUMBER_OF_RESULTS_DESCRIPTION);
        required_configs.add_options()("quantification_type,Q",
                                       po::value<std::string>(&quantification_type)->required(),
                                       program_options_utils::QUANTIFICATION_TYPE_DESCRIPTION);

        // Multi-shard specific parameters
        po::options_description multishard_configs("Multi-Shard Parameters");
        multishard_configs.add_options()("L_init",
                                         po::value<uint64_t>(&L_init)->default_value(20),
                                         "Initial search list size for all shards. Default: 20");
        multishard_configs.add_options()("L_big",
                                         po::value<uint64_t>(&L_big)->default_value(20),
                                         "L increment for shards contributing to top-k. Default: 20");
        multishard_configs.add_options()("L_small",
                                         po::value<uint64_t>(&L_small)->default_value(10),
                                         "L increment for shards NOT contributing to top-k. Default: 10");
        multishard_configs.add_options()("L_end",
                                         po::value<uint64_t>(&L_end)->default_value(0),
                                         "Termination condition: stop when sum of all L >= L_end. "
                                         "Default: num_shards * 40");

        // Optional parameters
        po::options_description optional_configs("Optional");
        optional_configs.add_options()("gt_file", po::value<std::string>(&gt_file)->default_value(std::string("null")),
                                       program_options_utils::GROUND_TRUTH_FILE_DESCRIPTION);
        optional_configs.add_options()("beamwidth,W", po::value<uint32_t>(&W)->default_value(2),
                                       program_options_utils::BEAMWIDTH);
        optional_configs.add_options()("num_nodes_to_cache",
                                       po::value<uint32_t>(&num_nodes_to_cache)->default_value(0),
                                       program_options_utils::NUMBER_OF_NODES_TO_CACHE);
        optional_configs.add_options()("num_threads,T",
                                       po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                                       program_options_utils::NUMBER_THREADS_DESCRIPTION);
        optional_configs.add_options()("fail_if_recall_below",
                                       po::value<float>(&fail_if_recall_below)->default_value(0.0f),
                                       program_options_utils::FAIL_IF_RECALL_BELOW);
        optional_configs.add_options()("use_rabitq_fastscan",
                                       po::value<bool>(&use_rabitq_fastscan)->default_value(true),
                                       "Use RabitQ FastScan for distance computation. Default: true");
        optional_configs.add_options()("rabit_use_hacc",
                                       po::value<bool>(&rabit_use_hacc)->default_value(true),
                                       "Use high accuracy FastScan for RabitQ. Default: true");
        optional_configs.add_options()("debug",
                                       po::value<bool>(&debug)->default_value(false),
                                       "Enable debug logging for first query. Default: false");

        // Merge all parameter groups
        desc.add(required_configs).add(multishard_configs).add(optional_configs);

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << '\n';
        return -1;
    }

    // Parse metric
    diskann::Metric metric;
    if (dist_fn == std::string("mips"))
    {
        metric = diskann::Metric::INNER_PRODUCT;
    }
    else if (dist_fn == std::string("l2"))
    {
        metric = diskann::Metric::L2;
    }
    else if (dist_fn == std::string("cosine"))
    {
        metric = diskann::Metric::COSINE;
    }
    else
    {
        std::cout << "Unsupported distance function. Currently only L2/Inner Product/Cosine are supported."
                  << std::endl;
        return -1;
    }

    // Parse index paths
    std::vector<std::string> index_path_prefixes = parse_index_paths(index_paths_input);
    if (index_path_prefixes.empty())
    {
        std::cerr << "Error: No valid index paths provided" << std::endl;
        return -1;
    }

    diskann::cout << "Parsed " << index_path_prefixes.size() << " shard index paths:" << std::endl;
    for (size_t i = 0; i < index_path_prefixes.size(); i++)
    {
        diskann::cout << "  [" << i << "] " << index_path_prefixes[i] << std::endl;
    }

    try
    {
        if (data_type == std::string("float"))
            return search_multi_shard_index<float>(metric, index_path_prefixes, result_path_prefix, query_file,
                                                   gt_file, num_threads, K, W, num_nodes_to_cache,
                                                   L_init, L_big, L_small, L_end, fail_if_recall_below,
                                                   quantification_type, use_rabitq_fastscan, rabit_use_hacc, debug);
        else if (data_type == std::string("int8"))
            return search_multi_shard_index<int8_t>(metric, index_path_prefixes, result_path_prefix, query_file,
                                                    gt_file, num_threads, K, W, num_nodes_to_cache,
                                                    L_init, L_big, L_small, L_end, fail_if_recall_below,
                                                    quantification_type, use_rabitq_fastscan, rabit_use_hacc, debug);
        else if (data_type == std::string("uint8"))
            return search_multi_shard_index<uint8_t>(metric, index_path_prefixes, result_path_prefix, query_file,
                                                     gt_file, num_threads, K, W, num_nodes_to_cache,
                                                     L_init, L_big, L_small, L_end, fail_if_recall_below,
                                                     quantification_type, use_rabitq_fastscan, rabit_use_hacc, debug);
        else
        {
            std::cerr << "Unsupported data type. Use float or int8 or uint8" << std::endl;
            return -1;
        }
    }
    catch (const std::exception &e)
    {
        std::cout << std::string(e.what()) << std::endl;
        diskann::cerr << "Multi-shard index search failed." << std::endl;
        return -1;
    }
}
