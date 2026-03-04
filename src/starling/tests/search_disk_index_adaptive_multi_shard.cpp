// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
//
// Adaptive Multi-Shard Search: Implements DiskANN-style adaptive L allocation
// where shards contributing to top-k get larger L increments (L_big) while
// non-contributing shards get smaller increments (L_small).

#include <atomic>
#include <cstring>
#include <iomanip>
#include <mutex>
#include <omp.h>
#include <pq_flash_index.h>
#include <set>
#include <string.h>
#include <time.h>
#include <unordered_set>
#include <boost/program_options.hpp>

#include "aux_utils.h"
#include "index.h"
#include "math_utils.h"
#include "memory_mapper.h"
#include "partition_and_pq.h"
#include "timer.h"
#include "utils.h"
#include "percentile_stats.h"
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
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;

namespace po = boost::program_options;

// Use MultiShardSearchParams from multi_shard_search.h
using AdaptiveSearchParams = diskann::MultiShardSearchParams;

template<typename T>
int search_disk_index_adaptive_multi_shard(
    diskann::Metric& metric, 
    const std::string& parent_index_path,
    const std::string& data_path,
    const std::string& query_file,
    const std::string& gt_file,
    const unsigned num_threads,
    const unsigned recall_at,
    const unsigned beamwidth,
    const unsigned num_nodes_to_cache,
    const _u32 search_io_limit,
    const _u32 mem_L,
    const std::string& split_type,
    const _u32 split_K,
    const std::string& disk_file_name,
    const std::string& quantification_type,
    const AdaptiveSearchParams& adaptive_params,
    const bool use_page_search = true,
    const float use_ratio = 1.0,
    const bool use_reorder_data = false,
    const bool use_sq = false,
    const bool debug = false) {

    diskann::cout << "Adaptive Multi-Shard Search Parameters (Incremental):" << std::endl;
    diskann::cout << "  L_capacity: " << adaptive_params.L_capacity << " (0=auto)" << std::endl;
    diskann::cout << "  L_init: " << adaptive_params.L_init << std::endl;
    diskann::cout << "  L_big: " << adaptive_params.L_big << std::endl;
    diskann::cout << "  L_small: " << adaptive_params.L_small << std::endl;
    diskann::cout << "  L_end: " << adaptive_params.L_end << std::endl;
    diskann::cout << "  Using incremental search (init_page_search/resume_page_search)" << std::endl;

    PQType pq_type;
    if (quantification_type == "PQ") {
        pq_type = PQType::PQ;
    } else if (quantification_type == "OPQ") {
        pq_type = PQType::OPQ;
    } else if (quantification_type == "LSQ") {
        pq_type = PQType::LSQ;
    } else if (quantification_type == "RABITQ") {
        pq_type = PQType::RABITQ;
    } else {
        diskann::cout << "Invalid quantification_type: " << quantification_type << std::endl;
        return -1;
    }

    // Load query data
    T* query = nullptr;
    unsigned* gt_ids = nullptr;
    float* gt_dists = nullptr;
    size_t query_num, query_dim, query_aligned_dim, gt_num, gt_dim;
    diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim, query_aligned_dim);

    bool calc_recall_flag = false;
    if (gt_file != std::string("null") && gt_file != std::string("NULL") && file_exists(gt_file)) {
        diskann::load_truthset(gt_file, gt_ids, gt_dists, gt_num, gt_dim);
        if (gt_num != query_num) {
            diskann::cout << "Error: Mismatch in number of queries and ground truth" << std::endl;
            exit(-1);
        }
        calc_recall_flag = true;
    }

    // Load shard index paths
    std::cout << "Loading indices from: " << parent_index_path << std::endl;
    std::vector<std::string> shard_paths;
    for (const auto& entry : fs::directory_iterator(parent_index_path)) {
        if (!fs::is_directory(entry.path())) continue;
        std::string filename = entry.path().filename().string();
        if (filename.find(split_type + "_shard_") != std::string::npos) {
            shard_paths.push_back(entry.path().string());
        }
    }

    auto get_id = [](const std::string& path) {
        size_t shard_pos = path.find("shard_");
        if (shard_pos == std::string::npos) {
            throw std::invalid_argument("Invalid shard path format");
        }
        size_t start_pos = shard_pos + 6;
        size_t end_pos = path.find('_', start_pos);
        if (end_pos == std::string::npos) {
            throw std::invalid_argument("Invalid shard path format");
        }
        return std::stoi(path.substr(start_pos, end_pos - start_pos));
    };

    std::sort(shard_paths.begin(), shard_paths.end(), 
              [&get_id](const std::string& a, const std::string& b) {
        return get_id(a) < get_id(b);
    });

    size_t num_shards = shard_paths.size();
    diskann::cout << "Found " << num_shards << " shards" << std::endl;

    // Load shard ID mappings
    std::vector<std::string> shard_id_paths;
    for (const auto& entry : fs::directory_iterator(data_path)) {
        if (fs::is_directory(entry.path())) continue;
        std::string filename = entry.path().filename().string();
        if (filename.find("_subshard-") != std::string::npos && 
            filename.find("_ids_uint32") != std::string::npos) {
            shard_id_paths.push_back(entry.path().string());
        }
    }

    std::sort(shard_id_paths.begin(), shard_id_paths.end(), 
              [](const std::string& a, const std::string& b) {
        auto get_id = [](const std::string& path) {
            size_t start_pos = path.find("_subshard-") + 10;
            size_t end_pos = path.find_first_not_of("0123456789", start_pos);
            return std::stoi(path.substr(start_pos, end_pos - start_pos));
        };
        return get_id(a) < get_id(b);
    });

    if (num_shards != shard_id_paths.size()) {
        diskann::cout << "Error: Mismatch in number of index and ID shards" << std::endl;
        exit(-1);
    }

    // Load original IDs for each shard
    std::vector<std::vector<uint32_t>> original_ids(num_shards);
    for (_u32 shard = 0; shard < num_shards; shard++) {
        uint32_t* ids = nullptr;
        size_t npts, nd;
        diskann::load_bin<uint32_t>(shard_id_paths[shard], ids, npts, nd);
        assert(nd == 1 && "ID file must have dimension 1");
        original_ids[shard] = std::vector<uint32_t>(ids, ids + npts);
        delete[] ids;
    }

    // Load indices
    std::vector<std::shared_ptr<AlignedFileReader>> readers(num_shards);
    std::vector<std::unique_ptr<diskann::PQFlashIndex<T>>> flash_indices;

    for (_u32 shard = 0; shard < num_shards; shard++) {
        std::string index_path = shard_paths[shard] + "/";
#ifdef _WINDOWS
        readers[shard].reset(new WindowsAlignedFileReader());
#else
        readers[shard].reset(new LinuxAlignedFileReader());
#endif
        flash_indices.push_back(std::unique_ptr<diskann::PQFlashIndex<T>>(
            new diskann::PQFlashIndex<T>(readers[shard], use_page_search, metric, use_sq)));

        std::string disk_file_path = index_path + disk_file_name;
        diskann::cout << "Loading shard " << shard << ": " << disk_file_path << std::endl;
        
        int res = flash_indices[shard]->load(num_threads, index_path.c_str(), disk_file_path, pq_type);
        if (res != 0) return res;

        // Load mem_index if specified
        if (mem_L) {
            std::string mem_path = "";
            for (const auto& entry : fs::directory_iterator(index_path)) {
                if (!fs::is_directory(entry.path())) continue;
                if (entry.path().filename().string().find("MEM_R_") != std::string::npos) {
                    mem_path = entry.path().string();
                    break;
                }
            }
            if (!mem_path.empty()) {
                flash_indices[shard]->load_mem_index(metric, query_dim, 
                    mem_path + "/_index", num_threads, mem_L);
            }
        }

        // Cache nodes if specified
        if (num_nodes_to_cache > 0) {
            std::string warmup_query_file = index_path + "_sample_data.bin";
            std::vector<uint32_t> node_list;
            flash_indices[shard]->generate_cache_list_from_sample_queries(
                warmup_query_file, 15, 6, num_nodes_to_cache, num_threads, 
                node_list, use_page_search, mem_L);
            flash_indices[shard]->load_cache_list(node_list);
        }
    }

    // Load centroids for shard selection (knn/spann split types)
    T* centroids = nullptr;
    size_t num_centroids, centroid_dim, centroid_aligned_dim;
    bool is_knn_split = split_type.find("knn") != std::string::npos || 
                        split_type.find("kmeans") != std::string::npos;
    bool is_spann_split = split_type.find("spann") != std::string::npos;
    
    if (is_knn_split || is_spann_split) {
        std::string centroids_path = data_path + "/_centroids.bin";
        if (!std::is_same_v<T, float>) {
            float* centroids_float = nullptr;
            diskann::load_aligned_bin<float>(centroids_path, centroids_float, 
                num_centroids, centroid_dim, centroid_aligned_dim);
            diskann::alloc_aligned((void**)&centroids, 
                num_centroids * centroid_aligned_dim * sizeof(T), 8 * sizeof(T));
            diskann::convert_types<float, T>(centroids_float, centroids, 
                num_centroids, centroid_aligned_dim);
            diskann::aligned_free(centroids_float);
        } else {
            diskann::load_aligned_bin<T>(centroids_path, centroids, 
                num_centroids, centroid_dim, centroid_aligned_dim);
        }
    }

    std::shared_ptr<diskann::Distance<T>> dist_cmp;
    dist_cmp.reset(diskann::get_distance_function<T>(metric));

    omp_set_num_threads(num_threads);

    // Result storage
    std::vector<uint32_t> query_result_ids(recall_at * query_num);
    std::vector<float> query_result_dists(recall_at * query_num);
    std::vector<size_t> query_result_shards(recall_at * query_num);
    
    // Statistics
    std::vector<uint32_t> query_iterations(query_num);
    std::vector<uint64_t> query_final_L_sum(query_num);
    std::vector<uint64_t> query_total_ios(query_num);

    auto stats = new diskann::QueryStats[query_num];
    diskann::Timer query_timer;

    diskann::cout << "\nStarting adaptive multi-shard search with incremental API..." << std::endl;

#pragma omp parallel for schedule(dynamic, 1)
    for (_s64 i = 0; i < (int64_t)query_num; i++) {
        auto qs = std::chrono::high_resolution_clock::now();
        T* query_vec = query + (i * query_aligned_dim);

        // Step 1: Select shards to search
        std::vector<size_t> selected_indices;
        if (is_knn_split || is_spann_split) {
            std::vector<std::pair<float, size_t>> distances;
            for (size_t idx = 0; idx < num_shards; ++idx) {
                float dist = dist_cmp->compare(query_vec, 
                    centroids + idx * query_aligned_dim, query_aligned_dim);
                distances.emplace_back(dist, idx);
            }
            std::sort(distances.begin(), distances.end());
            for (size_t k = 0; k < split_K && k < distances.size(); ++k) {
                selected_indices.push_back(distances[k].second);
            }
        } else {
            // natural split: search all shards
            for (size_t k = 0; k < num_shards; ++k) {
                selected_indices.push_back(k);
            }
        }

        // Step 2: Create MultiShardSearchContext with selected shards
        diskann::MultiShardSearchContext<T> context(adaptive_params);
        for (auto idx : selected_indices) {
            context.add_shard(idx, flash_indices[idx].get());
        }

        // Step 3: Execute incremental multi-shard search
        std::vector<uint64_t> result_ids(recall_at);
        std::vector<float> result_dists(recall_at);
        std::vector<size_t> result_shards(recall_at);

        bool debug_this_query = debug && (i == 0);
        context.search(query_vec, recall_at, mem_L, beamwidth, 
                       search_io_limit, use_ratio,
                       result_ids.data(), result_dists.data(), 
                       result_shards.data(),
                       original_ids, debug_this_query);

        // Step 4: Convert results to global IDs and store
        for (size_t r = 0; r < recall_at; ++r) {
            query_result_ids[i * recall_at + r] = result_ids[r];
            query_result_dists[i * recall_at + r] = result_dists[r];
            query_result_shards[i * recall_at + r] = result_shards[r];
        }

        query_iterations[i] = context.get_iteration_count();
        query_final_L_sum[i] = context.get_global_L_sum();
        query_total_ios[i] = context.get_total_ios();

        auto qe = std::chrono::high_resolution_clock::now();
        stats[i].total_us = std::chrono::duration_cast<std::chrono::microseconds>(qe - qs).count();
    }

    double total_time = query_timer.elapsed() / 1000000.0;
    double qps = query_num / total_time;

    // Calculate statistics
    double mean_iterations = 0, mean_L_sum = 0, mean_ios = 0;
    for (size_t i = 0; i < query_num; i++) {
        mean_iterations += query_iterations[i];
        mean_L_sum += query_final_L_sum[i];
        mean_ios += query_total_ios[i];
    }
    mean_iterations /= query_num;
    mean_L_sum /= query_num;
    mean_ios /= query_num;

    auto mean_latency = diskann::get_mean_stats<float>(
        stats, query_num,
        [](const diskann::QueryStats& s) { return s.total_us.load(); });

    auto latency_999 = diskann::get_percentile_stats<float>(
        stats, query_num, 0.999,
        [](const diskann::QueryStats& s) { return s.total_us.load(); });

    auto mean_cpus = diskann::get_mean_stats<float>(
        stats, query_num,
        [](const diskann::QueryStats& s) { return s.cpu_us.load(); });

    // Calculate recall
    float recall = 0;
    if (calc_recall_flag) {
        recall = diskann::calculate_recall(query_num, gt_ids, gt_dists, gt_dim,
                                           query_result_ids.data(), recall_at, recall_at);
    }

    // Print shard distribution in top-k results
    std::vector<size_t> shard_contribution(num_shards, 0);
    for (size_t i = 0; i < query_num * recall_at; i++) {
        shard_contribution[query_result_shards[i]]++;
    }

    // Print detailed results
    diskann::cout << "\nAdaptive Multi-Shard Search Results (Incremental):" << std::endl;
    diskann::cout << "  - Total queries: " << query_num << std::endl;
    diskann::cout << "  - Total time: " << total_time << " seconds" << std::endl;
    diskann::cout << "  - QPS: " << qps << std::endl;
    diskann::cout << "  - Mean latency: " << mean_latency << " us" << std::endl;
    diskann::cout << "  - 99.9% latency: " << latency_999 << " us" << std::endl;
    diskann::cout << "  - Mean IOs: " << mean_ios << std::endl;
    diskann::cout << "  - Mean iterations: " << mean_iterations << std::endl;
    diskann::cout << "  - Mean final L sum: " << mean_L_sum << std::endl;
    if (calc_recall_flag) {
        diskann::cout << "  - Recall@" << recall_at << ": " << recall << std::endl;
    }

    diskann::cout << "\nShard contribution to results:" << std::endl;
    for (size_t s = 0; s < num_shards; s++) {
        double percentage = 100.0 * shard_contribution[s] / (query_num * recall_at);
        diskann::cout << "  Shard " << s << ": " << shard_contribution[s] 
                      << " (" << percentage << "%)" << std::endl;
    }

    // Standard formatted output (compatible with run_benchmark.sh / quick_run.py parser)
    diskann::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
    diskann::cout.precision(2);

    std::string recall_string = "Recall@" + std::to_string(recall_at);
    diskann::cout << std::setw(6) << "L" << std::setw(12)
                  << std::setw(16) << "QPS" << std::setw(16) << "Mean Latency"
                  << std::setw(16) << "99.9 Latency" << std::setw(16)
                  << "Mean IOs" << std::setw(16) << "CPU (s)"
                  << std::setw(15) << "Peak Mem(MB)";
    if (calc_recall_flag) {
        diskann::cout << std::setw(16) << recall_string << std::endl;
    } else {
        diskann::cout << std::endl;
    }

    diskann::cout << std::setw(6) << adaptive_params.L_end << std::setw(12)
                  << std::setw(16) << (float)qps << std::setw(16) << (float)mean_latency
                  << std::setw(16) << (float)latency_999 << std::setw(16) << (float)mean_ios
                  << std::setw(16) << (float)mean_cpus
                  << std::setw(15) << getProcessPeakRSS();
    if (calc_recall_flag) {
        diskann::cout << std::setw(16) << recall << std::endl;
    } else {
        diskann::cout << std::endl;
    }

    // Save results
    std::string result_output_prefix = parent_index_path + "/result_adaptive_" + 
        split_type + "_" + std::to_string(num_shards) + "/";
    fs::create_directories(result_output_prefix);

    std::string cur_result_path = result_output_prefix + "_idx_uint32.bin";
    diskann::save_bin<_u32>(cur_result_path, query_result_ids.data(), query_num, recall_at);

    cur_result_path = result_output_prefix + "_dists_float.bin";
    diskann::save_bin<float>(cur_result_path, query_result_dists.data(), query_num, recall_at);

    diskann::cout << "Results saved to: " << result_output_prefix << std::endl;

    // Cleanup
    delete[] stats;
    diskann::aligned_free(query);
    if (centroids != nullptr) diskann::aligned_free(centroids);

    return 0;
}

int main(int argc, char** argv) {
    std::string data_type, dist_fn, parent_index_path, data_path,
        query_file, gt_file, split_type, disk_file_name, quantification_type;
    unsigned num_threads, K, W, num_nodes_to_cache, search_io_limit, split_K;
    unsigned mem_L;
    uint64_t L_capacity, L_init, L_big, L_small, L_end;
    bool use_reorder_data = false;
    bool use_page_search = true;
    float use_ratio = 1.0;
    bool use_sq = false;
    bool debug = false;

    po::options_description desc{"Arguments"};
    try {
        desc.add_options()("help,h", "Print help message");
        
        // Required parameters
        desc.add_options()("data_type", po::value<std::string>(&data_type)->required(),
                           "Data type <int8/uint8/float>");
        desc.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                           "Distance function <l2/mips/cosine>");
        desc.add_options()("parent_index_path", 
                           po::value<std::string>(&parent_index_path)->required(),
                           "Parent path containing all index shards");
        desc.add_options()("data_path", po::value<std::string>(&data_path)->required(),
                           "Data path containing shard ID files");
        desc.add_options()("query_file", po::value<std::string>(&query_file)->required(),
                           "Query file in binary format");
        desc.add_options()("recall_at,K", po::value<uint32_t>(&K)->required(),
                           "Number of neighbors to return");
        desc.add_options()("split_type", po::value<std::string>(&split_type)->required(),
                           "Split type <natural/knn/spann>");
        desc.add_options()("disk_file_name", po::value<std::string>(&disk_file_name)->required(),
                           "Disk index file name (e.g., _disk.index)");

        // Adaptive search parameters
        desc.add_options()("L_capacity", po::value<uint64_t>(&L_capacity)->default_value(0),
                           "Fixed retset capacity per shard (0=auto from L_end/num_shards). Default: 0");
        desc.add_options()("L_init", po::value<uint64_t>(&L_init)->default_value(20),
                           "Initial search stopping point for all shards. Default: 20");
        desc.add_options()("L_big", po::value<uint64_t>(&L_big)->default_value(20),
                           "L_stop increment for contributing shards. Default: 20");
        desc.add_options()("L_small", po::value<uint64_t>(&L_small)->default_value(10),
                           "L_stop increment for non-contributing shards. Default: 10");
        desc.add_options()("L_end", po::value<uint64_t>(&L_end)->default_value(0),
                           "Stop when sum(L_stop) >= L_end. Default: num_shards * 100");

        // Optional parameters
        desc.add_options()("gt_file", 
                           po::value<std::string>(&gt_file)->default_value("null"),
                           "Ground truth file");
        desc.add_options()("beamwidth,W", po::value<uint32_t>(&W)->default_value(2),
                           "Beamwidth for search. Default: 2");
        desc.add_options()("num_nodes_to_cache", 
                           po::value<uint32_t>(&num_nodes_to_cache)->default_value(0),
                           "Number of nodes to cache. Default: 0");
        desc.add_options()("search_io_limit",
                           po::value<uint32_t>(&search_io_limit)
                               ->default_value(std::numeric_limits<_u32>::max()),
                           "Max IOs for search");
        desc.add_options()("num_threads,T",
                           po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                           "Number of threads");
        desc.add_options()("mem_L", po::value<unsigned>(&mem_L)->default_value(0),
                           "L for in-memory navigation graph. Default: 0 (disabled)");
        desc.add_options()("split_K", po::value<uint32_t>(&split_K)->default_value(10),
                           "Max shards to search per query. Default: 10");
        desc.add_options()("quantification_type,Q",
                           po::value<std::string>(&quantification_type)->default_value("PQ"),
                           "Quantification type: PQ/OPQ/LSQ/RABITQ");
        desc.add_options()("use_page_search", 
                           po::value<bool>(&use_page_search)->default_value(true),
                           "Use page search (1) or beam search (0). Default: 1");
        desc.add_options()("use_ratio", 
                           po::value<float>(&use_ratio)->default_value(1.0f),
                           "Page search use ratio. Default: 1.0");
        desc.add_options()("use_sq", po::value<bool>(&use_sq)->default_value(false),
                           "Use SQ-compressed disk vectors. Default: false");
        desc.add_options()("debug", po::value<bool>(&debug)->default_value(false),
                           "Enable debug output for first query. Default: false");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return -1;
    }

    diskann::Metric metric;
    if (dist_fn == "mips") {
        metric = diskann::Metric::INNER_PRODUCT;
    } else if (dist_fn == "l2") {
        metric = diskann::Metric::L2;
    } else if (dist_fn == "cosine") {
        metric = diskann::Metric::COSINE;
    } else {
        std::cout << "Unsupported distance function: " << dist_fn << std::endl;
        return -1;
    }

    AdaptiveSearchParams adaptive_params(L_capacity, L_init, L_big, L_small, L_end);

    try {
        if (data_type == "float") {
            return search_disk_index_adaptive_multi_shard<float>(
                metric, parent_index_path, data_path, query_file, gt_file,
                num_threads, K, W, num_nodes_to_cache, search_io_limit, mem_L,
                split_type, split_K, disk_file_name, quantification_type,
                adaptive_params, use_page_search, use_ratio, use_reorder_data, 
                use_sq, debug);
        } else if (data_type == "int8") {
            return search_disk_index_adaptive_multi_shard<int8_t>(
                metric, parent_index_path, data_path, query_file, gt_file,
                num_threads, K, W, num_nodes_to_cache, search_io_limit, mem_L,
                split_type, split_K, disk_file_name, quantification_type,
                adaptive_params, use_page_search, use_ratio, use_reorder_data, 
                use_sq, debug);
        } else if (data_type == "uint8") {
            return search_disk_index_adaptive_multi_shard<uint8_t>(
                metric, parent_index_path, data_path, query_file, gt_file,
                num_threads, K, W, num_nodes_to_cache, search_io_limit, mem_L,
                split_type, split_K, disk_file_name, quantification_type,
                adaptive_params, use_page_search, use_ratio, use_reorder_data, 
                use_sq, debug);
        } else {
            std::cerr << "Unsupported data type: " << data_type << std::endl;
            return -1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Search failed: " << e.what() << std::endl;
        return -1;
    }
}
