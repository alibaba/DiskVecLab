// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <atomic>
#include <cstring>
#include <iomanip>
#include <mutex>
#include <omp.h>
#include <pq_flash_index.h>
#include <set>
#include <string.h>
#include <time.h>
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

#include "elpis/Index.h"
#include "elpis/QueryEngine.h"

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

#define WARMUP false

namespace po = boost::program_options;

void print_stats(std::string category, std::vector<float> percentiles,
                 std::vector<float> results) {
	diskann::cout << std::setw(20) << category << ": " << std::flush;
	for (uint32_t s = 0; s < percentiles.size(); s++) {
		diskann::cout << std::setw(8) << percentiles[s] << "%";
	}
	diskann::cout << std::endl;
	diskann::cout << std::setw(22) << " " << std::flush;
	for (uint32_t s = 0; s < percentiles.size(); s++) {
		diskann::cout << std::setw(9) << results[s];
	}
	diskann::cout << std::endl;
}

template<typename T>
int search_disk_index_multi_split(
    diskann::Metric& metric, const std::string& parent_index_path, const std::string& data_path,
    const std::string& query_file,
    const std::string& gt_file,
    const unsigned num_threads, const unsigned recall_at,
    const unsigned beamwidth, const unsigned num_nodes_to_cache,
    const _u32 search_io_limit, const std::vector<unsigned>& Lvec,
    const _u32 mem_L,
	const std::string& split_type,
    const _u32 split_K,
    const std::string& disk_file_name,
    const std::string& quantification_type,
    const std::string& elpis_index_path,
    const int elpis_ef = 10,
    const unsigned int elpis_nworker = 0,
    const bool elpis_flatt = 0,
    const unsigned int elpis_nprobes = 0,
    const bool elpis_parallel = 0,
    const bool use_page_search=true,
    const float use_ratio=1.0,
    const bool use_reorder_data = false,
    const bool use_sq = false,
    const bool use_adaptive_search = false,
	const std::string& adaptive_shard_scheduler = "contrib",
    const uint64_t L_capacity = 0,
    const uint64_t L_init = 20,
    const uint64_t L_big = 20,
    const uint64_t L_small = 10,
    const uint64_t L_end = 100,
    const bool debug = false) {
	diskann::cout << "Search parameters: #threads: " << num_threads << ", ";
    diskann::cout << "Search parameters: #quantification_type: " << quantification_type << ", ";
    PQType pq_type;
    if (quantification_type == "PQ")
    {
      pq_type = PQType::PQ;
    } else if (quantification_type == "OPQ")
    {
      pq_type = PQType::OPQ;
    } else if (quantification_type == "LSQ")
    {
      pq_type = PQType::LSQ;
    } else if (quantification_type == "RABITQ")
    {
      pq_type = PQType::RABITQ;
    } else {
      diskann::cout << "not valid quantification_type: " << quantification_type << std::flush;
      return -1;
    }

	if (beamwidth <= 0)
		diskann::cout << "beamwidth to be optimized for each L value" << std::flush;
	else
		diskann::cout << " beamwidth: " << beamwidth << std::flush;
	if (search_io_limit == std::numeric_limits<_u32>::max())
		diskann::cout << "." << std::endl;
	else
		diskann::cout << ", io_limit: " << search_io_limit << "." << std::endl;

	if(use_sq && !std::is_same<T, float>::value) {
		std::cout << "erro, only support float sq" << std::endl;
		exit(-1);
	}

	// load query bin
	T*        query = nullptr;
	unsigned* gt_ids = nullptr;
	float*    gt_dists = nullptr;
	size_t    query_num, query_dim, query_aligned_dim, gt_num, gt_dim;
	diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim,
	                             query_aligned_dim);

	bool calc_recall_flag = false;
	if (gt_file != std::string("null") && gt_file != std::string("NULL") &&
	      file_exists(gt_file)) {
		diskann::load_truthset(gt_file, gt_ids, gt_dists, gt_num, gt_dim);
		if (gt_num != query_num) {
			diskann::cout
			          << "Error. Mismatch in number of queries and ground truth data"
			          << std::endl;
			exit(-1);
		}
		calc_recall_flag = true;
	}

	std::cout << "loading indices paths: " << parent_index_path << std::endl;
	// load indices paths
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
			throw std::invalid_argument("Invalid shard path format: missing 'shard_'");
		}

		size_t start_pos = shard_pos + 6;
		size_t end_pos = path.find('_', start_pos);
		if (end_pos == std::string::npos) {
			throw std::invalid_argument("Invalid shard path format: missing 'shard_'");
		}

		std::string id_str = path.substr(start_pos, end_pos - start_pos);
		return std::stoi(id_str);
	};
	// sort
	std::sort(shard_paths.begin(), shard_paths.end(), [&get_id](const std::string& a, const std::string& b) {
		return get_id(a) < get_id(b);
	});

	size_t num_shards = shard_paths.size();

	std::cout << "loading data shard : " << data_path << ", total " << num_shards << std::endl;
	// load shard id paths
	std::vector<std::string> shard_id_paths;
	for (const auto& entry : fs::directory_iterator(data_path)) {
		if (fs::is_directory(entry.path())) continue;

		std::string filename = entry.path().filename().string();
		if (filename.find("_subshard-") != std::string::npos && filename.find("_ids_uint32") != std::string::npos) {
			shard_id_paths.push_back(entry.path().string());
		}
	}
	// Sort shards by numeric ID in directory name
	std::sort(shard_id_paths.begin(), shard_id_paths.end(), [](const std::string& a, const std::string& b) {
		auto get_id = [](const std::string& path) {
			size_t start_pos = path.find("_subshard-") + 10;
			if (start_pos == std::string::npos)
				throw std::invalid_argument("Invalid shard path format: missing 'shard_'");

			size_t end_pos = path.find_first_not_of("0123456789", start_pos);
			std::string id_str = path.substr(start_pos, end_pos - start_pos);
			return std::stoi(id_str);
		};
		return get_id(a) < get_id(b);
	});

	if (num_shards != shard_id_paths.size()) {
		diskann::cout << "Error. Mismatch in number of id and data shards" << std::endl;
		exit(-1);
	}

    std::unordered_map<unsigned int, unsigned int> leaf_id_to_shard_index;
    Index * elpis_index = nullptr;
    if (split_type.find("elpis") != std::string::npos) {
	    for (_u32 shard = 0; shard < num_shards; shard++) {
            std::string index_path = shard_paths[shard] + "/";
            unsigned int id = get_id(index_path);


           if (leaf_id_to_shard_index.find(id) != leaf_id_to_shard_index.end()) {
               std::cerr << "Warning: Duplicate ID " << id << " found for shard " << shard
                       << ", previous shard was " << leaf_id_to_shard_index[id] << std::endl;
           }
           leaf_id_to_shard_index[id] = shard;
        }
        elpis_index = Index::Read(const_cast<char*>(elpis_index_path.c_str()),1,"");
        if (leaf_id_to_shard_index.size() != num_shards) {
            std::cerr << "Error: leaf_id_to_shard_index size mismatch! Expected: "
                 << num_shards << ", Actual: " << leaf_id_to_shard_index.size() << std::endl;
            exit(-1);
        }
    }

	std::vector<std::vector<uint32_t>> original_ids(num_shards);
	for (_u32 shard = 0; shard < num_shards; shard++) {
		std::string id_path = shard_id_paths[shard];
		uint32_t* ids = nullptr;
		size_t npts, nd;
		diskann::load_bin<uint32_t>(id_path, ids, npts, nd);

		assert(nd == 1 && "ID file must have dimension 1");

		original_ids[shard] = std::vector<uint32_t>(ids, ids + npts);

		delete[] ids;
	}


	uint64_t warmup_L = 20;
	uint64_t warmup_num = 0, warmup_dim = 0, warmup_aligned_dim = 0;
	T*       warmup = nullptr;

	std::vector<std::shared_ptr<AlignedFileReader>> readers(num_shards);
	std::vector<std::unique_ptr<diskann::PQFlashIndex<T>>> flash_indices;
	std::vector<std::vector<uint32_t>> optimized_beamwidth(num_shards, std::vector<uint32_t>(Lvec.size()));

	for (_u32 shard = 0; shard < num_shards; shard++) {
		std::string index_path = shard_paths[shard] + "/";

#ifdef _WINDOWS
#ifndef USE_BING_INFRA
		    readers[shard].reset(new WindowsAlignedFileReader());
#else
		    readers[shard].reset(new diskann::BingAlignedFileReader());
#endif
#else
		    readers[shard].reset(new LinuxAlignedFileReader());
#endif

	    // Initialize and load index
		flash_indices.push_back(std::unique_ptr<diskann::PQFlashIndex<T>>(
		        new diskann::PQFlashIndex<T>(readers[shard], use_page_search, metric, use_sq)));

		std::string disk_file_path = index_path + disk_file_name;
		diskann::cout << "reading " << disk_file_path << std::endl;
		int res = flash_indices[shard]->load(num_threads, index_path.c_str(), disk_file_path, pq_type);
		if (res != 0) {
			return res;
		}

		std::string mem_path = "";
		for (const auto& entry : fs::directory_iterator(index_path)) {
			if (!fs::is_directory(entry.path())) continue;

			std::string filename = entry.path().filename().string();
			if (filename.find("MEM_R_") != std::string::npos) {
				mem_path = entry.path().string();
			}
		}

		// Load mem_index if specified
		if (mem_L) {
			std::string mem_index_path = mem_path + "/_index";
			flash_indices[shard]->load_mem_index(metric, query_dim, mem_index_path, num_threads, mem_L);
		}

		std::string warmup_query_file = index_path + "_sample_data.bin";

		// cache bfs levels
		std::vector<uint32_t> node_list;
		diskann::cout << "Caching " << num_nodes_to_cache
		                  << " BFS nodes around medoid(s)" << std::endl;
		//_pFlashIndex->cache_bfs_levels(num_nodes_to_cache, node_list);
		if (num_nodes_to_cache > 0) {
			if(use_sq) {
				std::cout << "not support sq cache, please use mem index" << std::endl;
				exit(-1);
			}
			flash_indices[shard]->generate_cache_list_from_sample_queries(
			          warmup_query_file, 15, 6, num_nodes_to_cache, num_threads, node_list, use_page_search, mem_L);
			flash_indices[shard]->load_cache_list(node_list);
		}

		node_list.clear();
		node_list.shrink_to_fit();

		omp_set_nested(1);
		omp_set_max_active_levels(2);
		omp_set_num_threads(num_threads);

		if (WARMUP) {
			if (file_exists(warmup_query_file)) {
				diskann::load_aligned_bin<T>(warmup_query_file, warmup, warmup_num,
				                                     warmup_dim, warmup_aligned_dim);
			} else {
				warmup_num = (std::min)((_u32) 150000, (_u32) 15000 * num_threads);
				warmup_dim = query_dim;
				warmup_aligned_dim = query_aligned_dim;
				diskann::alloc_aligned(((void**) &warmup),
				                               warmup_num * warmup_aligned_dim * sizeof(T),
				                               8 * sizeof(T));
				std::memset(warmup, 0, warmup_num * warmup_aligned_dim * sizeof(T));
				std::random_device              rd;
				std::mt19937                    gen(rd());
				std::uniform_int_distribution<> dis(-128, 127);
				for (uint32_t i = 0; i < warmup_num; i++) {
					for (uint32_t d = 0; d < warmup_dim; d++) {
						warmup[i * warmup_aligned_dim + d] = (T) dis(gen);
					}
				}
			}
			diskann::cout << "Warming up index... " << std::flush;
			std::vector<uint64_t> warmup_result_ids_64(warmup_num, 0);
			std::vector<float>    warmup_result_dists(warmup_num, 0);

#pragma omp parallel for schedule(dynamic, 1)
			for (_s64 i = 0; i < (int64_t) warmup_num; i++) {
				flash_indices[shard]->cached_beam_search(warmup + (i * warmup_aligned_dim), 1,
				                                         warmup_L,
				                                         warmup_result_ids_64.data() + (i * 1),
				                                         warmup_result_dists.data() + (i * 1), 4);
			}
			diskann::cout << "..done" << std::endl;
		}
		for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++) {
			_u64 L = Lvec[test_id];
			if (beamwidth <= 0) {
				diskann::cout << "Tuning beamwidth for shard " << shard << std::endl;
				optimized_beamwidth[shard][test_id] = optimize_beamwidth(flash_indices[shard], warmup, warmup_num,
																         warmup_aligned_dim, L, 2);
			} else {
				optimized_beamwidth[shard][test_id] = beamwidth;
			}
		}
		diskann::cout << "Loaded index " << index_path << shard << std::endl;
	}

	T* centroids = nullptr;
	size_t num_centroids, centroid_dim, centroid_aligned_dim;
	// load centroids for pruning
	bool is_knn_split = split_type.find("knn") != std::string::npos || split_type.find("kmeans") != std::string::npos;
	bool is_spann_split = split_type.find("spann") != std::string::npos;
	if (is_knn_split || is_spann_split) {
		std::string centroids_path = data_path + "/_centroids.bin";
		if (is_knn_split && !std::is_same_v<T, float>) {
			float* centroids_float = nullptr;
            diskann::load_aligned_bin<float>(centroids_path, centroids_float, num_centroids, centroid_dim, centroid_aligned_dim);
			// Allocate memory for centroids before conversion
			diskann::alloc_aligned((void**)&centroids, num_centroids * centroid_aligned_dim * sizeof(T), 8 * sizeof(T));
			diskann::convert_types<float, T>(centroids_float, centroids, num_centroids, centroid_aligned_dim);
			// Free the temporary float array
			diskann::aligned_free(centroids_float);
        } else {
            diskann::load_aligned_bin<T>(centroids_path, centroids, num_centroids, centroid_dim, centroid_aligned_dim);
        }
	}

	std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
	std::vector<std::vector<float>>    query_result_dists(Lvec.size());

	std::shared_ptr<diskann::Distance<T>>     dist_cmp;
	dist_cmp.reset(diskann::get_distance_function<T>(metric));

	// ============================================================================
	// Adaptive Multi-Shard Search Mode (using incremental search)
	// ============================================================================
	if (use_adaptive_search) {
		diskann::cout << "\n=== Adaptive Multi-Shard Search Mode ==="<< std::endl;
		diskann::cout << "  adaptive_shard_scheduler: " << adaptive_shard_scheduler << std::endl;
		diskann::cout << "  L_capacity: " << L_capacity << " (0=auto)" << std::endl;
		diskann::cout << "  L_init: " << L_init << std::endl;
		diskann::cout << "  L_big: " << L_big << std::endl;
		diskann::cout << "  L_small: " << L_small << std::endl;
		diskann::cout << "  L_end: " << L_end << std::endl;

		// Create adaptive search params
		diskann::MultiShardSearchParams adaptive_params(L_capacity, L_init, L_big, L_small, L_end);
		adaptive_params.adaptive_shard_scheduler = adaptive_shard_scheduler;

		// Result storage
		std::vector<uint32_t> adaptive_result_ids(recall_at * query_num);
		std::vector<float> adaptive_result_dists(recall_at * query_num);
		std::vector<size_t> adaptive_result_shards(recall_at * query_num);
		std::vector<uint32_t> query_iterations(query_num);
		std::vector<uint64_t> query_final_L_sum(query_num);
		std::vector<uint64_t> query_total_ios(query_num);

		auto stats = new diskann::QueryStats[query_num];
		auto s = std::chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(dynamic, 1)
		for (_s64 i = 0; i < (int64_t)query_num; i++) {
			auto qs = std::chrono::high_resolution_clock::now();
			T* query_vec = query + (i * query_aligned_dim);

			// Step 1: Select shards to search
			std::vector<size_t> selected_indices;
			if (is_knn_split || is_spann_split) {
				std::vector<std::pair<float, size_t>> distances;
				for (size_t idx = 0; idx < num_shards; ++idx) {
					float dist = dist_cmp->compare(query_vec, centroids + idx * query_aligned_dim, query_aligned_dim);
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

			// Step 4: Store results
			for (size_t r = 0; r < recall_at; ++r) {
				adaptive_result_ids[i * recall_at + r] = result_ids[r];
				adaptive_result_dists[i * recall_at + r] = result_dists[r];
				adaptive_result_shards[i * recall_at + r] = result_shards[r];
			}

			query_iterations[i] = context.get_iteration_count();
			query_final_L_sum[i] = context.get_global_L_sum();
			query_total_ios[i] = context.get_total_ios();

			auto qe = std::chrono::high_resolution_clock::now();
			stats[i].total_us = std::chrono::duration_cast<std::chrono::microseconds>(qe - qs).count();
		}

		auto e = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> diff = e - s;

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
			                                   adaptive_result_ids.data(), recall_at, recall_at);
		}

		// Print shard distribution in top-k results
		std::vector<size_t> shard_contribution(num_shards, 0);
		for (size_t i = 0; i < query_num * recall_at; i++) {
			shard_contribution[adaptive_result_shards[i]]++;
		}

		// Print detailed results
		double qps = query_num / diff.count();
		diskann::cout << "\nAdaptive Multi-Shard Search Results:" << std::endl;
		diskann::cout << "  L_capacity=" << L_capacity << " L_init=" << L_init << " L_big=" << L_big << " L_small=" << L_small << " L_end=" << L_end << std::endl;
		diskann::cout << "  - Total queries: " << query_num << std::endl;
		diskann::cout << "  - Total time: " << diff.count() << " seconds" << std::endl;
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

		// Standard formatted output (same format as non-adaptive search, compatible with parsers)
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

		diskann::cout << std::setw(6) << L_end << std::setw(12)
		              << std::setw(16) << (float)qps << std::setw(16) << (float)mean_latency
		              << std::setw(16) << (float)latency_999 << std::setw(16) << (float)mean_ios
		              << std::setw(16) << (float)mean_cpus
		              << std::setw(15) << getProcessPeakRSS();
		if (calc_recall_flag) {
			diskann::cout << std::setw(16) << recall << std::endl;
		} else {
			diskann::cout << std::endl;
		}

		delete[] stats;

		// Cleanup and return
		diskann::aligned_free(query);
		if (centroids != nullptr) diskann::aligned_free(centroids);

		return (recall >= 0) ? 0 : -1;
	}

	// ============================================================================
	// Original Multi-Shard Search Mode (L-value sweep)
	// ============================================================================

	for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++) {
		// diskann::cout << "Lvec: " << Lvec[test_id] << std::endl;
		query_result_ids[test_id].resize(recall_at * query_num);
		query_result_dists[test_id].resize(recall_at * query_num);
		std::vector<uint64_t> query_result_ids_64(recall_at * query_num);
		_u64 L = Lvec[test_id];

		if (L < recall_at) {
			diskann::cout << "Ignoring search with L:" << L
			                    << " since it's smaller than K:" << recall_at << std::endl;
			continue;
		}

		auto                  s = std::chrono::high_resolution_clock::now();
		auto stats = new diskann::QueryStats[query_num];
#pragma omp parallel for schedule(dynamic, 1)
		for (_s64 i = 0; i < (int64_t) query_num; i++) {
			auto qs = std::chrono::high_resolution_clock::now();
			// diskann::cout << "search query " <<  i << "" << std::endl;
			//diskann::cout << "search query " <<  i << "" << std::endl;
			T* query_vec = query + (i * query_aligned_dim);

            QueryEngine * elpis_queryengine = nullptr;
		    std::vector<std::vector<uint64_t>> partial_ids;
		    std::vector<std::vector<float>> partial_dists;
			std::vector<size_t> selected_indices;
			if (is_knn_split || is_spann_split) {
				if (centroids == nullptr) {
					std::cerr << "Error: Centroids not loaded but required for split_type=" << split_type << std::endl;
					exit(-1);
				}
				std::vector<std::pair<float, size_t>> distances;
				for (size_t idx = 0; idx < num_shards; ++idx) {
					float dist = dist_cmp->compare(query_vec, centroids + idx * query_aligned_dim, query_aligned_dim);
					distances.emplace_back(dist, idx);
				}
				std::sort(distances.begin(), distances.end());
				for (size_t k = 0; k < split_K && k < distances.size(); ++k) {
					selected_indices.push_back(distances[k].second);
				}
                partial_ids.resize(selected_indices.size());
                partial_dists.resize(selected_indices.size());
			} else if (split_type.find("natural") != std::string::npos) {
				for (size_t k = 0; k < num_shards; ++k) {
					selected_indices.push_back(k);
				}
                partial_ids.resize(selected_indices.size());
                partial_dists.resize(selected_indices.size());
			} else if (split_type.find("elpis") != std::string::npos) {
                elpis_queryengine = new QueryEngine("", elpis_index, elpis_ef, split_K, elpis_parallel, elpis_nworker, elpis_flatt, recall_at);
                // Convert query vector to float if needed
                float* float_query_vec = nullptr;
                if constexpr (std::is_same_v<T, float>) {
                    float_query_vec = query_vec;
                } else {
                    float_query_vec = new float[query_aligned_dim];
                	diskann::convert_types<T, float>(query_vec, float_query_vec, 1, query_aligned_dim);
                }

                unsigned int first_leaf_id = elpis_queryengine->get_first_leaf(float_query_vec);

                selected_indices.push_back(leaf_id_to_shard_index[first_leaf_id]);
                // search first leaf
                size_t idx = selected_indices[0];
                std::vector<uint64_t> first_partial_ids(recall_at);
                std::vector<float> first_partial_dists(recall_at);
				if (use_page_search) {
					if(use_sq) {
						flash_indices[idx]->page_search_sq(
						                  query_vec, recall_at, mem_L, L,
						                  first_partial_ids.data(), first_partial_dists.data(),
						                  optimized_beamwidth[0][test_id], search_io_limit, use_reorder_data, use_ratio, stats + i);
					} else {
						flash_indices[idx]->page_search(
						                  query + (i * query_aligned_dim), recall_at, mem_L, L,
						                  first_partial_ids.data(), first_partial_dists.data(),
						                  optimized_beamwidth[0][test_id], search_io_limit, use_reorder_data, use_ratio, stats + i);
					}
				} else {
					if(use_sq) {
						std::cout << "diskann current not support sq..." << std::endl;
						exit(-1);
					}
					flash_indices[idx]->cached_beam_search(
					                query + (i * query_aligned_dim), recall_at, L,
					                first_partial_ids.data(), first_partial_dists.data(),
					                optimized_beamwidth[0][test_id], search_io_limit, use_reorder_data, stats + i, mem_L);
				}
                float first_leaf_max_dist = first_partial_dists[first_partial_dists.size() - 1];
                elpis_queryengine->get_other_leaves(float_query_vec, recall_at, first_leaf_max_dist);
                std::unordered_set<unsigned int> visited_leaves = elpis_queryengine->get_unique_visited_leaves();
                for (const auto& leaf_id : visited_leaves) {
                    selected_indices.push_back(leaf_id_to_shard_index[leaf_id]);
                }
                //for (const auto& pair : leaf_id_to_shard_index) {
                //    selected_indices.push_back(pair.second);
                //}
                partial_ids.resize(selected_indices.size());
                partial_dists.resize(selected_indices.size());
                partial_ids[0] = first_partial_ids;
                partial_dists[0] = first_partial_dists;

                // Clean up if we allocated memory
                if constexpr (!std::is_same_v<T, float>) {
                    delete[] float_query_vec;
                }
            }

            if (selected_indices.size() > optimized_beamwidth.size()) {
                std::cerr << "Error: selected_indices size mismatch! Expected: <= "
                     << optimized_beamwidth.size() << ", Actual: " << selected_indices.size() << std::endl;
                exit(-1);
            }

			//diskann::cout << "selected index: ";
			//for (auto idx : selected_indices) {
			//	diskann::cout << idx << " ";
			//}
			//diskann::cout << std::endl;


// #pragma omp parallel for schedule(dynamic, 1)
			for (size_t s_idx = 0; s_idx < selected_indices.size(); ++s_idx) {
                if (split_type.find("elpis") != std::string::npos && s_idx == 0) {
                    continue;
                }
				//diskann::cout << "search index " << selected_indices[s_idx] << std::endl;
				partial_ids[s_idx].resize(recall_at);
				partial_dists[s_idx].resize(recall_at);
				size_t idx = selected_indices[s_idx];

				// Using branching outside the for loop instead of inside and
				// std::function/std::mem_fn for less switching and function calling overhead
				if (use_page_search) {
					if(use_sq) {
						flash_indices[idx]->page_search_sq(
						                  query_vec, recall_at, mem_L, L,
						                  partial_ids[s_idx].data(), partial_dists[s_idx].data(),
						                  optimized_beamwidth[s_idx][test_id], search_io_limit, use_reorder_data, use_ratio, stats + i);
					} else {
						flash_indices[idx]->page_search(
						                  query + (i * query_aligned_dim), recall_at, mem_L, L,
						                  partial_ids[s_idx].data(), partial_dists[s_idx].data(),
						                  optimized_beamwidth[s_idx][test_id], search_io_limit, use_reorder_data, use_ratio, stats + i);
					}
				} else {
					if(use_sq) {
						std::cout << "diskann current not support sq..." << std::endl;
						exit(-1);
					}
					flash_indices[idx]->cached_beam_search(
					                query + (i * query_aligned_dim), recall_at, L,
					                partial_ids[s_idx].data(), partial_dists[s_idx].data(),
					                optimized_beamwidth[s_idx][test_id], search_io_limit, use_reorder_data, stats + i, mem_L);
				}
			}

			// get top K after merge
			std::unordered_set<size_t> unique_ids;
			std::vector<std::pair<float, uint64_t>> merged_results;
			for (size_t s_idx = 0; s_idx < selected_indices.size(); ++s_idx) {
				for (size_t r = 0; r < recall_at; ++r) {
					size_t shard_idx = selected_indices[s_idx];
					uint32_t local_id = partial_ids[s_idx][r];
					uint32_t global_id = original_ids[shard_idx][local_id];
					if (unique_ids.count(global_id) > 0) {
						continue; // Skip duplicates
					}
					unique_ids.insert(global_id);
					merged_results.emplace_back(partial_dists[s_idx][r], global_id);
				}
			}

			std::sort(merged_results.begin(), merged_results.end());
			for (size_t r = 0; r < recall_at && r < merged_results.size(); ++r) {
				query_result_ids_64[i * recall_at + r] = merged_results[r].second;
				query_result_dists[test_id][i * recall_at + r] = merged_results[r].first;
			}
			auto qe = std::chrono::high_resolution_clock::now();
			stats[i].total_us =
			        std::chrono::duration_cast<std::chrono::microseconds>(qe - qs).count();
        }

		auto  e = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> diff = e - s;
		float qps = (1.0 * query_num) / (1.0 * diff.count());

		diskann::convert_types<uint64_t, uint32_t>(query_result_ids_64.data(),
		                                           query_result_ids[test_id].data(),
		                                           query_num, recall_at);

		auto mean_latency = diskann::get_mean_stats<float>(
			stats, query_num,
			[](const diskann::QueryStats& stats) { return stats.total_us.load(); });

		auto latency_999 = diskann::get_percentile_stats<float>(
			stats, query_num, 0.999,
			[](const diskann::QueryStats& stats) { return stats.total_us.load(); });
		auto mean_ios = diskann::get_mean_stats<unsigned>(
			stats, query_num,
			[](const diskann::QueryStats& stats) { return stats.n_ios.load(); });

		auto mean_cpus = diskann::get_mean_stats<float>(
			stats, query_num,
			[](const diskann::QueryStats& stats) { return stats.cpu_us.load(); });
		float recall = 0;
		if (calc_recall_flag) {
			recall = diskann::calculate_recall(query_num, gt_ids, gt_dists, gt_dim,
			                                           query_result_ids[test_id].data(),
			                                           recall_at, recall_at);
		}

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
			diskann::cout
				<< "==============================================================="
				   "======================================================="
				<< std::endl;
		}

		diskann::cout << std::setw(6) << L << std::setw(12)
		                    << std::setw(16) << qps << std::setw(16) << mean_latency
		                    << std::setw(16) << latency_999 << std::setw(16) << mean_ios
		                    << std::setw(16) << mean_cpus
		                    << std::setw(15) << getProcessPeakRSS();
		if (calc_recall_flag) {
			diskann::cout << std::setw(16) << recall << std::endl;
		} else {
			diskann::cout << std::endl;
        }

		delete[] stats;
	}
	diskann::cout << "Done searching. Now saving results " << std::endl;
	_u64 test_id = 0;
	std::string result_output_prefix = parent_index_path + "/result_" + split_type + "_" + std::to_string(num_shards) + "/";
	fs::create_directories(result_output_prefix);
	for (auto L : Lvec) {
		if (L < recall_at)
		      continue;

		std::string cur_result_path =
		        result_output_prefix + "_" + std::to_string(L) + "_idx_uint32.bin";
		diskann::save_bin<_u32>(cur_result_path, query_result_ids[test_id].data(),
		                            query_num, recall_at);

		cur_result_path =
		        result_output_prefix + "_" + std::to_string(L) + "_dists_float.bin";
		diskann::save_bin<float>(cur_result_path,
		                             query_result_dists[test_id++].data(), query_num,
		                             recall_at);
	}

	diskann::aligned_free(query);
	if (warmup != nullptr)
	    diskann::aligned_free(warmup);
	if (centroids != nullptr)
		diskann::aligned_free(centroids);
	return 0;
}

int main(int argc, char** argv) {
	std::string data_type, dist_fn, parent_index_path, data_path,
	      query_file, gt_file, split_type, disk_file_name;
    std::string quantification_type;
	unsigned              num_threads, K, W, num_nodes_to_cache, search_io_limit, split_K;
	unsigned              mem_L;
	std::vector<unsigned> Lvec;
	bool                  use_reorder_data = false;
	bool                  use_page_search = true;
	float                 use_ratio = 1.0;
	bool use_sq = false;
    std::string elpis_index_path;
    int elpis_ef = 10;
    unsigned int elpis_nworker = 0;
    bool elpis_flatt = 0;
    unsigned int elpis_nprobes = 0;
    bool elpis_parallel = 0;
	bool use_adaptive_search = false;
	std::string adaptive_shard_scheduler;
	uint64_t L_capacity = 0, L_init = 20, L_big = 20, L_small = 10, L_end = 0;
	bool debug = false;

	po::options_description desc {"Arguments"};
	try {
		desc.add_options()("help,h", "Print information on arguments");
		desc.add_options()("data_type",
		                       po::value<std::string>(&data_type)->required(),
		                       "data type <int8/uint8/float>");
		desc.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
		                       "distance function <l2/mips/fast_l2>");
		desc.add_options()("parent_index_path",
		                       po::value<std::string>(&parent_index_path)->required(),
		                       "Parent path containing all index shards");
		desc.add_options()("data_path",
							   po::value<std::string>(&data_path)->required(),
							   "Data path containing all shards");
		desc.add_options()("query_file",
		                       po::value<std::string>(&query_file)->required(),
		                       "Query file in binary format");
		desc.add_options()(
		        "gt_file",
		        po::value<std::string>(&gt_file)->default_value(std::string("null")),
		        "ground truth file for the queryset");
		desc.add_options()("recall_at,K", po::value<uint32_t>(&K)->required(),
		                       "Number of neighbors to be returned");
		desc.add_options()("search_list,L",
		                       po::value<std::vector<unsigned>>(&Lvec)->multitoken(),
		                       "List of L values of search");
		desc.add_options()("beamwidth,W", po::value<uint32_t>(&W)->default_value(2),
		                       "Beamwidth for search. Set 0 to optimize internally.");
		desc.add_options()(
		        "num_nodes_to_cache",
		        po::value<uint32_t>(&num_nodes_to_cache)->default_value(0),
		        "Beamwidth for search");
		desc.add_options()("search_io_limit",
		                       po::value<uint32_t>(&search_io_limit)
		                           ->default_value(std::numeric_limits<_u32>::max()),
		                       "Max #IOs for search");
		desc.add_options()(
		        "num_threads,T",
		        po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
		        "Number of threads used for building index (defaults to "
		        "omp_get_num_procs())");
		desc.add_options()("use_reorder_data",
		                       po::bool_switch()->default_value(false),
		                       "Include full precision data in the index. Use only in "
		                       "conjuction with compressed data on SSD.");
		desc.add_options()("use_sq",
		                       po::value<bool>(&use_sq)->default_value(0),
		                       "Use SQ-compressed disk vector.");
		desc.add_options()("mem_L", po::value<unsigned>(&mem_L)->default_value(0),
		                       "The L of the in-memory navigation graph while searching. Use 0 to disable");
		desc.add_options()("use_page_search", po::value<bool>(&use_page_search)->default_value(1),
		                       "Use 1 for page search (default), 0 for DiskANN beam search");
		desc.add_options()("use_ratio", po::value<float>(&use_ratio)->default_value(1.0f),
		                       "The percentage of how many vectors in a page to search each time");
		desc.add_options()("split_type",
							   po::value<std::string>(&split_type)->required(),
							   "split type <natural/knn/spann/elpis>");
		desc.add_options()("split_K",
		                       po::value<uint32_t>(&split_K)->default_value(10),
		                       "Max number of shards to search per query");
		desc.add_options()("disk_file_name", po::value<std::string>(&disk_file_name)->required(),
						   "The name of the disk file (_disk.index in the original DiskANN)");
        desc.add_options()("quantification_type,Q", po::value<std::string>(&quantification_type)->default_value("PQ"),
                           "the quantification type used in search");
        desc.add_options()("elpis_index_path", po::value<std::string>(&elpis_index_path)->default_value(""),
                           "ELPIS index path");
        desc.add_options()("elpis_ef", po::value<int>(&elpis_ef)->default_value(10),
                           "ELPIS ef");
        desc.add_options()("elpis_nworker", po::value<unsigned int>(&elpis_nworker)->default_value(0),
                           "ELPIS nworker");
        desc.add_options()("elpis_nprobes", po::value<unsigned int>(&elpis_nprobes)->default_value(0),
                           "ELPIS nprobes");
        desc.add_options()("elpis_flatt", po::value<bool>(&elpis_flatt)->default_value(0),
                           "ELPIS flatt");
        desc.add_options()("elpis_parallel", po::value<bool>(&elpis_parallel)->default_value(0),
                           "ELPIS parallel");

		// Adaptive multi-shard search parameters
		desc.add_options()("use_adaptive_search",
		                   po::value<bool>(&use_adaptive_search)->default_value(false),
		                   "Enable adaptive L allocation across shards. Default: false");
		desc.add_options()("adaptive_shard_scheduler",
		                   po::value<std::string>(&adaptive_shard_scheduler)->default_value("global_topL_pages"),
		                   "Adaptive shard scheduler: contrib | radius_topL | global_topL_pages. Default: contrib");
		desc.add_options()("L_capacity",
		                   po::value<uint64_t>(&L_capacity)->default_value(0),
		                   "Fixed retset capacity per shard (0=auto from L_end/num_shards). Default: 0");
		desc.add_options()("L_init",
		                   po::value<uint64_t>(&L_init)->default_value(20),
		                   "Initial search stopping point for all shards. Default: 20");
		desc.add_options()("L_big",
		                   po::value<uint64_t>(&L_big)->default_value(20),
		                   "L_stop increment for contributing shards. Default: 20");
		desc.add_options()("L_small",
		                   po::value<uint64_t>(&L_small)->default_value(10),
		                   "L_stop increment for non-contributing shards. Default: 10");
		desc.add_options()("L_end",
		                   po::value<uint64_t>(&L_end)->default_value(0),
		                   "Termination: stop when sum(L_stop) >= L_end. 0=auto. Default: 0");
		desc.add_options()("debug",
		                   po::value<bool>(&debug)->default_value(false),
		                   "Enable debug output for first query. Default: false");

		po::variables_map vm;
		po::store(po::parse_command_line(argc, argv, desc), vm);
		if (vm.count("help")) {
			std::cout << desc;
			return 0;
		}
		po::notify(vm);
	}
	catch (const std::exception& ex) {
		std::cerr << ex.what() << '\n';
		return -1;
	}

	diskann::Metric metric;
	if (dist_fn == std::string("mips")) {
		metric = diskann::Metric::INNER_PRODUCT;
	} else if (dist_fn == std::string("l2")) {
		metric = diskann::Metric::L2;
	} else if (dist_fn == std::string("cosine")) {
		metric = diskann::Metric::COSINE;
	} else {
		std::cout << "Unsupported distance function. Currently only L2/ Inner "
		                 "Product/Cosine are supported."
		              << std::endl;
		return -1;
	}

	if (use_ratio < 0 || use_ratio > 1.0f) {
		std::cout << "use_ratio should be in the range [0, 1] (inclusive)." << std::endl;
		return -1;
	}

	if ((data_type != std::string("float")) &&
	      (metric == diskann::Metric::INNER_PRODUCT)) {
		std::cout << "Currently support only floating point data for Inner Product."
		              << std::endl;
		return -1;
	}
	if ((data_type != std::string("float")) &&
	      (use_sq)) {
		std::cout << "Currently support only float sq"
		              << std::endl;
		return -1;
	}

	if (use_reorder_data && data_type != std::string("float")) {
		std::cout << "Error: Reorder data for reordering currently only "
		                 "supported for float data type."
		              << std::endl;
		return -1;
	}
	if(!use_page_search && use_sq) {
		std::cout << "Currently not support diskann + sq" << std::endl;
		return -1;
	}
	try {
		if (data_type == std::string("float"))
		      return search_disk_index_multi_split<float>(metric, parent_index_path, data_path,
		                                      query_file, gt_file,
		                                      num_threads, K, W, num_nodes_to_cache,
		                                      search_io_limit, Lvec, mem_L, split_type, split_K, disk_file_name, quantification_type, elpis_index_path,
		                                      elpis_ef, elpis_nworker, elpis_flatt, elpis_nprobes, elpis_parallel,
		                                      use_page_search, use_ratio, use_reorder_data, use_sq,
		                                      use_adaptive_search, adaptive_shard_scheduler, L_capacity, L_init, L_big, L_small, L_end, debug);
		else if (data_type == std::string("int8"))
		      return search_disk_index_multi_split<int8_t>(metric, parent_index_path, data_path,
		                                       query_file, gt_file,
		                                       num_threads, K, W, num_nodes_to_cache,
		                                       search_io_limit, Lvec, mem_L, split_type, split_K, disk_file_name, quantification_type, elpis_index_path,
		                                       elpis_ef, elpis_nworker, elpis_flatt, elpis_nprobes, elpis_parallel,
		                                       use_page_search, use_ratio, use_reorder_data, false,
		                                       use_adaptive_search, adaptive_shard_scheduler, L_capacity, L_init, L_big, L_small, L_end, debug);
		else if (data_type == std::string("uint8"))
		      return search_disk_index_multi_split<uint8_t>(
		          metric, parent_index_path, data_path, query_file, gt_file,
		          num_threads, K, W, num_nodes_to_cache, search_io_limit, Lvec, mem_L, split_type, split_K, disk_file_name, quantification_type, elpis_index_path,
		          elpis_ef, elpis_nworker, elpis_flatt, elpis_nprobes, elpis_parallel,
		          use_page_search, use_ratio, use_reorder_data, false,
		          use_adaptive_search, adaptive_shard_scheduler, L_capacity, L_init, L_big, L_small, L_end, debug);
		else {
			std::cerr << "Unsupported data type. Use float or int8 or uint8"
			                << std::endl;
			return -1;
		}
	} catch (const std::exception& e) {
		std::cout << std::string(e.what()) << std::endl;
		diskann::cerr << "Index search failed." << std::endl;
		return -1;
	}
}