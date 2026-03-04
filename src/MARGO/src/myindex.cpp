
#include <iostream>
#include <omp.h>
#include <float.h>

#include "timer.h"
#include "myindex.h"


namespace diskann {

  template<typename T, typename TagT>
  myIndex<T, TagT>::myIndex(Metric m, const size_t dim, const size_t max_points,
                        const bool dynamic_index, const bool enable_tags,
                        const bool support_eager_delete,
                        const bool concurrent_consolidate)
                         :Index<T, TagT>(m, dim, max_points, dynamic_index, enable_tags, 
                                         support_eager_delete, concurrent_consolidate) {
    const size_t total_internal_points = this->_max_points + this->_num_frozen_pts;
    this->weights_.resize(total_internal_points);
    this->wws_.resize(total_internal_points, 0.0f);
  }

  template<typename T, typename TagT>
  void myIndex<T, TagT>::mybuild(const char *             filename,
                             const size_t             num_points_to_load,
                             Parameters &             parameters,
                             const std::vector<TagT> &tags) {
    std::cout << "running mybuild" << std::endl;
    if (num_points_to_load == 0)
      throw ANNException("Do not call build with 0 points", -1, __FUNCSIG__,
                         __FILE__, __LINE__);

    if (!file_exists(filename)) {
      diskann::cerr << "Data file " << filename
                    << " does not exist!!! Exiting...." << std::endl;
      std::stringstream stream;
      stream << "Data file " << filename << " does not exist." << std::endl;
      diskann::cerr << stream.str() << std::endl;
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }

    size_t file_num_points, file_dim;
    if (filename == nullptr) {
      throw diskann::ANNException("Can not build with an empty file", -1,
                                  __FUNCSIG__, __FILE__, __LINE__);
    }

    diskann::get_bin_metadata(filename, file_num_points, file_dim);
    if (file_num_points > this->_max_points) {
      std::stringstream stream;
      stream << "ERROR: Driver requests loading " << num_points_to_load
             << " points and file has " << file_num_points << " points, but "
             << "index can support only " << this->_max_points
             << " points as specified in constructor." << std::endl;
      aligned_free(this->_data);
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }

    if (num_points_to_load > file_num_points) {
      std::stringstream stream;
      stream << "ERROR: Driver requests loading " << num_points_to_load
             << " points and file has only " << file_num_points << " points."
             << std::endl;
      aligned_free(this->_data);
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }

    if (file_dim != this->_dim) {
      std::stringstream stream;
      stream << "ERROR: Driver requests loading " << this->_dim << " dimension,"
             << "but file has " << file_dim << " dimension." << std::endl;
      diskann::cerr << stream.str() << std::endl;
      aligned_free(this->_data);
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }

    copy_aligned_data_from_file<T>(filename, this->_data, file_num_points, file_dim,
                                   this->_aligned_dim);
    if (this->_normalize_vecs) {
      for (uint64_t i = 0; i < file_num_points; i++) {
        normalize(this->_data + this->_aligned_dim * i, this->_aligned_dim);
      }
    }

    diskann::cout << "Using only first " << num_points_to_load
                  << " from file.. " << std::endl;

    this->_nd = num_points_to_load;
    this->my_build_with_data_populated(parameters, tags);
  }

  template<typename T, typename TagT>
  void myIndex<T, TagT>::my_build_with_data_populated(Parameters &parameters, 
                                                      const std::vector<TagT> &tags) {
    std::cout << "running my build with data popoulated" << std::endl;
    diskann::cout << "Starting index build with " << this->_nd << " points... "
                  << std::endl;

    if (this->_nd < 1)
      throw ANNException("Error: Trying to build an index with 0 points", -1,
                         __FUNCSIG__, __FILE__, __LINE__);

    if (this->_enable_tags && tags.size() != this->_nd) {
      std::stringstream stream;
      stream << "ERROR: Driver requests loading " << this->_nd << " points from file,"
             << "but tags vector is of size " << tags.size() << "."
             << std::endl;
      diskann::cerr << stream.str() << std::endl;
      aligned_free(this->_data);
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }
    if (this->_enable_tags) {
      for (size_t i = 0; i < tags.size(); ++i) {
        this->_tag_to_location[tags[i]] = (unsigned) i;
        (this->_location_to_tag).set(static_cast<unsigned>(i), tags[i]);
      }
    }

    this->generate_frozen_point();
    this->mylink(parameters);

    if (this->_support_eager_delete) {
      this->update_in_graph();  // copying values to in_graph
    }

    #pragma omp parallel for schedule(dynamic, 4096)
    for( size_t i = 0; i < this->_nd; i++) {
      auto &pool = this->_final_graph[i];
      for( auto &nbr : pool) {
        LockGuard guard( this->_locks[nbr]);
        this->wws_[nbr] += 1.0f;
      }
    }

    #pragma omp parallel for schedule(dynamic, 4096)
    for( size_t i = 0; i < this->_nd; i++) {
      auto &w_pool = this->weights_[i];
      for( auto &weight : w_pool) {
        weight *= this->wws_[i];
      }
    }

    size_t max = 0, min = SIZE_MAX, total = 0, cnt = 0;
    float max_w = 0.0f, min_w = FLT_MAX, total_w = 0.0f;
    size_t cnt_0 = 0;
    for (size_t i = 0; i < this->_nd; i++) {
      auto &pool = this->_final_graph[i];
      max = (std::max)(max, pool.size());
      min = (std::min)(min, pool.size());
      total += pool.size();
      if (pool.size() < 2)
        cnt++;
      auto &pool_w = this->weights_[i];
      max_w = (std::max)(max_w, *std::max_element(pool_w.begin(), pool_w.end()));
      min_w = (std::min)(min_w, *std::min_element(pool_w.begin(), pool_w.end()));
      total_w += std::accumulate(pool_w.begin(), pool_w.end(), 0);
      for( auto &tmp : pool_w) {
        if( tmp < 2) 
          cnt_0++;
      }
    }
    diskann::cout << "Index built with degree: max:" << max
                  << "  avg:" << (float) total / (float) (this->_nd + this->_num_frozen_pts)
                  << "  min:" << min << "  count(deg<2):" << cnt << std::endl;
    diskann::cout << "Index built with weight: max:" << max_w
                  << "  avg:" << (float) total_w / (float) (total)
                  << "  min:" << min_w 
                  << "  count(weight<2):" << cnt_0 << " #edges:" << total
                  << std::endl;

    this->_max_observed_degree = (std::max)((unsigned) max, this->_max_observed_degree);
    this->_has_built = true;
  }

  template<typename T, typename TagT>
  void myIndex<T, TagT>::mylink(Parameters &parameters) {
    std::cout << "running mylink" << std::endl;
    unsigned num_threads = parameters.Get<unsigned>("num_threads");
    if (num_threads != 0)
      omp_set_num_threads(num_threads);

    uint32_t num_syncs =
        (unsigned) DIV_ROUND_UP(this->_nd + this->_num_frozen_pts, (64 * 64));
    if (num_syncs < 40)
      num_syncs = 40;
    diskann::cout << "Number of syncs: " << num_syncs << std::endl;

    this->_saturate_graph = parameters.Get<bool>("saturate_graph");

    if (num_threads != 0)
      omp_set_num_threads(num_threads);

    this->_indexingQueueSize = parameters.Get<unsigned>("L");  // Search list size
    this->_indexingRange = parameters.Get<unsigned>("R");
    this->_indexingMaxC = parameters.Get<unsigned>("C");
    const float last_round_alpha = parameters.Get<float>("alpha");
    unsigned    L = this->_indexingQueueSize;

    std::vector<unsigned> Lvec;
    Lvec.push_back(L);
    Lvec.push_back(L);
    const unsigned NUM_RNDS = 2;
    this->_indexingAlpha = 1.0f;

    /* visit_order is a vector that is initialized to the entire graph */
    std::vector<unsigned>          visit_order;
    std::vector<diskann::Neighbor> pool, tmp;
    tsl::robin_set<unsigned>       visited;
    visit_order.reserve(this->_nd + this->_num_frozen_pts);
    for (unsigned i = 0; i < (unsigned) this->_nd; i++) {
      visit_order.emplace_back(i);
    }

    if (this->_num_frozen_pts > 0)
      visit_order.emplace_back((unsigned) this->_max_points);

    // if there are frozen points, the first such one is set to be the _start
    if (this->_num_frozen_pts > 0)
      this->_start = (unsigned) this->_max_points;
    else
      this->_start = this->calculate_entry_point();

    if (this->_support_eager_delete) {
      (this->_in_graph).reserve(this->_max_points + this->_num_frozen_pts);
      (this->_in_graph).resize(this->_max_points + this->_num_frozen_pts);
    }

    for (uint64_t p = 0; p < this->_nd; p++) {
      this->_final_graph[p].reserve(
          (size_t)(std::ceil(this->_indexingRange * GRAPH_SLACK_FACTOR * 1.05)));
      this->weights_.reserve(
          (size_t)(std::ceil(this->_indexingRange * GRAPH_SLACK_FACTOR * 1.05)));
    }

    std::random_device               rd;
    std::mt19937                     gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    // creating a initial list to begin the search process. it has _start and
    // random other nodes
    std::set<unsigned> unique_start_points;
    unique_start_points.insert(this->_start);

    std::vector<unsigned> init_ids;
    for (auto pt : unique_start_points)
      init_ids.emplace_back(pt);

    diskann::Timer link_timer;
    for (uint32_t rnd_no = 0; rnd_no < NUM_RNDS; rnd_no++) {
      const size_t total_internal_points = this->_max_points + this->_num_frozen_pts;
      this->wws_.clear();
      this->wws_.resize( total_internal_points, 0.0f);
      #pragma omp parallel for schedule(dynamic, 4096)
      for( size_t i = 0; i < total_internal_points; i++) {
        for( auto &weight : this->weights_[i]) {
          weight = 1.0f;
        }
      }

      L = Lvec[rnd_no];

      if (rnd_no == NUM_RNDS - 1) {
        if (last_round_alpha > 1)
          this->_indexingAlpha = last_round_alpha;
      }

      double   sync_time = 0, total_sync_time = 0;
      double   inter_time = 0, total_inter_time = 0;
      size_t   inter_count = 0, total_inter_count = 0;
      unsigned progress_counter = 0;

      size_t round_size = DIV_ROUND_UP(this->_nd, num_syncs);  // size of each batch
      std::vector<unsigned> need_to_sync(this->_max_points + this->_num_frozen_pts, 0);

      std::vector<std::vector<unsigned>> pruned_list_vector(round_size);
      std::vector<std::vector<float>> weight_list_vector(round_size);

      for (uint32_t sync_num = 0; sync_num < num_syncs; sync_num++) {
        // std::cout << "current sync_num: " << sync_num << std::endl;
        size_t start_id = sync_num * round_size;
        size_t end_id =
            (std::min)(this->_nd + this->_num_frozen_pts, (sync_num + 1) * round_size);

        auto s = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff;

#pragma omp parallel for schedule(dynamic)
        for (_s64 node_ctr = (_s64) start_id; node_ctr < (_s64) end_id;
             ++node_ctr) {
          auto                     node = visit_order[node_ctr];
          size_t                   node_offset = node_ctr - start_id;
          tsl::robin_set<unsigned> visited;
          std::vector<unsigned> &pruned_list = pruned_list_vector[node_offset];
          std::vector<float> &weight_list = weight_list_vector[node_offset];
          // get nearest neighbors of n in tmp. pool contains all the
          // points that were checked along with their distance from
          // n. visited contains all the points visited, just the ids
          std::vector<Neighbor> pool;
          pool.reserve(L * 2);
          visited.reserve(L * 2);
          this->get_expanded_nodes(node, L, init_ids, pool, visited);
          // check the neighbors of the query that are not part of
          // visited, check their distance to the query, and add it to
          if (!this->_final_graph[node].empty())
            for (auto id : this->_final_graph[node]) {
              if (visited.find(id) == visited.end() && id != node) {
                float dist =
                    this->_distance->compare(this->_data + this->_aligned_dim * (size_t) node,
                                       this->_data + this->_aligned_dim * (size_t) id,
                                       (unsigned) this->_aligned_dim);
                pool.emplace_back(Neighbor(id, dist, true));
                visited.insert(id);
              }
            }
          this->my_prune_neighbors(node, pool, pruned_list, weight_list);
        }
        diff = std::chrono::high_resolution_clock::now() - s;
        sync_time += diff.count();
        // std::cout << "\t1st prune" << std::endl;

// prune_neighbors will check pool, and remove some of the points and
// create a cut_graph, which contains neighbors for point n
#pragma omp parallel for schedule(dynamic, 64)
        for (_s64 node_ctr = (_s64) start_id; node_ctr < (_s64) end_id;
             ++node_ctr) {
          _u64                   node = visit_order[node_ctr];
          size_t                 node_offset = node_ctr - start_id;
          std::vector<unsigned> &pruned_list = pruned_list_vector[node_offset];
          this->_final_graph[node].clear();
          for (auto id : pruned_list)
            this->_final_graph[node].emplace_back(id);
          std::vector<float> &weight_list = weight_list_vector[node_offset];
          this->weights_[node].clear();
          for (auto w : weight_list)
            this->weights_[node].emplace_back(w);
          weight_list.clear();
          weight_list.shrink_to_fit();
        }
        // std::cout << "\tadd edge" << std::endl;
        s = std::chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(dynamic, 64)
        for (_s64 node_ctr = start_id; node_ctr < (_s64) end_id; ++node_ctr) {
          auto                   node = visit_order[node_ctr];
          _u64                   node_offset = node_ctr - start_id;
          std::vector<unsigned> &pruned_list = pruned_list_vector[node_offset];
          this->my_batch_inter_insert(node, pruned_list, need_to_sync);
          pruned_list.clear();
          pruned_list.shrink_to_fit();
        }
        // std::cout << "\tinsert reverse" << std::endl;

#pragma omp parallel for schedule(dynamic, 65536)
        for (_s64 node_ctr = 0; node_ctr < (_s64)(visit_order.size());
             node_ctr++) {
          auto node = visit_order[node_ctr];
          if (need_to_sync[node] != 0) {
            need_to_sync[node] = 0;
            inter_count++;
            tsl::robin_set<unsigned> dummy_visited(0);
            std::vector<Neighbor>    dummy_pool(0);
            std::vector<unsigned>    new_out_neighbors;
            std::vector<float>    new_out_weights;

            for (auto cur_nbr : this->_final_graph[node]) {
              if (dummy_visited.find(cur_nbr) == dummy_visited.end() &&
                  cur_nbr != node) {
                float dist =
                    this->_distance->compare(this->_data + this->_aligned_dim * (size_t) node,
                                             this->_data + this->_aligned_dim * (size_t) cur_nbr,
                                             (unsigned) this->_aligned_dim);
                dummy_pool.emplace_back(Neighbor(cur_nbr, dist, true));
                dummy_visited.insert(cur_nbr);
              }
            }
            this->my_prune_neighbors(node, dummy_pool, new_out_neighbors, new_out_weights);

            this->_final_graph[node].clear();
            for (auto id : new_out_neighbors)
              this->_final_graph[node].emplace_back(id);
            
            this->weights_[node].clear();
            for (auto w : new_out_weights)
              this->weights_[node].emplace_back(w);
          }
        }
        // std::cout << "\t2nd prune" << std::endl;

        diff = std::chrono::high_resolution_clock::now() - s;
        inter_time += diff.count();

        if ((sync_num * 100) / num_syncs > progress_counter) {
          diskann::cout.precision(4);
          diskann::cout << "Completed  (round: " << rnd_no
                        << ", sync: " << sync_num << "/" << num_syncs
                        << " with L " << L << ")"
                        << " sync_time: " << sync_time << "s"
                        << "; inter_time: " << inter_time << "s" << std::endl;

          total_sync_time += sync_time;
          total_inter_time += inter_time;
          total_inter_count += inter_count;
          sync_time = 0;
          inter_time = 0;
          inter_count = 0;
          progress_counter += 5;
        }
      }
// Gopal. Splitting nsg_dll into separate DLLs for search and build.
// This code should only be available in the "build" DLL.
#if defined(RELEASE_UNUSED_TCMALLOC_MEMORY_AT_CHECKPOINTS) && \
    defined(DISKANN_BUILD)
      MallocExtension::instance()->ReleaseFreeMemory();
#endif
      if (this->_nd > 0) {
        diskann::cout << "Completed Pass " << rnd_no << " of data using L=" << L
                      << " and alpha=" << parameters.Get<float>("alpha")
                      << ". Stats: ";
        diskann::cout << "search+prune_time=" << total_sync_time
                      << "s, inter_time=" << total_inter_time
                      << "s, inter_count=" << total_inter_count << std::endl;
      }
    }

    if (this->_nd > 0) {
      diskann::cout << "Starting final cleanup.." << std::flush;
    }
#pragma omp parallel for schedule(dynamic, 65536)
    for (_s64 node_ctr = 0; node_ctr < (_s64)(visit_order.size()); node_ctr++) {
      auto node = visit_order[node_ctr];
      if (this->_final_graph[node].size() > this->_indexingRange) {
        tsl::robin_set<unsigned> dummy_visited(0);
        std::vector<Neighbor>    dummy_pool(0);
        std::vector<unsigned>    new_out_neighbors;
        std::vector<float>    new_out_weights;

        for (auto cur_nbr : this->_final_graph[node]) {
          if (dummy_visited.find(cur_nbr) == dummy_visited.end() &&
              cur_nbr != node) {
            float dist =
                (this->_distance)->compare(this->_data + this->_aligned_dim * (size_t) node,
                                           this->_data + this->_aligned_dim * (size_t) cur_nbr,
                                           (unsigned) this->_aligned_dim);
            dummy_pool.emplace_back(Neighbor(cur_nbr, dist, true));
            dummy_visited.insert(cur_nbr);
          }
        }
        this->my_prune_neighbors(node, dummy_pool, new_out_neighbors, new_out_weights);

        this->_final_graph[node].clear();
        for (auto id : new_out_neighbors)
          this->_final_graph[node].emplace_back(id);
        
        this->weights_[node].clear();
          for (auto w : new_out_weights)
            this->weights_[node].emplace_back(w);
      }
    }
    if (this->_nd > 0) {
      diskann::cout << "done. Link time: "
                    << ((double) link_timer.elapsed() / (double) 1000000) << "s"
                    << std::endl;
    }
  }

  template<typename T, typename TagT>
  void myIndex<T, TagT>::my_prune_neighbors(const unsigned        location,
                                            std::vector<Neighbor> &pool,
                                            std::vector<unsigned> &pruned_list,
                                            std::vector<float> &weight_list) {
    my_prune_neighbors(location, pool, this->_indexingRange, this->_indexingMaxC,
                       this->_indexingAlpha, pruned_list, weight_list);
  }

  template<typename T, typename TagT>
  void myIndex<T, TagT>::my_prune_neighbors(const unsigned        location,
                                            std::vector<Neighbor> &pool,
                                            const _u32            range,
                                            const _u32  max_candidate_size,
                                            const float alpha,
                                            std::vector<unsigned> &pruned_list,
                                            std::vector<float> &weight_list) {
    if (pool.size() == 0) {
      std::stringstream ss;
      ss << "Thread loc:" << std::this_thread::get_id()
         << " Pool address: " << &pool << std::endl;
      std::cout << ss.str();
      throw diskann::ANNException("Pool passed to prune_neighbors is empty",
                                  -1);
    }

    this->_max_observed_degree = (std::max)(this->_max_observed_degree, range);

    // sort the pool based on distance to query
    std::sort(pool.begin(), pool.end());

    // std::vector<Neighbor> result;
    std::vector<std::pair<Neighbor, float>> result;
    result.reserve(range);

    this->my_occlude_list(pool, alpha, range, max_candidate_size, result, location);

    pruned_list.clear();
    weight_list.clear();
    assert(result.size() <= range);
    for (auto iter : result) {
      // if (iter.id != location)
      //   pruned_list.emplace_back(iter.id);
      if( iter.first.id != location) {
        pruned_list.emplace_back(iter.first.id);
        weight_list.emplace_back(iter.second);
        // {
        //   LockGuard guard( this->_locks[iter.first.id]);
        //   this->wws_[iter.first.id] += 1.0f;
        // }
      }
    }

    if (this->_saturate_graph && alpha > 1) {
      for (uint32_t i = 0; i < pool.size() && pruned_list.size() < range; i++) {
        if ((std::find(pruned_list.begin(), pruned_list.end(), pool[i].id) ==
             pruned_list.end()) &&
            pool[i].id != location) {
          pruned_list.emplace_back(pool[i].id);
          weight_list.emplace_back(1.0f);
        }
      }
    }
  }

  template<typename T, typename TagT>
  void myIndex<T, TagT>::my_occlude_list(std::vector<Neighbor> &pool,
                                         const float alpha, const unsigned degree,
                                         const unsigned         maxc,
                                         std::vector<std::pair<Neighbor, float>> &result, 
                                         const unsigned location) {
    if (pool.size() == 0)
      return;

    assert(std::is_sorted(pool.begin(), pool.end()));
    if (pool.size() > maxc)
      pool.resize(maxc);
    std::vector<float> occlude_factor(pool.size(), 0);

    float cur_alpha = 1;
    while (cur_alpha <= alpha && result.size() < degree) {
      // used for MIPS, where we store a value of eps in cur_alpha to
      // denote pruned out entries which we can skip in later rounds.
      float eps = cur_alpha + 0.01f;

      for (auto iter = pool.begin();
           result.size() < degree && iter != pool.end(); ++iter) {
        if (occlude_factor[iter - pool.begin()] > cur_alpha) {
          continue;
        }
        occlude_factor[iter - pool.begin()] = std::numeric_limits<float>::max();
        // result.push_back(*iter);
        float weight = 1.0f;
        auto tmp_iter = std::find(this->_final_graph[location].begin(), 
                                  this->_final_graph[location].end(), iter->id);
        if( tmp_iter != this->_final_graph[location].end()) {
          weight = this->weights_[location][tmp_iter - this->_final_graph[location].begin()];
        }
        for (auto iter2 = iter + 1; iter2 != pool.end(); iter2++) {
          auto t = iter2 - pool.begin();
          if (occlude_factor[t] > alpha)
            continue;
          float djk =
              (this->_distance)->compare(this->_data + this->_aligned_dim * (size_t) iter2->id,
                                         this->_data + this->_aligned_dim * (size_t) iter->id,
                                         (unsigned) this->_aligned_dim);
          if (this->_dist_metric == diskann::Metric::L2 ||
              this->_dist_metric == diskann::Metric::COSINE) {
            if (djk == 0.0) {
              occlude_factor[t] = std::numeric_limits<float>::max();
              // auto tmp_iter2 = std::find(this->_final_graph[location].begin(), 
              //                            this->_final_graph[location].end(), iter2->id);
              // if( tmp_iter2 != this->_final_graph[location].end()) {
                // weight += this->weights_[location][tmp_iter2 - this->_final_graph[location].begin()];
                // weight += 1.0f;
              // }
              // else {
                // weight += 1.0f;
                // {
                //   LockGuard guard( this->_locks[iter2->id]);
                //   this->wws_[iter2->id] += 1.0f;
                // }
              // }
              weight += 1.0f;
              {
                LockGuard guard( this->_locks[iter2->id]);
                this->wws_[iter2->id] += 1.0f;
              }
            }
            else {
              float ratio_dist = iter2->distance / djk;
              if( occlude_factor[t] <= cur_alpha && ratio_dist > cur_alpha) {
                // auto tmp_iter2 = std::find(this->_final_graph[location].begin(), 
                //                            this->_final_graph[location].end(), iter2->id);
                // if( tmp_iter2 != this->_final_graph[location].end()) {
                  // weight += this->weights_[location][tmp_iter2 - this->_final_graph[location].begin()];
                  // weight += 1.0f;
                // }
                // else {
                  // weight += 1.0f;
                  // {
                  //   LockGuard guard( this->_locks[iter2->id]);
                  //   this->wws_[iter2->id] += 1.0f;
                  // }
                // }
                weight += 1.0f;
                {
                  LockGuard guard( this->_locks[iter2->id]);
                  this->wws_[iter2->id] += 1.0f;
                }
              }
              occlude_factor[t] = std::max(occlude_factor[t], ratio_dist);
              // occlude_factor[t] =
              //     std::max(occlude_factor[t], iter2->distance / djk);
            }
          } else if (this->_dist_metric == diskann::Metric::INNER_PRODUCT) {
            // Improvization for flipping max and min dist for MIPS
            float x = -iter2->distance;
            float y = -djk;
            if (y > cur_alpha * x) {
              if( occlude_factor[t] < eps - 1e-6) {
                // auto tmp_iter2 = std::find(this->_final_graph[location].begin(), 
                //                            this->_final_graph[location].end(), iter2->id);
                // if( tmp_iter2 != this->_final_graph[location].end()) {
                  // weight += this->weights_[location][tmp_iter2 - this->_final_graph[location].begin()];
                  // weight += 1.0f;
                // }
                // else {
                  // weight += 1.0f;
                  // {
                  //   LockGuard guard( this->_locks[iter2->id]);
                  //   this->wws_[iter2->id] += 1.0f;
                  // }
                // }
                weight += 1.0f;
                {
                  LockGuard guard( this->_locks[iter2->id]);
                  this->wws_[iter2->id] += 1.0f;
                }
              }
              occlude_factor[t] = std::max(occlude_factor[t], eps);
            }
          }
        }
        result.emplace_back(std::make_pair( *iter, weight));
        // {
        //   LockGuard guard( this->_locks[iter->id]);
        //   if( cur_alpha < 1.0001f && iter->id != location) {
        //     this->wws_[iter->id] += 1.0f;
        //   }
        // }
      }
      cur_alpha *= 1.2;
    }
  }

  template<typename T, typename TagT>
  void myIndex<T, TagT>::my_batch_inter_insert(
          unsigned n, const std::vector<unsigned> &pruned_list, const _u32 range,
          std::vector<unsigned> &need_to_sync) {
    // assert(!src_pool.empty());

    for (auto des : pruned_list) {
      if (des == n)
        continue;
      // des.loc is the loc of the neighbors of n
      assert(des >= 0 && des < this->_max_points + this->_num_frozen_pts);
      if (des > this->_max_points)
        diskann::cout << "error. " << des << " exceeds max_pts" << std::endl;
      // des_pool contains the neighbors of the neighbors of n

      {
        LockGuard guard(this->_locks[des]);
        if (std::find(this->_final_graph[des].begin(), this->_final_graph[des].end(), n) ==
            this->_final_graph[des].end()) {
          this->_final_graph[des].push_back(n);
          this->weights_[des].push_back(1.0f);
          // this->wws_[des] += 1.0f;
          if (this->_final_graph[des].size() >
              (unsigned) (range * GRAPH_SLACK_FACTOR))
            need_to_sync[des] = 1;
        }
      }  // des lock is released by this point
    }
  }

  template<typename T, typename TagT>
  void myIndex<T, TagT>::my_batch_inter_insert(
          unsigned n, const std::vector<unsigned> &pruned_list,
          std::vector<unsigned> &need_to_sync) {
    my_batch_inter_insert(n, pruned_list, this->_indexingRange, need_to_sync);
  }

  template<typename T, typename TagT>
  void myIndex<T, TagT>::mysave(const char *filename, bool compact_before_save) {
    std::cout << "running mysave" << std::endl;
    diskann::Timer timer;

    if (compact_before_save) {
      this->compact_data();
      this->compact_frozen_point();
    } else {
      if (not this->_data_compacted) {
        throw ANNException(
            "Index save for non-compacted index is not yet implemented", -1,
            __FUNCSIG__, __FILE__, __LINE__);
      }
    }

    std::unique_lock<std::shared_timed_mutex> ul(this->_update_lock);
    std::unique_lock<std::shared_timed_mutex> cl(this->_consolidate_lock);
    std::unique_lock<std::shared_timed_mutex> tl(this->_tag_lock);
    std::unique_lock<std::shared_timed_mutex> dl(this->_delete_lock);

    if (!this->_save_as_one_file) {
      std::string graph_file = std::string(filename);
      std::string tags_file = std::string(filename) + ".tags";
      std::string data_file = std::string(filename) + ".data";
      std::string delete_list_file = std::string(filename) + ".del";
      std::string weights_file = std::string(filename) + ".weights";

      // Because the save_* functions use append mode, ensure that
      // the files are deleted before save. Ideally, we should check
      // the error code for delete_file, but will ignore now because
      // delete should succeed if save will succeed.
      delete_file(graph_file);
      this->save_graph(graph_file);
      delete_file(data_file);
      this->save_data(data_file);
      delete_file(tags_file);
      this->save_tags(tags_file);
      delete_file(delete_list_file);
      this->save_delete_list(delete_list_file);
      delete_file(weights_file);
      this->my_save_weights(weights_file);
    } else {
      diskann::cout << "Save index in a single file currently not supported. "
                       "Not saving the index."
                    << std::endl;
    }

    this->reposition_frozen_point_to_end();

    diskann::cout << "Time taken for save: " << timer.elapsed() / 1000000.0
                  << "s." << std::endl;
  }

  template<typename T, typename TagT>
  _u64 myIndex<T, TagT>::my_save_weights(std::string graph_file) {
    std::ofstream out;
    open_file_to_write(out, graph_file);

    _u64 file_offset = 0;  // we will use this if we want
    out.seekp(file_offset, out.beg);

    float max_weight = 0.0f;
    out.write((char *) &max_weight, sizeof(float));
    _u64 num_edges = 0;
    out.write((char *) &num_edges, sizeof(_u64));
    out.write((char *) &this->_nd, sizeof(_u64));
    for (unsigned i = 0; i < this->_nd; i++) {
      unsigned out_degree = (unsigned) this->weights_[i].size();
      out.write((char *) &out_degree, sizeof(unsigned));
      out.write((char *) this->weights_[i].data(), out_degree * sizeof(float));
      max_weight = std::max( max_weight,
                             *std::max_element( this->weights_[i].begin(), 
                                                this->weights_[i].end()));
      num_edges += out_degree;
    }
    out.seekp(file_offset, out.beg);
    out.write((char *) &max_weight, sizeof(float));
    out.write((char *) &num_edges, sizeof(_u64));
    out.close();
    std::cout << "saving weights..."
              << ", #edges:" << num_edges
              << ", #vertice:" << this->_nd
              << ", max weight:" << max_weight
              << std::endl;
    return num_edges;  // number of bytes written
  }

  // EXPORTS
  template DISKANN_DLLEXPORT class myIndex<float, int32_t>;
  template DISKANN_DLLEXPORT class myIndex<int8_t, int32_t>;
  template DISKANN_DLLEXPORT class myIndex<uint8_t, int32_t>;
  template DISKANN_DLLEXPORT class myIndex<float, uint32_t>;
  template DISKANN_DLLEXPORT class myIndex<int8_t, uint32_t>;
  template DISKANN_DLLEXPORT class myIndex<uint8_t, uint32_t>;
  template DISKANN_DLLEXPORT class myIndex<float, int64_t>;
  template DISKANN_DLLEXPORT class myIndex<int8_t, int64_t>;
  template DISKANN_DLLEXPORT class myIndex<uint8_t, int64_t>;
  template DISKANN_DLLEXPORT class myIndex<float, uint64_t>;
  template DISKANN_DLLEXPORT class myIndex<int8_t, uint64_t>;
  template DISKANN_DLLEXPORT class myIndex<uint8_t, uint64_t>;
}