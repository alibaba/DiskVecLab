#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <memory>
#include <utility>
#include <assert.h>
#include <omp.h>
#include <cstring>
#include <algorithm>
#include <float.h>
// #include <numeric>
#include <chrono>
#include <unordered_map>
#include <mutex>
#include <queue>

#include "aux_utils.h"
#include "partition_and_pq.h"
#include "math_utils.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#define BLOCK_SIZE 5000000


struct my_timer {
    std::chrono::time_point<std::chrono::high_resolution_clock> begin_time_;
    std::chrono::time_point<std::chrono::high_resolution_clock> end_time_;

    my_timer() {
        begin_time_ = std::chrono::high_resolution_clock::now();
    }

    void reset() {
        begin_time_ = std::chrono::high_resolution_clock::now();
    }

    void timing( const std::string &message) {
        end_time_ = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>( end_time_ - begin_time_);
        std::cout << message << " in " << duration.count() << "s" << std::endl;
    }
};


struct edge {
    unsigned u_;
    unsigned v_;
    float w_;

    edge() : u_(0), v_(0), w_(0.0f) {}
    edge( unsigned u, unsigned v, float w) :
          u_(u), v_(v), w_(w) {}

    bool operator < ( const edge &other) {
        return w_ > other.w_;
    }
};


class cluster {

public:

    cluster( std::vector<unsigned> &points, unsigned *id_in_cluster) {
        size_ = points.size();
        ori_ids_.resize( size_);
        undir_graph_.resize( size_);
        undir_weight_.resize( size_);
        #pragma omp parallel for
        for( unsigned i = 0; i < size_; i++) {
            ori_ids_[i] = points[i];
            id_in_cluster[points[i]] = i;
        }
    }

    // 在 greedy_layout 之后调用，释放不再需要的邻接表内存
    void release_graph_memory() {
        std::vector<std::vector<unsigned>>().swap(undir_graph_);
        std::vector<std::vector<float>>().swap(undir_weight_);
        // id2p_ 如果你后面不再用，也可以顺便清掉（可选）
        // std::vector<unsigned>().swap(id2p_);
    }


    void get_direct_graph( std::vector<std::vector<unsigned>> &ori_graph,
                           std::vector<std::vector<float>> &ori_weight,
                           uint32_t *labels, unsigned *id_in_cluster) {
        #pragma omp parallel for
        for( unsigned i = 0; i < size_; i++) {
            unsigned ori_id = ori_ids_[i];
            unsigned degree = ori_graph[ori_id].size();
            for( unsigned j = 0; j < degree; j++) {
                unsigned nbr = ori_graph[ori_id][j];
                if( labels[nbr] != labels[ori_id]) {
                    continue;
                }
                unsigned local_id = id_in_cluster[nbr];
                undir_graph_[i].push_back( local_id);
                undir_weight_[i].push_back( ori_weight[ori_id][j]);
            }
        }
    }

    void trans2undirect() {

        std::vector<std::vector<unsigned>> in_graph( size_);
        std::vector<std::vector<float>> in_weight( size_);
        std::vector<std::mutex> mtxs( size_);
        #pragma omp parallel for
        for( unsigned i = 0; i < size_; i++) {
            unsigned degree = undir_graph_[i].size();
            for( unsigned j = 0; j < degree; j++) {
                unsigned nbr = undir_graph_[i][j];
                std::lock_guard<std::mutex> lock( mtxs[nbr]);
                in_graph[nbr].push_back( i);
                in_weight[nbr].push_back( undir_weight_[i][j]);
            }
        }

        #pragma omp parallel for
        for( unsigned i = 0; i < size_; i++) {
            unsigned degree = in_graph[i].size();
            std::vector<unsigned> append_nbrs;
            std::vector<float> append_weights;
            append_nbrs.reserve( degree);
            append_weights.reserve( degree);
            for( unsigned j = 0; j < degree; j++) {
                unsigned nbr = in_graph[i][j];
                auto iter = std::find(undir_graph_[i].begin(), undir_graph_[i].end(), nbr);
                if( iter == undir_graph_[i].end()) {
                    append_nbrs.push_back( nbr);
                    append_weights.push_back( in_weight[i][j]);
                }
                else {
                    undir_weight_[i][iter-undir_graph_[i].begin()] += in_weight[i][j];
                }
            }
            undir_graph_[i].insert( undir_graph_[i].end(), 
                                    std::make_move_iterator( append_nbrs.begin()),
                                    std::make_move_iterator( append_nbrs.end()));
            undir_weight_[i].insert( undir_weight_[i].end(),
                                     std::make_move_iterator( append_weights.begin()),
                                     std::make_move_iterator( append_weights.end()));
        }
    }


    void push_nbrs( unsigned id, std::vector<bool> &vis,
                    std::unordered_map<unsigned, float> &map) {
        unsigned degree = undir_graph_[id].size();
        for( unsigned j = 0; j < degree; j++) {
            unsigned nbr = undir_graph_[id][j];
            if( vis[nbr]) {
                continue;
            }
            auto iter = map.find( nbr);
            if( iter == map.end()) {
                map[nbr] = undir_weight_[id][j];
            }
            else {
                iter->second += undir_weight_[id][j];
            }
        }
    }

    bool find_next( unsigned &id, std::unordered_map<unsigned, float> &map) {
        if( map.empty()) {
            return false;
        }
        float benefit = 0.0f;
        for( auto &pair : map) {
            if( benefit < pair.second) {
                id = pair.first;
                benefit = pair.second;
            }
        }
        map.erase( id);
        return true;
    }


    void greedy_layout( unsigned page_capacity, bool post=false) {
        unsigned num_pages = size_ / page_capacity + (size_ % page_capacity != 0);
        id2p_.resize( size_);
        p2id_.resize( num_pages);
        for( auto &page : p2id_) {
            page.reserve( page_capacity);
        }
        unsigned num_edge = 0;
        for( unsigned i = 0; i < size_; i++) {
            num_edge += undir_graph_[i].size();
        }
        num_edge /= 2;
        std::vector<edge> all_edges;
        all_edges.reserve( num_edge);
        for( unsigned i = 0; i < size_; i++) {
            unsigned degree = undir_graph_[i].size();
            for( unsigned j = 0; j < degree; j++) {
                unsigned nbr = undir_graph_[i][j];
                if( nbr < i) {
                    continue;
                }
                all_edges.emplace_back( i, nbr, undir_weight_[i][j]);
            }
        }
        std::sort( all_edges.begin(), all_edges.end());

        std::vector<bool> vis( size_, false);
        unsigned page_id = 0;
        unsigned cur_capacity = page_capacity;
        unsigned unfull_cnt = 0;
        unsigned added_cnt = 0;
        for( unsigned edge_id = 0; page_id < num_pages && edge_id < num_edge;) {
            edge cur_edge = all_edges[edge_id];
            unsigned u = cur_edge.u_;
            unsigned v = cur_edge.v_;
            if( cur_capacity == page_capacity) {
                if( vis[u] || vis[v]) {
                    edge_id++;
                    continue;
                }
                id2p_[u] = page_id;
                id2p_[v] = page_id;
                p2id_[page_id].push_back(u);
                p2id_[page_id].push_back(v);
                vis[u] = true;
                vis[v] = true;
                cur_capacity -= 2;
                added_cnt += 2;
                continue;
            }
            if( cur_capacity == 0) {
                cur_capacity = page_capacity;
                page_id++;
                continue;
            }

            std::unordered_map<unsigned, float> id2benefit;
            for( unsigned &id : p2id_[page_id]) {
                push_nbrs( id, vis, id2benefit);
            }
            while( cur_capacity > 0) {
                unsigned target_id;
                if( !find_next( target_id, id2benefit)) {
                    break;
                }
                id2p_[target_id] = page_id;
                p2id_[page_id].push_back(target_id);
                vis[target_id] = true;
                cur_capacity--;
                added_cnt++;
                push_nbrs( target_id, vis, id2benefit);
            }
            if( cur_capacity > 0) {
                unfull_cnt++;
            }
            page_id++;
            cur_capacity = page_capacity;
        }
        // std::cout << unfull_cnt << " " << added_cnt << " " << size_ <<std::endl;
        tmp_cnt_ = size_ - added_cnt;
        for( auto &page : p2id_) {
            if( page.size() < page_capacity) {
                tmp_cnt_ += page.size();
            }
        }

        if( !post) {
            get_unassigned( vis, page_capacity);
        }
        else {
            unsigned pid = 0;
            for( unsigned i = 0; i < size_; i++) {
                if( !vis[i]) {
                    while( p2id_[pid].size() == page_capacity) {
                        pid++;
                    }
                    p2id_[pid].push_back( i);
                }
            }
        }
    }


    void get_unassigned( std::vector<bool> &vis, unsigned page_capacity) {
        unassigned_.reserve( tmp_cnt_);
        for( unsigned i = 0; i < size_; i++) {
            if( !vis[i] || p2id_[id2p_[i]].size() < page_capacity) {
                unassigned_.push_back( ori_ids_[i]);
            }
        }
        assert( unassigned_.size() == tmp_cnt_);
    }
    

    unsigned size_;
    std::vector<unsigned> ori_ids_;
    std::vector<std::vector<unsigned>> undir_graph_;
    std::vector<std::vector<float>> undir_weight_;
    std::vector<unsigned> id2p_;
    std::vector<std::vector<unsigned>> p2id_;
    unsigned tmp_cnt_;
    std::vector<unsigned> unassigned_;
};


class mincut {

public:

    void load_meta_data( std::string &disk_path) {
        std::ifstream meta_reader( disk_path, std::ios::binary);
        int meta_n, meta_dim;
        std::vector<uint64_t> meta_data;
        meta_reader.read((char*) &meta_n, sizeof(int));
        meta_reader.read((char*) &meta_dim, sizeof(int));
        meta_data.resize( meta_n);
        meta_reader.read((char*) meta_data.data(), meta_n * sizeof(uint64_t));
        this->npts_ = (unsigned) meta_data[0];
        this->page_capacity_ = (unsigned) meta_data[4];
        this->n_pages_ = (unsigned) DIV_ROUND_UP( this->npts_, this->page_capacity_);
        std::cout << "npts: " << this->npts_ << ", #nodes per page: " 
        << this->page_capacity_ << ", #pages: " << this->n_pages_ << std::endl;
        meta_reader.close();
    }

    void load_index_graph( std::string &index_path) {
        my_timer timer;

        this->index_graph_.resize( this->npts_);
        this->index_weight_.resize( this->npts_);
        std::ifstream index_reader( index_path, std::ios::binary);
        std::ifstream weight_reader( index_path + ".weights", std::ios::binary);

        float max_weight;
        uint64_t npts, num_edges;
        unsigned max_degree;
        weight_reader.read((char*) &max_weight, sizeof(float));
        weight_reader.read((char*) &num_edges, sizeof(uint64_t));
        weight_reader.read((char*) &npts, sizeof(uint64_t));
        uint64_t offset = sizeof(uint64_t);
        index_reader.seekg(offset, std::ios::beg);
        index_reader.read((char*) &max_degree, sizeof(unsigned));
        std::cout << "read " << npts << " vetices, " 
                  << num_edges << " edges, with max weight: "
                  << max_weight << ", max degree: "
                  << max_degree << std::endl;

        offset = sizeof(uint64_t) + sizeof(unsigned);
        index_reader.seekg(offset, std::ios::cur);

        for( unsigned i = 0; i < this->npts_; i++) {
            unsigned degree_index;
            unsigned degree_weights;
            index_reader.read((char*) &degree_index, sizeof(unsigned));
            weight_reader.read((char*) &degree_weights, sizeof(unsigned));
            assert( degree_weights == degree_index);

            this->index_graph_[i].resize( degree_index);
            this->index_weight_[i].resize( degree_weights);
            index_reader.read((char*) this->index_graph_[i].data(), degree_index * sizeof(unsigned));
            weight_reader.read((char*) this->index_weight_[i].data(), degree_weights * sizeof( float));
        }
        index_reader.close();
        weight_reader.close();

        timer.timing("Load index graph");
    }

    void my_kmeans(std::string &base_path, uint32_t *labels,
                size_t n_clusters, size_t max_k_means_reps = 12) {
        my_timer timer;

        // ========= 1. 抽样 + 训练 k-means（保持你原来的逻辑） =========
        size_t train_dim;
        size_t train_size;
        float *train_data;

        // 抽样比例：25600 / npts_
        double p_val = ((double)25600UL / (double)this->npts_);
        gen_random_slice<float>(base_path, p_val, train_data, train_size, train_dim);

        float *centroids = new float[n_clusters * train_dim];
        kmeans::kmeanspp_selecting_pivots(train_data, train_size, train_dim,
                                        centroids, n_clusters);
        kmeans::run_lloyds(train_data, train_size, train_dim,
                        centroids, n_clusters,
                        max_k_means_reps, NULL, NULL);

        delete[] train_data;
        timer.timing("Cluster");

        // ========= 2. 使用 mmap 映射整个 base 文件 =========
        timer.reset();

        int fd = ::open(base_path.c_str(), O_RDONLY);
        if (fd == -1) {
            perror("open base_path failed");
            throw std::runtime_error("open base_path failed");
        }

        struct stat sb;
        if (fstat(fd, &sb) == -1) {
            perror("fstat on base_path failed");
            ::close(fd);
            throw std::runtime_error("fstat on base_path failed");
        }

        size_t file_size = static_cast<size_t>(sb.st_size);

        void *map_ptr = ::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (map_ptr == MAP_FAILED) {
            perror("mmap base_path failed");
            ::close(fd);
            throw std::runtime_error("mmap base_path failed");
        }

        // 文件格式假定为：
        // int npts_in_file;
        // int dim_in_file;
        // float data[npts_in_file * dim_in_file];
        int *header = reinterpret_cast<int *>(map_ptr);
        int npts_in_file = header[0];
        int dim_in_file  = header[1];

        // 简单 sanity check
        assert(static_cast<unsigned>(npts_in_file) == this->npts_);
        assert(static_cast<size_t>(dim_in_file) == train_dim);

        // 跳过两个 int 的头部
        float *base_vectors = reinterpret_cast<float *>(header + 2);

        // ========= 3. 直接在 mmap 上做 assign labels =========
        math_utils::compute_closest_centers(
            base_vectors,
            this->npts_,
            train_dim,
            centroids,
            n_clusters,
            1,
            labels
        );

        // ========= 4. 释放资源 =========
        if (::munmap(map_ptr, file_size) == -1) {
            perror("munmap base_path failed");
        }
        ::close(fd);

        delete[] centroids;
        timer.timing("Get labels (mmap)");
    }


    // void my_kmeans( std::string &base_path, uint32_t *labels,
    //                 size_t n_clusters, size_t max_k_means_reps = 12) {
    //     my_timer timer;
    //     size_t train_dim;
    //     size_t train_size;
    //     float *train_data;
    //     double p_val = ((double) 25600UL / (double) this->npts_);
    //     gen_random_slice<float>(base_path, p_val, train_data, train_size, train_dim);
    //     float *centroids = new float[n_clusters * train_dim];
    //     kmeans::kmeanspp_selecting_pivots(train_data, train_size, train_dim,
    //                                       centroids, n_clusters);
    //     kmeans::run_lloyds(train_data, train_size, train_dim, centroids,
    //                        n_clusters, max_k_means_reps, NULL, NULL);
    //     delete [] train_data;
    //     timer.timing("Cluster");

    //     timer.reset();
    //     std::ifstream base_reader( base_path, std::ios::binary);
    //     base_reader.seekg( 2 * sizeof(int), std::ios::beg);
    //     float *base_vectors = new float [(uint64_t) this->npts_ * train_dim];
    //     base_reader.read((char*) base_vectors, this->npts_ * train_dim * sizeof(float));
    //     timer.timing("Load base vectors");

    //     timer.reset();
    //     math_utils::compute_closest_centers(base_vectors, this->npts_, train_dim, 
    //                                         centroids, n_clusters, 1, labels);
    //     delete [] centroids;
    //     timer.timing("Get labels");
    // }

    void merge_unassigned( cluster **clusters, unsigned n_cluster, 
                           std::vector<unsigned> &ivf, uint32_t *labels) {
        unsigned cnt = 0;
        for( unsigned i = 0; i < n_cluster; i++) {
            cnt += clusters[i]->tmp_cnt_;
        }
        std::cout << cnt << std::endl;
        ivf.reserve( cnt);
        for( unsigned i = 0; i < n_cluster; i++) {
            ivf.insert( ivf.end(),
                        std::make_move_iterator( clusters[i]->unassigned_.begin()),
                        std::make_move_iterator( clusters[i]->unassigned_.end()));
        }
        #pragma omp parallel for
        for( unsigned i = 0; i < cnt; i++) {
            labels[ivf[i]] = n_cluster;
        }
    }

    void concat_one_cluster( cluster *cur_cluster, unsigned &page_id, bool post=false) {
        unsigned theta = post ? 1 : this->page_capacity_;
        std::vector<std::vector<unsigned>> &cur_p2id = cur_cluster->p2id_;
        std::vector<unsigned> &cur_ori_ids = cur_cluster->ori_ids_;
        for( auto &page : cur_p2id) {
            if( page.size() >= theta) {
                for( auto &local_id : page) {
                    this->p2id_[page_id].push_back( cur_ori_ids[local_id]);
                }
                page_id++;
            }
        }
    }

    void concatenate_pages( cluster **clusters, cluster *post_cluster, unsigned n_cluster) {
        this->p2id_.resize( this->n_pages_);
        unsigned page_id = 0;
        for( unsigned i = 0; i < n_cluster; i++) {
            concat_one_cluster( clusters[i], page_id);
        }
        concat_one_cluster( post_cluster, page_id, true);
        std::cout << page_id << std::endl;
        assert( page_id == this->n_pages_);
        this->id2p_.resize( this->npts_);
        #pragma omp parallel for
        for( unsigned i = 0; i < this->n_pages_; i++) {
            for( auto &cur_id : this->p2id_[i]) {
                this->id2p_[cur_id] = i;
            }
        }
    }

    void partition( std::string &base_path, size_t n_clusters) {
        uint32_t *labels;
        labels = new uint32_t [this->npts_];
        my_kmeans( base_path, labels, n_clusters);

        my_timer timer;
        std::vector<std::vector<unsigned>> cluster_ivf( n_clusters);
        std::vector<std::mutex> cluster_locks( n_clusters);
        #pragma omp parallel for
        for( unsigned i = 0; i < this->npts_; i++) {
            uint32_t cluster_id = labels[i];
            std::lock_guard<std::mutex> lock(cluster_locks[cluster_id]);
            cluster_ivf[cluster_id].push_back( i);
        }
        timer.timing("Get cluster ivf");

        timer.reset();
        cluster **clusters = new cluster *[n_clusters];
        unsigned *id_in_cluster = new unsigned [this->npts_];

        timer.timing("Prepare clusters");
        timer.reset();

        #pragma omp parallel for
        for (unsigned i = 0; i < n_clusters; i++) {
            // 1. 构造 cluster
            cluster *cur_cluster = new cluster(cluster_ivf[i], id_in_cluster);

            // 2. 建立定向/无向子图
            cur_cluster->get_direct_graph(this->index_graph_,
                                        this->index_weight_,
                                        labels,
                                        id_in_cluster);
            cur_cluster->trans2undirect();

            // 3. page 内 greedy 布局
            cur_cluster->greedy_layout(this->page_capacity_);

            // 4. 立刻释放这个 cluster 的大图邻接表，保留 ori_ids_ / p2id_ / unassigned_ 等
            cur_cluster->release_graph_memory();

            // 5. 记录指针供后续 merge_unassigned / concatenate_pages 使用
            clusters[i] = cur_cluster;
        }

        timer.timing("Init clusters");

        // timer.reset();
        // cluster **clusters = new cluster *[n_clusters];
        // unsigned *id_in_cluster = new unsigned [this->npts_];
        // for( unsigned i = 0; i < n_clusters; i++) {
        //     cluster *cur_cluster = new cluster( cluster_ivf[i], id_in_cluster);
        //     cur_cluster->get_direct_graph( this->index_graph_, this->index_weight_, 
        //                                    labels, id_in_cluster);
        //     cur_cluster->trans2undirect();
        //     clusters[i] = cur_cluster;
        // }
        // #pragma omp parallel for
        // for( unsigned i = 0; i < n_clusters; i++) {
        //     clusters[i]->greedy_layout( this->page_capacity_);
        // }
        // timer.timing("Init clusters");

        timer.reset();
        std::vector<unsigned> tmp_ivf;
        merge_unassigned( clusters, n_clusters, tmp_ivf, labels);
        cluster *post_process = new cluster( tmp_ivf, id_in_cluster);
        post_process->get_direct_graph( this->index_graph_, this->index_weight_, 
                                        labels, id_in_cluster);
        post_process->trans2undirect();
        post_process->greedy_layout( this->page_capacity_, true);
        timer.timing("Postprocess");
        std::cout << post_process->tmp_cnt_ << std::endl;

        timer.reset();
        concatenate_pages( clusters, post_process, n_clusters);
        timer.timing("Concatenate pages");

        delete post_process;

        #pragma omp parallel for
        for( unsigned i = 0; i < n_clusters; i++) {
            delete clusters[i];
        }
        delete [] clusters;
        delete [] id_in_cluster;
        delete [] labels;

        // trans_and_trunc( 64);
        // get_all_tightness();
        // optimize();
    }

    void trans_and_trunc( unsigned trunc_size) {
        my_timer timer;

        this->undir_graph_.resize( this->npts_);
        this->undir_weight_.resize( this->npts_);
        std::vector<std::vector<unsigned>> append_graph( this->npts_);
        std::vector<std::vector<float>> append_weight( this->npts_);
        std::vector<std::mutex> mtxs( this->npts_);
        #pragma omp parallel for
        for( unsigned i = 0; i < this->npts_; i++) {
            unsigned degree = this->index_graph_[i].size();
            for( unsigned j = 0; j < degree; j++) {
                unsigned nbr = this->index_graph_[i][j];
                auto iter = std::find( this->index_graph_[nbr].begin(),
                                       this->index_graph_[nbr].end(),
                                       i);
                float cur_w = this->index_weight_[i][j];
                if( iter != this->index_graph_[nbr].end()) {
                    cur_w += this->index_weight_[nbr][iter-this->index_graph_[nbr].begin()];
                }
                else {
                    std::lock_guard<std::mutex> lock( mtxs[nbr]);
                    append_graph[nbr].push_back(i);
                    append_weight[nbr].push_back(cur_w);
                }
                std::lock_guard<std::mutex> lock( mtxs[i]);
                this->undir_graph_[i].push_back( nbr);
                this->undir_weight_[i].push_back( cur_w);
            }
        }
        #pragma omp parallel for
        for( unsigned i = 0; i < this->npts_; i++) {
            this->undir_graph_[i].insert( this->undir_graph_[i].end(),
                                          std::make_move_iterator( append_graph[i].begin()),
                                          std::make_move_iterator( append_graph[i].end()));
            this->undir_weight_[i].insert( this->undir_weight_[i].end(),
                                           std::make_move_iterator( append_weight[i].begin()),
                                           std::make_move_iterator( append_weight[i].end()));
        }
        timer.timing("Generate undirected graph");

        timer.reset();
        #pragma omp parallel for
        for( unsigned i = 0; i < this->npts_; i++) {
            unsigned degree = this->undir_graph_[i].size();
            std::priority_queue<std::pair<float, unsigned>,
                                std::vector<std::pair<float, unsigned>>,
                                std::greater<std::pair<float, unsigned>>> pq;
            for( unsigned j = 0; j < degree; j++) {
                float cur_w = this->undir_weight_[i][j];
                unsigned nbr = this->undir_graph_[i][j];
                if( pq.size() < trunc_size) {
                    pq.push( std::make_pair( cur_w, nbr));
                }
                else if( pq.top().first < cur_w) {
                    pq.pop();
                    pq.push( std::make_pair( cur_w, nbr));
                }
            }
            this->undir_graph_[i].clear();
            this->undir_weight_[i].clear();
            this->undir_graph_[i].reserve( pq.size());
            this->undir_weight_[i].reserve( pq.size());
            while( !pq.empty()) {
                this->undir_graph_[i].push_back( pq.top().second);
                this->undir_weight_[i].push_back( pq.top().first);
                pq.pop();
            }
        }
        timer.timing("Truncate graph");
    }

    float get_one_tightness( unsigned i, unsigned page_id, unsigned exclude) {
        float tight = 0.0f;
        for( unsigned &pt_id : this->p2id_[page_id]) {
            if( pt_id == exclude) {
                continue;
            }
            auto iter = std::find( this->undir_graph_[i].begin(),
                                   this->undir_graph_[i].end(),
                                   pt_id);
            if( iter != this->undir_graph_[i].end()) {
                tight += this->undir_weight_[i][iter-this->undir_graph_[i].begin()];
            }
        }
        return tight;
    }

    void get_all_tightness() {
        my_timer timer;
        this->tightness_.resize( this->npts_, 0.0f);
        #pragma omp parallel for
        for( unsigned i = 0; i < this->npts_; i++) {
            this->tightness_[i] = get_one_tightness( i, this->id2p_[i], i);
        }
        timer.timing("Get all tightness");
    }

    void calc_obj() {
        float obj = 0.0f;
        for( unsigned i = 0; i < this->npts_; i++) {
            obj += this->tightness_[i];
        }
        std::cout << "Obj value: " << obj << std::endl;
    }

    void run_iteration() {
        unsigned trunc_size = 0;
        for( unsigned i = 0; i < this->npts_; i++) {
            if( trunc_size < this->undir_graph_[i].size()) {
                trunc_size = this->undir_graph_[i].size();
            }
        }
        std::cout << trunc_size << std::endl;
        //  = this->undir_graph_[0].size();
        std::vector<float> all_profits( trunc_size);
        std::vector<unsigned> all_targets( trunc_size);
        std::vector<float> all_tights_i( trunc_size);
        std::vector<float> all_tights_targets( trunc_size);
        unsigned cnt = 0;
        for( unsigned i = 0; i < this->npts_; i++) {
            unsigned degree = this->undir_graph_[i].size();
            #pragma omp parallel for
            for( unsigned j = 0; j < degree; j++) {
                unsigned nbr = this->undir_graph_[i][j];
                unsigned pid_i = this->id2p_[i];
                unsigned pid_nbr = this->id2p_[nbr];
                if( pid_i == pid_nbr) {
                    all_profits[j] = 0.0f;
                    continue;
                }
                float tight = get_one_tightness( i, pid_nbr, nbr)
                            + this->undir_weight_[i][j];
                unsigned target_id;
                float max_profit = 0.0f, new_tight_i, new_tight_target;
                for( unsigned &pt_id : this->p2id_[pid_nbr]) {
                    if( pt_id == nbr) {
                        continue;
                    }
                    float benefit_i = tight;
                    auto iter_i = std::find( this->undir_graph_[i].begin(),
                                             this->undir_graph_[i].end(),
                                             pt_id);
                    if( iter_i != this->undir_graph_[i].end()) {
                        benefit_i -= this->undir_weight_[i][iter_i-this->undir_graph_[i].begin()];
                    }
                    float benefit_target = get_one_tightness( pt_id, pid_i, i);
                    float profit = benefit_i + benefit_target
                                 - this->tightness_[i] - this->tightness_[pt_id];
                    if( profit > max_profit) {
                        max_profit = profit;
                        target_id = pt_id;
                        new_tight_i = benefit_i;
                        new_tight_target = benefit_target;
                    }
                }
                if( max_profit < 0.1f) {
                    all_profits[j] = 0.0f;
                }
                else {
                    all_profits[j] = max_profit;
                    all_targets[j] = target_id;
                    all_tights_i[j] = new_tight_i;
                    all_tights_targets[j] = new_tight_target;
                }
            }
            float change_profit = 0.0f;
            unsigned change_id;
            float tight_i, tight_change;
            for( unsigned j = 0; j < degree; j++) {
                if( change_profit < all_profits[j]) {
                    change_profit = all_profits[j];
                    change_id = all_targets[j];
                    tight_i = all_tights_i[j];
                    tight_change = all_tights_targets[j];
                }
            }
            if( change_profit > 0.1f) {
                unsigned pid_i = this->id2p_[i];
                unsigned pid_change = this->id2p_[change_id];
                std::swap( this->id2p_[i], this->id2p_[change_id]);
                auto iter_i = std::find( this->p2id_[pid_i].begin(),
                                         this->p2id_[pid_i].end(),
                                         i);
                assert( iter_i != this->p2id_[pid_i].end());
                auto iter_change = std::find( this->p2id_[pid_change].begin(),
                                              this->p2id_[pid_change].end(),
                                              change_id);
                assert( iter_change != this->p2id_[change_id].end());
                this->p2id_[pid_i][iter_i-this->p2id_[pid_i].begin()] = change_id;
                this->p2id_[pid_change][iter_change-this->p2id_[pid_change].begin()] = i;
                this->tightness_[i] = tight_i;
                this->tightness_[change_id] = tight_change;
                cnt++;
            }
        }
        std::cout << cnt << std::endl;
    }

    void optimize( unsigned num_iteration = 4) {
        calc_obj();
        for( unsigned i = 0; i < num_iteration; i++) {
            run_iteration();
            calc_obj();
        }
    }

    void save_partition( std::string &file) {
        std::ofstream partition_writer( file, std::ios::binary);
        uint64_t u64_capacity = this->page_capacity_;
        uint64_t u64_npage = this->n_pages_;
        uint64_t u64_npts = this->npts_;
        partition_writer.write((char*) &u64_capacity, sizeof(uint64_t));
        partition_writer.write((char*) &u64_npage, sizeof(uint64_t));
        partition_writer.write((char*) &u64_npts, sizeof(uint64_t));
        for( unsigned i = 0; i< this->n_pages_; i++) {
            unsigned s = this->p2id_[i].size();
            if( s != this->page_capacity_) {
                std::cout << i << " " << s << std::endl;
            }
            partition_writer.write((char*) &s, sizeof(unsigned));
            partition_writer.write((char*) this->p2id_[i].data(), s * sizeof(unsigned));
        }
        partition_writer.write((char*) this->id2p_.data(), this->npts_ * sizeof(unsigned));
    }

    unsigned npts_;
    unsigned page_capacity_;
    unsigned n_pages_;
    std::vector<std::vector<unsigned>> index_graph_;
    std::vector<std::vector<float>> index_weight_;
    std::vector<std::vector<unsigned>> undir_graph_;
    std::vector<std::vector<float>> undir_weight_;
    std::vector<unsigned> id2p_;
    std::vector<std::vector<unsigned>> p2id_;
    std::vector<float> tightness_;
};