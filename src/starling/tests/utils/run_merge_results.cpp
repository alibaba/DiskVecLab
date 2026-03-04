#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <experimental/filesystem>
#include <algorithm>
#include <stdexcept>
#include <utility>
#include "utils.h"

namespace fs = std::experimental::filesystem;

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <input_dir> <output_prefix> <K>" << std::endl;
        return -1;
    }

    std::string input_dir = argv[1];
    std::string output_prefix = argv[2];
    size_t K = std::stoul(argv[3]);

    std::vector<std::pair<std::string, std::string>> file_pairs; // {dist_file, id_file}

    // Step 1: collect file pair
    for (const auto& entry : fs::directory_iterator(input_dir)) {
        if (!fs::is_directory(entry.path())) continue;

        std::string dist_file, id_file;
        bool found_dist = false, found_id = false;

        for (const auto& sub_entry : fs::directory_iterator(entry.path())) {
            if (!fs::is_regular_file(sub_entry.path())) continue;
            std::string filename = sub_entry.path().filename().string();
            if (filename.find("_dists_float.bin") != std::string::npos) {
                dist_file = sub_entry.path().string();
                found_dist = true;
            } else if (filename.find("_idx_uint32.bin") != std::string::npos) {
                id_file = sub_entry.path().string();
                found_id = true;
            }
        }

        if (found_dist && found_id) {
            file_pairs.push_back({dist_file, id_file});
        } else {
            std::cerr << "Warning: Subdirectory " << entry.path().string() << " does not contain both _dists_float.bin and _idx_uint32.bin files." << std::endl;
        }
    }

    if (file_pairs.empty()) {
        std::cerr << "No valid _dists_float.bin and _idx_uint32.bin file pairs found in subdirectories of: " << input_dir << std::endl;
        return -1;
    }

    std::vector<std::pair<float*, uint32_t*>> data_ptrs;
    std::vector<std::pair<size_t, size_t>> dims; // {dist_dim, id_dim}
    size_t npts = 0;

    // Step 2: read all data
    for (const auto& [dis_file, id_file] : file_pairs) {
        float* dist_data = nullptr;
        uint32_t* id_data = nullptr;
        size_t dist_npts = 0, dist_dim = 0, id_npts = 0, id_dim = 0;

        diskann::load_bin<float>(dis_file, dist_data, dist_npts, dist_dim);
        diskann::load_bin<uint32_t>(id_file, id_data, id_npts, id_dim);

        if (dist_npts != id_npts || dist_dim != id_dim) {
            std::cerr << "File pair mismatch: " << dis_file << " and " << id_file << std::endl;
            return -1;
        }

        if (npts == 0) {
            npts = dist_npts;
        } else if (npts != dist_npts) {
            std::cerr << "All files must have the same number of queries (npts)" << std::endl;
            return -1;
        }

        data_ptrs.push_back({dist_data, id_data});
        dims.push_back({dist_dim, id_dim});
    }

    // Step 3: malloc
    size_t total_size = npts * K;
    float* output_dists = new float[total_size];
    uint32_t* output_ids = new uint32_t[total_size];

    // Step 4: get top-K
    for (size_t i = 0; i < npts; ++i) {
        std::vector<std::pair<float, uint32_t>> merged;

        for (size_t f = 0; f < data_ptrs.size(); ++f) {
            float* dist_data = data_ptrs[f].first + i * dims[f].first;
            uint32_t* id_data = data_ptrs[f].second + i * dims[f].second;

            for (size_t j = 0; j < dims[f].first; ++j) {
                merged.emplace_back(dist_data[j], id_data[j]);
            }
        }

        // sort
        std::sort(merged.begin(), merged.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        // get top K
        size_t copy_count = std::min(K, merged.size());
        for (size_t k = 0; k < copy_count; ++k) {
            output_dists[i * K + k] = merged[k].first;
            output_ids[i * K + k] = merged[k].second;
        }
        // fill empty
        for (size_t k = copy_count; k < K; ++k) {
            output_dists[i * K + k] = 0.0f;
            output_ids[i * K + k] = 0;
        }
    }

    diskann::save_bin<uint32_t>(output_prefix + "_idx_uint32.bin", output_ids, npts, K);
    diskann::save_bin<float>(output_prefix + "_dists_float.bin", output_dists, npts, K);

    // Step 6: clean
    for (auto& [dist_data, id_data] : data_ptrs) {
        delete[] dist_data;
        delete[] id_data;
    }
    delete[] output_dists;
    delete[] output_ids;

    std::cout << "Output files saved to: " << output_prefix << "_idx_uint32.bin and " << output_prefix << "_dists_float.bin" << std::endl;
    return 0;
}
