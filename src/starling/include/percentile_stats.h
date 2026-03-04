// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <algorithm>
#ifdef _WINDOWS
#include <numeric>
#endif
#include <string>
#include <vector>

#include <atomic>

#include "distance.h"
#include "parameters.h"

namespace diskann {
  struct QueryStats
  {
      std::atomic<long long> total_us = 0; // total time to process query in micros
      std::atomic<long long> io_us = 0;    // total time spent in IO
      std::atomic<long long> cpu_us = 0;   // total time spent in CPU

      std::atomic<unsigned> n_4k = 0;         // # of 4kB reads
      std::atomic<unsigned> n_8k = 0;         // # of 8kB reads
      std::atomic<unsigned> n_12k = 0;        // # of 12kB reads
      std::atomic<unsigned> n_ios = 0;        // total # of IOs issued
      std::atomic<unsigned> read_size = 0;    // total # of bytes read
      std::atomic<unsigned> n_cmps_saved = 0; // # cmps saved
      std::atomic<unsigned> n_cmps = 0;       // # cmps
      std::atomic<unsigned> n_cache_hits = 0; // # cache_hits
      std::atomic<unsigned> n_hops = 0;       // # search hops
  };

  template<typename T>
  inline T get_percentile_stats(
      QueryStats *stats, uint64_t len, float percentile,
      const std::function<T(const QueryStats &)> &member_fn) {
    std::vector<T> vals(len);
    for (uint64_t i = 0; i < len; i++) {
      vals[i] = member_fn(stats[i]);
    }

    std::sort(vals.begin(), vals.end(),
              [](const T &left, const T &right) { return left < right; });

    auto retval = vals[(uint64_t)(percentile * len)];
    vals.clear();
    return retval;
  }

  template<typename T>
  inline double get_mean_stats(
      QueryStats *stats, uint64_t len,
      const std::function<T(const QueryStats &)> &member_fn) {
    double avg = 0;
    for (uint64_t i = 0; i < len; i++) {
      avg += (double) member_fn(stats[i]);
    }
    return avg / len;
  }

  // The following two functions are used when getting statistics while range searching on only queries with
  // non-zero gt lengths
  template<typename T>
  inline T get_percentile_stats_gt(
      QueryStats *stats, uint64_t len, float percentile,
      const std::function<T(const QueryStats &)> &member_fn, std::vector<std::vector<uint32_t>> &gt) {
    std::vector<T> vals;
    for (uint64_t i = 0; i < len; i++) {
      if (gt[i].size()) vals.push_back(member_fn(stats[i]));
    }

    std::sort(vals.begin(), vals.end(),
              [](const T &left, const T &right) { return left < right; });

    auto retval = vals[(uint64_t)(percentile * vals.size())];
    vals.clear();
    return retval;
  }

  template<typename T>
  inline double get_mean_stats_gt(
      QueryStats *stats, uint64_t len,
      const std::function<T(const QueryStats &)> &member_fn, std::vector<std::vector<uint32_t>> &gt) {
    uint32_t cnt = 0;
    double avg = 0;
    for (uint64_t i = 0; i < len; i++) {
      if (gt[i].size()) {
        ++cnt;
        avg += (double) member_fn(stats[i]);
      }
    }
    return avg / cnt;
  }
}  // namespace diskann
