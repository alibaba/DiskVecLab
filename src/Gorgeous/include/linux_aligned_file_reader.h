// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#ifndef _WINDOWS

#include "aligned_file_reader.h"

class LinuxAlignedFileReader : public AlignedFileReader {
 private:
  uint64_t   file_sz;
  FileHandle file_desc;
  IOContext  bad_ctx;  // default-constructed = invalid (impl == nullptr)

 public:
  LinuxAlignedFileReader();
  ~LinuxAlignedFileReader();

  IOContext &get_ctx();

  // register thread-id for a context
  void register_thread();

  // de-register thread-id for a context
  void deregister_thread();
  void deregister_all_threads();

  // Open & close ops
  // Blocking calls. Single-fd backend: only the first open() takes effect.
  int  open(const std::string &uri, int flags = O_RDONLY) override;
  void close() override;

  // Single-fd interface
  void read(std::vector<AlignedRead> &read_reqs, IOContext &ctx,
            bool async = false) override;
  int  submit_reqs(std::vector<AlignedRead> &read_reqs, IOContext &ctx) override;

  // Multi-fd bridges: fid must be 0; otherwise throw ANNException.
  int submit_read_reqs(std::vector<AlignedRead> &read_reqs, int fid,
                       IOContext &ctx) override;
  int submit_read_reqs(std::vector<AlignedRead> &read_reqs,
                       std::vector<int> &fids, IOContext &ctx) override;
  int submit_write_reqs(std::vector<AlignedWrite> &write_reqs,
                        std::vector<int> &fids, IOContext &ctx) override;

  void get_events(IOContext &ctx, int n_ops) override;
  int  get_events(IOContext &ctx, int min_r, int max_r,
                  std::vector<char *> &ofts) override;
};

#endif
