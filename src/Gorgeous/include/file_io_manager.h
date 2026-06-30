// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "aligned_file_reader.h"

// Multi-fd Linux libaio reader. open() can be called repeatedly and returns
// a fid; submit_read_reqs/submit_write_reqs accept either a per-batch fid or
// per-request fids. Inherits from AlignedFileReader so callers that prefer
// the legacy single-fd surface (read/submit_reqs) can use it transparently
// — those bridges call submit_read_reqs(fid=0).
class FileIOManager : public AlignedFileReader {
 private:
  std::vector<FileHandle> fds;      // file descs
  IOContext               bad_ctx;  // default-constructed = invalid

 public:
  IOContext& get_ctx() override;

  ~FileIOManager() override;

  void register_thread() override;
  void deregister_thread() override;
  void deregister_all_threads() override;

  // Multi-fd open: every successful call appends a new fd and returns its fid.
  int  open(const std::string& uri, int flags = O_RDONLY) override;
  void close() override;

  // Multi-fd interface
  int submit_read_reqs(std::vector<AlignedRead>& read_reqs,
                       std::vector<int>& fids, IOContext& ctx) override;
  int submit_read_reqs(std::vector<AlignedRead>& read_reqs, int fid,
                       IOContext& ctx) override;
  int submit_write_reqs(std::vector<AlignedWrite>& write_reqs,
                        std::vector<int>& fids, IOContext& ctx) override;

  // Single-fd bridges expected by AlignedFileReader. Both forward to
  // submit_read_reqs(reqs, 0, ctx); read() blocks via execute_io.
  void read(std::vector<AlignedRead>& read_reqs, IOContext& ctx,
            bool async = false) override;
  int  submit_reqs(std::vector<AlignedRead>& read_reqs, IOContext& ctx) override;

  // Multi-fd convenience kept for legacy callers (DecoIndex via fid args).
  void read(std::vector<AlignedRead>& read_reqs, IOContext& ctx, int fid);

  void get_events(IOContext& ctx, int n_ops) override;
  int  get_events(IOContext& ctx, int min_r, int max_r,
                  std::vector<char*>& ofts) override;
};
