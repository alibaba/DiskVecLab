// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#ifdef _WINDOWS
#ifndef USE_BING_INFRA
#include <Windows.h>
#include <fcntl.h>
#include <malloc.h>
#include <minwinbase.h>

#include <cstdio>
#include <mutex>
#include <thread>
#include "aligned_file_reader.h"
#include "tsl/robin_map.h"
#include "utils.h"
#include "windows_customizations.h"

// Backend-specific payload stored inside IOContext::impl.
struct WindowsIOContextImpl {
  HANDLE                  fhandle = NULL;
  HANDLE                  iocp = NULL;
  std::vector<OVERLAPPED> reqs;
};

class WindowsAlignedFileReader : public AlignedFileReader {
 private:
  std::wstring m_filename;

 protected:
  // virtual IOContext createContext();

 public:
  DISKANN_DLLEXPORT WindowsAlignedFileReader(){};
  DISKANN_DLLEXPORT virtual ~WindowsAlignedFileReader(){};

  // Open & close ops
  // Blocking calls. Single-fd backend: only the first open() takes effect.
  DISKANN_DLLEXPORT int  open(const std::string &uri,
                              int                flags = O_RDONLY) override;
  DISKANN_DLLEXPORT void close() override;

  DISKANN_DLLEXPORT void register_thread() override;
  DISKANN_DLLEXPORT void deregister_thread() override {
    // TODO: Needs implementation.
  }
  DISKANN_DLLEXPORT void deregister_all_threads() override {
    // TODO: Needs implementation.
  }
  DISKANN_DLLEXPORT IOContext &get_ctx() override;

  // Single-fd interface
  DISKANN_DLLEXPORT void read(std::vector<AlignedRead> &read_reqs,
                              IOContext &ctx, bool async = false) override;
  DISKANN_DLLEXPORT int  submit_reqs(std::vector<AlignedRead> &read_reqs,
                                     IOContext &               ctx) override;

  // Multi-fd bridges (fid must be 0; otherwise throws).
  DISKANN_DLLEXPORT int submit_read_reqs(std::vector<AlignedRead> &read_reqs,
                                         int fid, IOContext &ctx) override;
  DISKANN_DLLEXPORT int submit_read_reqs(std::vector<AlignedRead> &read_reqs,
                                         std::vector<int> &        fids,
                                         IOContext &               ctx) override;
  DISKANN_DLLEXPORT int submit_write_reqs(std::vector<AlignedWrite> &write_reqs,
                                          std::vector<int> &         fids,
                                          IOContext &                ctx) override;

  DISKANN_DLLEXPORT void get_events(IOContext &ctx, int n_ops) override;
  DISKANN_DLLEXPORT int  get_events(IOContext &ctx, int min_r, int max_r,
                                    std::vector<char *> &ofts) override;
};
#endif  // USE_BING_INFRA
#endif  //_WINDOWS
