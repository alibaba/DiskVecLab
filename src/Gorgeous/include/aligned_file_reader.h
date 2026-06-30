// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#define MAX_IO_DEPTH 128

#include <vector>
#include <atomic>

#ifndef _WINDOWS
#include <fcntl.h>
#include <libaio.h>
#include <unistd.h>
// Opaque per-thread I/O context. Backends store their own state in `impl`
// (libaio: io_context_t cast to void*; S3: pointer to a per-thread completion
// queue). Default-constructed instances are invalid.
struct IOContext {
  void* impl = nullptr;
  bool  valid() const { return impl != nullptr; }
};
#else
#include <Windows.h>
#include <minwinbase.h>

#ifndef USE_BING_INFRA
// Same opaque struct as the Linux branch. The Windows backend stores a
// pointer to its own WindowsIOContextImpl (defined alongside the reader).
struct IOContext {
  void* impl = nullptr;
  bool  valid() const { return impl != nullptr; }
};
#else
#include "IDiskPriorityIO.h"
#include <atomic>
// TODO: Caller code is very callous about copying IOContext objects
// all over the place. MUST verify that it won't cause leaks/logical
// errors.
// Because of such callous copying, we have to use ptr->atomic instead
// of atomic, as atomic is not copyable.
struct IOContext {
  enum Status { READ_WAIT = 0, READ_SUCCESS, READ_FAILED, PROCESS_COMPLETE };

  std::shared_ptr<ANNIndex::IDiskPriorityIO>               m_pDiskIO = nullptr;
  std::shared_ptr<std::vector<ANNIndex::AsyncReadRequest>> m_pRequests;
  std::shared_ptr<std::vector<Status>>                     m_pRequestsStatus;

  IOContext()
      : m_pRequestsStatus(new std::vector<Status>()),
        m_pRequests(new std::vector<ANNIndex::AsyncReadRequest>()) {
    (*m_pRequestsStatus).reserve(MAX_IO_DEPTH);
    (*m_pRequests).reserve(MAX_IO_DEPTH);
  }
};
#endif

#endif

#include <malloc.h>
#include <cstdio>
#include <mutex>
#include <thread>
#include "tsl/robin_map.h"
#include "utils.h"

// NOTE :: all 3 fields must be 512-aligned
struct AlignedRead {
  uint64_t offset;  // where to read from
  uint64_t len;     // how much to read
  void*    buf;     // where to read into

  AlignedRead() : offset(0), len(0), buf(nullptr) {
  }

  AlignedRead(uint64_t offset, uint64_t len, void* buf)
      : offset(offset), len(len), buf(buf) {
    assert(IS_512_ALIGNED(offset));
    assert(IS_512_ALIGNED(len));
    assert(IS_512_ALIGNED(buf));
    // assert(malloc_usable_size(buf) >= len);
  }
};

// NOTE :: all 3 fields must be 512-aligned
// actually this structure is the same as Read
struct AlignedWrite {
  uint64_t offset;  // where to write
  uint64_t len;     // how much to write
  void*    buf;     // where to write from

  AlignedWrite() : offset(0), len(0), buf(nullptr) {
  }

  AlignedWrite(uint64_t offset, uint64_t len, void* buf)
      : offset(offset), len(len), buf(buf) {
    assert(IS_512_ALIGNED(offset));
    assert(IS_512_ALIGNED(len));
    assert(IS_512_ALIGNED(buf));
  }
};

// Common base class for all aligned-block I/O backends. Two flavours of
// usage live behind one interface:
//   * Single-fd (legacy AlignedFileReader: PQFlashIndex / starling) — call
//     open() once, then read()/submit_reqs()/get_events().
//   * Multi-fd (legacy FileIOManager: DecoIndex / gorgeous) — call open()
//     multiple times collecting fids, then submit_read_reqs(reqs,fid,ctx)
//     / submit_read_reqs(reqs,fids,ctx) / submit_write_reqs / get_events.
//
// Backends that implement only one flavour bridge the other to the
// implemented one (single-fd reads default fid=0; multi-fd reads with fid=0
// behave like single-fd). Calls that genuinely make no sense for a backend
// (e.g. submit_write_reqs on a read-only AIO reader) throw ANNException.
class AlignedFileReader {
 protected:
  tsl::robin_map<std::thread::id, IOContext> ctx_map;
  std::mutex                                 ctx_mut;

 public:
  virtual ~AlignedFileReader(){};

  virtual IOContext& get_ctx() = 0;

  virtual void register_thread() = 0;
  virtual void deregister_thread() = 0;
  virtual void deregister_all_threads() = 0;

  // Open the URI for I/O; returns a file id (fid). Single-fd backends
  // ignore subsequent calls and return 0. Multi-fd backends return a new
  // fid each call.
  virtual int  open(const std::string& uri, int flags = O_RDONLY) = 0;
  virtual void close() = 0;

  // Single-fd blocking read: reads against the most recently opened fd.
  virtual void read(std::vector<AlignedRead>& read_reqs, IOContext& ctx,
                    bool async = false) = 0;

  // Single-fd async submit (fid is implicit / fid=0).
  virtual int submit_reqs(std::vector<AlignedRead>& read_reqs,
                          IOContext& ctx) = 0;

  // Multi-fd async submit: every request reads from the given fid, or from
  // the per-request fids[i].
  virtual int submit_read_reqs(std::vector<AlignedRead>& read_reqs, int fid,
                               IOContext& ctx) = 0;
  virtual int submit_read_reqs(std::vector<AlignedRead>& read_reqs,
                               std::vector<int>&         fids,
                               IOContext&                ctx) = 0;
  virtual int submit_write_reqs(std::vector<AlignedWrite>& write_reqs,
                                std::vector<int>&          fids,
                                IOContext&                 ctx) = 0;

  virtual void get_events(IOContext& ctx, int n_ops) = 0;
  virtual int  get_events(IOContext& ctx, int min_r, int max_r,
                          std::vector<char*>& ofts) = 0;
};
