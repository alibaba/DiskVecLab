// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "file_io_manager.h"

#include <cassert>
#include <cstdio>
#include <iostream>
#include "tsl/robin_map.h"
#include "utils.h"
#define MAX_EVENTS 1024

namespace {
  typedef struct io_event io_event_t;
  typedef struct iocb     iocb_t;

  inline io_context_t to_aio(const IOContext &ctx) {
    return reinterpret_cast<io_context_t>(ctx.impl);
  }

  void execute_io(io_context_t ctx, int fd, std::vector<AlignedRead> &read_reqs,
                  uint64_t n_retries = 0) {
#ifdef DEBUG
    for (auto &req : read_reqs) {
      assert(IS_ALIGNED(req.len, 512));
      // std::cout << "request:"<<req.offset<<":"<<req.len << std::endl;
      assert(IS_ALIGNED(req.offset, 512));
      assert(IS_ALIGNED(req.buf, 512));
      // assert(malloc_usable_size(req.buf) >= req.len);
    }
#endif
    // break-up requests into chunks of size MAX_EVENTS each
    uint64_t n_iters = ROUND_UP(read_reqs.size(), MAX_EVENTS) / MAX_EVENTS;
    for (uint64_t iter = 0; iter < n_iters; iter++) {
      uint64_t n_ops =
          std::min((uint64_t) read_reqs.size() - (iter * MAX_EVENTS),
                   (uint64_t) MAX_EVENTS);
      std::vector<iocb_t *>    cbs(n_ops, nullptr);
      std::vector<io_event_t>  evts(n_ops);
      std::vector<struct iocb> cb(n_ops);
      for (uint64_t j = 0; j < n_ops; j++) {
        io_prep_pread(cb.data() + j, fd, read_reqs[j + iter * MAX_EVENTS].buf,
                      read_reqs[j + iter * MAX_EVENTS].len,
                      read_reqs[j + iter * MAX_EVENTS].offset);
      }
      // initialize `cbs` using `cb` array
      //
      for (uint64_t i = 0; i < n_ops; i++) {
        cbs[i] = cb.data() + i;
      }
      uint64_t n_tries = 0;
      while (n_tries <= n_retries) {
        // issue reads
        int64_t ret = io_submit(ctx, (int64_t) n_ops, cbs.data());
        // if requests didn't get accepted
        if (ret != (int64_t) n_ops) {
          std::cerr << "io_submit() failed; returned " << ret
                    << ", expected=" << n_ops << ", ernno=" << errno << "="
                    << ::strerror(-ret) << ", try #" << n_tries + 1;
          std::cout << "ctx: " << ctx << "\n";
          exit(-1);
        } else {
          // wait on io_getevents
          ret = io_getevents(ctx, (int64_t) n_ops, (int64_t) n_ops, evts.data(),
                             nullptr);
          // if requests didn't complete
          if (ret != (int64_t) n_ops) {
            std::cerr << "io_getevents() failed; returned " << ret
                      << ", expected=" << n_ops << ", ernno=" << errno << "="
                      << ::strerror(-ret) << ", try #" << n_tries + 1;
            exit(-1);
          } else {
            break;
          }
        }
      }
      // disabled since req.buf could be an offset into another buf
      /*
      for (auto &req : read_reqs) {
        // corruption check
        assert(malloc_usable_size(req.buf) >= req.len);
      }
      */
    }
  }
}  // namespace

FileIOManager::~FileIOManager() {
  for (auto fd : this->fds) {
    int64_t ret;
    // check to make sure file_desc is closed
    ret = ::fcntl(fd, F_GETFD);
    if (ret == -1) {
      if (errno != EBADF) {
        std::cerr << "close() not called" << std::endl;
        // close file desc
        ret = ::close(fd);
        // error checks
        if (ret == -1) {
          std::cerr << "close() failed; returned " << ret << ", errno=" << errno
                    << ":" << ::strerror(errno) << std::endl;
        }
      }
    }
  }
}

IOContext &FileIOManager::get_ctx() {
  std::unique_lock<std::mutex> lk(ctx_mut);
  if (ctx_map.find(std::this_thread::get_id()) == ctx_map.end()) {
    std::cerr << "bad thread access; returning invalid IOContext" << std::endl;
    return this->bad_ctx;
  } else {
    return ctx_map[std::this_thread::get_id()];
  }
}

void FileIOManager::register_thread() {
  auto                         my_id = std::this_thread::get_id();
  std::unique_lock<std::mutex> lk(ctx_mut);
  if (ctx_map.find(my_id) != ctx_map.end()) {
    std::cerr << "multiple calls to register_thread from the same thread"
              << std::endl;
    return;
  }
  io_context_t aio_ctx = 0;
  int          ret = io_setup(MAX_EVENTS, &aio_ctx);
  if (ret != 0) {
    lk.unlock();
    assert(errno != EAGAIN);
    assert(errno != ENOMEM);
    std::cerr << "io_setup() failed; returned " << ret << ", errno=" << errno
              << ":" << ::strerror(errno) << std::endl;
  } else {
    diskann::cout << "allocating ctx: " << aio_ctx
                  << " to thread-id:" << my_id << std::endl;
    IOContext ioc;
    ioc.impl = reinterpret_cast<void *>(aio_ctx);
    ctx_map[my_id] = ioc;
  }
  lk.unlock();
}

void FileIOManager::deregister_thread() {
  auto                         my_id = std::this_thread::get_id();
  std::unique_lock<std::mutex> lk(ctx_mut);
  assert(ctx_map.find(my_id) != ctx_map.end());

  lk.unlock();
  IOContext &ctx = this->get_ctx();
  io_destroy(to_aio(ctx));
  lk.lock();
  ctx_map.erase(my_id);
  std::cerr << "returned ctx from thread-id:" << my_id << std::endl;
  lk.unlock();
}

void FileIOManager::deregister_all_threads() {
  std::unique_lock<std::mutex> lk(ctx_mut);
  for (auto x = ctx_map.begin(); x != ctx_map.end(); x++) {
    io_destroy(to_aio(x.value()));
  }
  ctx_map.clear();
}

int FileIOManager::open(const std::string &uri, int flags) {
  int file_desc;
  if (O_CREAT & flags) {
    file_desc = ::open(uri.c_str(), flags, S_IRUSR | S_IWUSR);
  } else {
    file_desc = ::open(uri.c_str(), flags);
  }
  // error checks
  assert(file_desc != -1);
  if (file_desc == -1) {
    std::cerr << "Failed to open file : " << uri << std::endl;
  }
  this->fds.push_back(file_desc);
  return this->fds.size() - 1;
}

void FileIOManager::close() {
  for (auto fd : this->fds) {
    // check to make sure file_desc is closed
    ::fcntl(fd, F_GETFD);

    ::close(fd);
  }
}

void FileIOManager::read(std::vector<AlignedRead> &read_reqs,
                         IOContext &ctx, int fid) {
  // assume all read from the same place
  assert(this->fds[fid] != -1);
  execute_io(to_aio(ctx), this->fds[fid], read_reqs);
}

// Single-fd bridges expected by AlignedFileReader: route to fid=0.
void FileIOManager::read(std::vector<AlignedRead> &read_reqs, IOContext &ctx,
                         bool /*async*/) {
  this->read(read_reqs, ctx, /*fid=*/0);
}

int FileIOManager::submit_reqs(std::vector<AlignedRead> &read_reqs,
                               IOContext &ctx) {
  return this->submit_read_reqs(read_reqs, /*fid=*/0, ctx);
}

int FileIOManager::submit_read_reqs(std::vector<AlignedRead> &read_reqs,
                                    std::vector<int> &fids,
                                    IOContext &ctx) {
  if (read_reqs.size() > MAX_EVENTS) {
    std::cerr << "The number of requests should not exceed " << MAX_EVENTS << std::endl;
    exit(-1);
  }
  int n_ops = read_reqs.size();
  std::vector<iocb_t *>    cbs(n_ops, nullptr);
  std::vector<struct iocb> cb(n_ops);
  for (int j = 0; j < n_ops; j++) {
    io_prep_pread(cb.data() + j, this->fds[fids[j]], read_reqs[j].buf,
                  read_reqs[j].len,
                  read_reqs[j].offset);
  }
  for (int i = 0; i < n_ops; i++) {
    cb[i].data = (void*) read_reqs[i].buf;
    cbs[i] = cb.data() + i;
  }

  io_context_t aio_ctx = to_aio(ctx);
  int ret = io_submit(aio_ctx, (int64_t) n_ops, cbs.data());
  if (ret != n_ops) {
    std::cerr << "io_submit() failed; returned " << ret
              << ", expected=" << n_ops << ", ernno=" << errno << "="
              << ::strerror(-ret);
    std::cout << "ctx: " << aio_ctx << "\n";
    exit(-1);
  }
  return n_ops;
}

int FileIOManager::submit_read_reqs(std::vector<AlignedRead> &read_reqs,
                                    int fid,
                                    IOContext &ctx) {
  if (read_reqs.size() > MAX_EVENTS) {
    std::cerr << "The number of requests should not exceed " << MAX_EVENTS << std::endl;
    exit(-1);
  }
  int n_ops = read_reqs.size();
  std::vector<iocb_t *>    cbs(n_ops, nullptr);
  std::vector<struct iocb> cb(n_ops);
  for (int j = 0; j < n_ops; j++) {
    io_prep_pread(cb.data() + j, this->fds[fid], read_reqs[j].buf,
                  read_reqs[j].len,
                  read_reqs[j].offset);
  }
  for (int i = 0; i < n_ops; i++) {
    cb[i].data = (void*) read_reqs[i].buf;
    cbs[i] = cb.data() + i;
  }

  io_context_t aio_ctx = to_aio(ctx);
  int ret = io_submit(aio_ctx, (int64_t) n_ops, cbs.data());
  if (ret != n_ops) {
    std::cerr << "io_submit() failed; returned " << ret
              << ", expected=" << n_ops << ", ernno=" << errno << "="
              << ::strerror(-ret);
    std::cout << "ctx: " << aio_ctx << "\n";
    exit(-1);
  }
  return n_ops;
}

int FileIOManager::submit_write_reqs(std::vector<AlignedWrite> &write_reqs,
                                     std::vector<int> &fids,
                                     IOContext &ctx) {
  if (write_reqs.size() > MAX_EVENTS) {
    std::cerr << "The number of requests should not exceed " << MAX_EVENTS << std::endl;
    exit(-1);
  }
  int n_ops = write_reqs.size();
  std::vector<iocb_t *>    cbs(n_ops, nullptr);
  std::vector<struct iocb> cb(n_ops);
  for (int j = 0; j < n_ops; j++) {
    io_prep_pwrite(cb.data() + j, this->fds[fids[j]], write_reqs[j].buf,
                  write_reqs[j].len,
                  write_reqs[j].offset);
  }
  for (int i = 0; i < n_ops; i++) {
    cb[i].data = (void*) write_reqs[i].buf;
    cbs[i] = cb.data() + i;
  }

  io_context_t aio_ctx = to_aio(ctx);
  int ret = io_submit(aio_ctx, (int64_t) n_ops, cbs.data());
  if (ret != n_ops) {
    std::cerr << "io_submit() failed; returned " << ret
              << ", expected=" << n_ops << ", ernno=" << errno << "="
              << ::strerror(-ret);
    std::cout << "ctx: " << aio_ctx << "\n";
    exit(-1);
  }
  return n_ops;
}

void FileIOManager::get_events(IOContext& ctx, int n_ops) {
  std::vector<io_event_t> evts(n_ops);
  auto ret = io_getevents(to_aio(ctx), (int64_t) n_ops, (int64_t) n_ops,
                          evts.data(), nullptr);
  if (ret != (int64_t) n_ops) {
    std::cerr << "io_getevents() failed; returned " << ret;
    exit(-1);
  }
}

int FileIOManager::get_events(IOContext& ctx, int min_r, int max_r, std::vector<char*>& ofts) {
  std::vector<io_event_t> evts(max_r);
  int64_t ret;
  io_context_t aio_ctx = to_aio(ctx);
  if (min_r != 0) {
    ret = io_getevents(aio_ctx, (int64_t) min_r, (int64_t) max_r, evts.data(), nullptr);
  }
  else {
    struct timespec ts = { 0, 0 } ;
    ret = io_getevents(aio_ctx, (int64_t) min_r, (int64_t) max_r, evts.data(), &ts);
  }
  if (ret < min_r) {
    std::cerr << "io_getevents() failed; returned " << ret;
    exit(-1);
  }
  for (int i = 0; i < ret; i++) {
    if (evts[i].res2 != 0) {
      std::cerr << "io_getevents() failed; returned " << ret;
      exit(-1);
    }
    ofts[i] = (char*) evts[i].data;
  }
  return ret;
}
