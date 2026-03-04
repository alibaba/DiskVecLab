#include "partitioner.h"
#include <omp.h>
#include <stdlib.h>
#include <boost/program_options.hpp>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

int main(int argc, char **argv) {
  namespace po = boost::program_options;
  std::string index_file, data_type, gp_file, freq_file;
  unsigned block_size, ldg_times, lock_nums, thead_nums, cut;
  bool use_disk, visual;

  po::options_description desc{"Arguments"};
  try {
    desc.add_options()("help,h", "print information");
    desc.add_options()("data_type", po::value<std::string>(&data_type)->required(), "data type <int8/uint8/float>");
    desc.add_options()("index_file", po::value<std::string>(&index_file)->required(),
                       "diskann diskann index or mem index");
    desc.add_options()("gp_file", po::value<std::string>(&gp_file)->required(), "output gp file");
    desc.add_options()("freq_file", po::value<std::string>(&freq_file)->default_value(""), "freq_file[optional]");
    desc.add_options()("thread_nums,T", po::value<unsigned>(&thead_nums)->default_value(omp_get_num_procs()),
                       "threads_nums");
    desc.add_options()("lock_nums", po::value<unsigned>(&lock_nums)->default_value(0),
                       "lock node nums, the lock nodes will not participate in the follow LDG paritioning");
    desc.add_options()("block_size,B", po::value<unsigned>(&block_size)->default_value(1),
                       "block size for one partition, 1 for 4KB, 2 for 8KB and so on.");
    desc.add_options()("ldg_times,L", po::value<unsigned>(&ldg_times)->default_value(4),
                       "exec ldg partition alg times, usually 8 is enough.");
    desc.add_options()("use_disk", po::value<bool>(&use_disk)->default_value(1),
                       "Use 1 for use disk index (default), 0 for DiskANN mem index");
    desc.add_options()("visual", po::value<bool>(&visual)->default_value(0),
                       "see real time progress of graph partition");
    desc.add_options()("cut", po::value<unsigned>(&cut)->default_value(INF), "cut adj list, use 3 means graph degree will be cut to 3");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc;
      return 0;
    }
    po::notify(vm);
  } catch (const std::exception &ex) {
    std::cerr << ex.what() << "\n";
  }
  omp_set_num_threads(thead_nums);
  GP::graph_partitioner partitioner(index_file.c_str(), data_type.c_str(), use_disk, block_size, visual,
                                    freq_file, cut);
  partitioner.graph_partition(gp_file.c_str(), ldg_times, lock_nums);
  return 0;
}
