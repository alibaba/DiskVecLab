#include "data_partition.h"

int main(int argc, char** argv) {
  if (argc < 5) {
    std::cerr << "Usage: " << argv[0] << " <partition_type> <base_file> <output_prefix> <num_shards> [data_type]" << std::endl;
    std::cerr << "  partition_type: natural, knn" << std::endl;
    std::cerr << "  data_type (for natural): fvecs, bvecs" << std::endl;
    return -1;
  }

  std::string partition_type = std::string(argv[1]);
  std::string base_file = std::string(argv[2]);
  std::string output_prefix = std::string(argv[3]);
  int num_shards = std::atoi(argv[4]);

  try {
    if (partition_type == std::string("natural")) {
      if (argc < 6) {
        std::cerr << "Data type required for natural partitioning" << std::endl;
        return -1;
      }
      std::string data_type = std::string(argv[5]);
      if (data_type == std::string("fvecs")) {
        diskann::natural_split<float>(base_file, output_prefix, num_shards);
      } else if (data_type == std::string("bvecs")) {
        diskann::natural_split<uint8_t>(base_file, output_prefix, num_shards);
      } else {
        std::cerr << "Unsupported data type: " << data_type << std::endl;
        return -1;
      }
    } else if (partition_type == std::string("knn")) {
      if (argc < 8) {
        std::cerr << "knn requires additional parameters" << std::endl;
        return -1;
      }
      diskann::knn_split(std::string(argv[5]), base_file, output_prefix, atof(argv[6]), (size_t) std::atoi(argv[7]), (size_t) std::atoi(argv[8]));
    } else {
      diskann::cerr << "Error. Unsupported partition type: " << partition_type <<  std::endl;
      return -1;
    }
    return 0;
  } catch (const std::exception& e) {
    std::cout << std::string(e.what()) << std::endl;
    diskann::cerr << "Data partition failed." << std::endl;
    return -1;
  }
}
