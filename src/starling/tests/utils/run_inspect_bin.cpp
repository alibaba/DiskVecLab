// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include "utils.h"

template<typename T>
void print_data(std::string& bin_file, T*& data, size_t& npts, size_t& dim, size_t K) {
  std::cout << "Data contents: " << bin_file << std::endl;
  std::cout << "Points: " << npts << std::endl;
  std::cout << "Dim: " << dim << std::endl;
  std::cout << "TOP K: " << K << std::endl;
  for (size_t i = 0; i < K; ++i) {
    std::cout << "| ";
    for (size_t j = 0; j < dim; ++j) {
      std::cout << data[i * dim + j] << " ";
    }
    std::cout << " |" << std::endl;
  }
  std::cout << "End of data contents." << std::endl;
}

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cout << argv[0] << " data_type input_bin K" << std::endl;
    exit(-1);
  }

  std::string data_type = argv[1];
  std::string input_bin = argv[2];
  size_t K = atoi(argv[3]);
  size_t    npts, nd;
  if (data_type == std::string("float")) {
    float* input;
    diskann::load_bin<float>(input_bin, input, npts, nd);
    print_data<float>(input_bin, input, npts, nd, K);
    delete[] input;
  }
  if (data_type == std::string("uint32_t")) {
    uint32_t* input;
    diskann::load_bin<uint32_t>(input_bin, input, npts, nd);
    print_data<uint32_t>(input_bin, input, npts, nd, K);
    delete[] input;
  }
}
