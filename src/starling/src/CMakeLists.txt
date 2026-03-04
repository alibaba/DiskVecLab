# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

set(CMAKE_CXX_STANDARD 17)

if(MSVC)
    add_subdirectory(dll)
else()
    #file(GLOB CPP_SOURCES *.cpp)
    set(CPP_SOURCES ann_exception.cpp aux_utils.cpp distance.cpp index.cpp
        linux_aligned_file_reader.cpp math_utils.cpp natural_number_map.cpp
        natural_number_set.cpp memory_mapper.cpp partition_and_pq.cpp data_partition.cpp
        pq_flash_index.cpp logger.cpp utils.cpp page_search.cpp visit_freq.cpp
        range_search.cpp rabit_quantizer.cpp fixed_chunk_pq_table_adapter.cpp llocal_search_quantizer.cpp)
    add_library(${PROJECT_NAME} ${CPP_SOURCES})
    add_library(${PROJECT_NAME}_s STATIC ${CPP_SOURCES})

    # Add ELPIS subdirectory
    add_subdirectory(elpis)
endif()
set(FAISS_ROOT /usr/local)
include_directories(${FAISS_ROOT}/include)

target_link_libraries(${PROJECT_NAME} PRIVATE ${FAISS_ROOT}/lib/libfaiss.so elpis)
install()
