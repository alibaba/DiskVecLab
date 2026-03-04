# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

set(CMAKE_CXX_STANDARD 17)

set(FAISS_ROOT /usr/local)
include_directories(${FAISS_ROOT}/include)

find_package(OpenMP REQUIRED)

add_executable(build_memory_index build_memory_index.cpp)
target_link_libraries(build_memory_index ${PROJECT_NAME} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} Boost::program_options)

add_executable(search_memory_index search_memory_index.cpp)
target_link_libraries(search_memory_index ${PROJECT_NAME} ${DISKANN_ASYNC_LIB} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} Boost::program_options)

add_executable(build_disk_index build_disk_index.cpp)
target_link_libraries(build_disk_index ${PROJECT_NAME} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} ${DISKANN_ASYNC_LIB} Boost::program_options)

add_executable(search_disk_index search_disk_index.cpp)
target_link_libraries(search_disk_index ${PROJECT_NAME} ${DISKANN_ASYNC_LIB} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} Boost::program_options)

add_executable(search_disk_index_multi_split search_disk_index_multi_split.cpp)
target_link_libraries(search_disk_index_multi_split ${PROJECT_NAME} elpis ${DISKANN_ASYNC_LIB} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} Boost::program_options -lstdc++fs)

add_executable(range_search_disk_index range_search_disk_index.cpp)
target_link_libraries(range_search_disk_index ${PROJECT_NAME} ${DISKANN_ASYNC_LIB} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} Boost::program_options)

# add_executable(range_search_disk_different_radius range_search_disk_different_radius.cpp)
# target_link_libraries(range_search_disk_different_radius ${PROJECT_NAME} ${DISKANN_ASYNC_LIB} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} Boost::program_options)
#add_executable(test_incremental_index test_incremental_index.cpp)
#target_link_libraries(test_incremental_index ${PROJECT_NAME} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} Boost::program_options)

add_executable(test_streaming_scenario test_streaming_scenario.cpp)
target_link_libraries(test_streaming_scenario ${PROJECT_NAME} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} Boost::program_options)

add_executable(test_insert_deletes_consolidate test_insert_deletes_consolidate.cpp)
target_link_libraries(test_insert_deletes_consolidate ${PROJECT_NAME} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} Boost::program_options)

add_executable(search_memory_index_dynamic search_memory_index_dynamic.cpp)
target_link_libraries(search_memory_index_dynamic ${PROJECT_NAME} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} Boost::program_options)

add_executable(search_disk_index_save_freq search_disk_index_save_freq.cpp)
target_link_libraries(search_disk_index_save_freq ${PROJECT_NAME} ${DISKANN_ASYNC_LIB} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} Boost::program_options)

add_executable(search_disk_index_adaptive_multi_shard search_disk_index_adaptive_multi_shard.cpp)
target_link_libraries(search_disk_index_adaptive_multi_shard ${PROJECT_NAME} ${DISKANN_ASYNC_LIB} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} Boost::program_options -lstdc++fs)