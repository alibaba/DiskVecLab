# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

set(CMAKE_CXX_STANDARD 17)

if(MSVC)
    add_subdirectory(dll)
else()
    #file(GLOB CPP_SOURCES *.cpp)
    set(CPP_SOURCES ann_exception.cpp aux_utils.cpp distance.cpp index.cpp
        linux_aligned_file_reader.cpp math_utils.cpp natural_number_map.cpp
        natural_number_set.cpp memory_mapper.cpp partition_and_pq.cpp
        logger.cpp utils.cpp
        io_registry.cpp local_filesystem.cpp file_streams.cpp
        gorgeous/deco_index.cpp gorgeous/index_search.cpp
        gorgeous/index_search_dup_graph.cpp
        starling/page_search.cpp starling/pq_flash_index.cpp
        file_io_manager.cpp)
    if(WITH_S3)
        list(APPEND CPP_SOURCES
            s3_backend/s3_uri.cpp
            s3_backend/s3_client_factory.cpp
            s3_backend/s3_file.cpp
            s3_backend/s3_filesystem.cpp
            s3_backend/s3_async_block_reader.cpp
            s3_backend/s3_register.cpp)
    endif()
    add_library(${PROJECT_NAME} ${CPP_SOURCES})
    add_library(${PROJECT_NAME}_s STATIC ${CPP_SOURCES})
    # io_registry self-registers Linux/FileIOManager factories at static-init,
    # so libdiskann unconditionally needs libaio. Propagate it to consumers so
    # all downstream executables (including build_memory_index and tests/utils)
    # link against it without each having to know.
    target_link_libraries(${PROJECT_NAME} tbb ${DISKANN_ASYNC_LIB})
    target_link_libraries(${PROJECT_NAME}_s tbb ${DISKANN_ASYNC_LIB})
    if(WITH_S3)
        target_include_directories(${PROJECT_NAME}   PRIVATE ${AWSSDK_INCLUDE_DIRS})
        target_include_directories(${PROJECT_NAME}_s PRIVATE ${AWSSDK_INCLUDE_DIRS})
        target_link_libraries(${PROJECT_NAME}   ${AWSSDK_LINK_LIBRARIES})
        target_link_libraries(${PROJECT_NAME}_s ${AWSSDK_LINK_LIBRARIES})
    endif()
endif()
install()
