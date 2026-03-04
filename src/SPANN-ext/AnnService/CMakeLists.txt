# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set(AnnService ${PROJECT_SOURCE_DIR}/AnnService)
set(Zstd ${PROJECT_SOURCE_DIR}/ThirdParty/zstd)
set(RabitqLib ${PROJECT_SOURCE_DIR}/AnnService/inc/Core/Common/rabitqlib)
set(FAISS_ROOT /usr/local)
include_directories(${FAISS_ROOT}/include)

# Find FAISS library
find_path(FAISS_INCLUDE_DIR
    NAMES faiss/IndexAdditiveQuantizer.h
    PATHS /usr/local/include /opt/local/include /usr/include
    PATH_SUFFIXES faiss
)

find_library(FAISS_LIBRARY
    NAMES faiss
    PATHS /usr/local/lib /opt/local/lib /usr/lib /usr/lib64
)

if(FAISS_INCLUDE_DIR AND FAISS_LIBRARY)
    include_directories(${FAISS_INCLUDE_DIR})
    message(STATUS "Found FAISS: ${FAISS_LIBRARY} ${FAISS_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "FAISS library not found. Please install FAISS or set FAISS_ROOT environment variable.")
endif()

include_directories(${AnnService})
include_directories(${RabitqLib})
include_directories(${Zstd}/lib)

file(GLOB_RECURSE HDR_FILES ${AnnService}/inc/Core/*.h  ${AnnService}/inc/Helper/*.h)
file(GLOB_RECURSE SRC_FILES ${AnnService}/src/Core/*.cpp ${AnnService}/src/Helper/*.cpp ${AnnService}/src/Core/Common/RaBitQQuantizer.cpp ${AnnService}/src/Core/Common/LSQQuantizer.cpp)

list(REMOVE_ITEM HDR_FILES
    ${AnnService}/inc/Core/Common/DistanceUtils.h
    ${AnnService}/inc/Core/Common/SIMDUtils.h 
    ${AnnService}/inc/Core/Common/InstructionUtils.h
    ${AnnService}/inc/Core/Common/CommonUtils.h
    ${AnnService}/inc/Core/Common/RaBitQQuantizer.h
    ${AnnService}/inc/Core/Common/LSQQuantizer.h
    )

list(REMOVE_ITEM SRC_FILES
    ${AnnService}/src/Core/Common/DistanceUtils.cpp
    ${AnnService}/src/Core/Common/SIMDUtils.cpp
    ${AnnService}/src/Core/Common/InstructionUtils.cpp
    ${AnnService}/src/Core/Common/RaBitQQuantizer.cpp
    ${AnnService}/src/Core/Common/LSQQuantizer.cpp
    )

add_library (DistanceUtils STATIC
    inc/Core/Common/DistanceUtils.h
    inc/Core/Common/SIMDUtils.h
    inc/Core/Common/InstructionUtils.h
    inc/Core/Common/CommonUtils.h
    inc/Core/Common/RaBitQQuantizer.h
    inc/Core/Common/LSQQuantizer.h
    inc/Core/Common/FlatQuantizer.h
    src/Core/Common/DistanceUtils.cpp
    src/Core/Common/SIMDUtils.cpp
    src/Core/Common/InstructionUtils.cpp
    src/Core/Common/RaBitQQuantizer.cpp
    src/Core/Common/LSQQuantizer.cpp
    src/Core/Common/FlatQuantizer.cpp
    )

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    target_compile_options(DistanceUtils PRIVATE -mavx2 -mavx -msse -msse2 -mavx512f -mavx512bw -mavx512dq -mfma -fPIC)
endif()

add_library (SPTAGLib SHARED ${SRC_FILES} ${HDR_FILES})
target_link_libraries (SPTAGLib DistanceUtils libzstd_shared ${NUMA_LIBRARY} ${FAISS_LIBRARY})
add_library (SPTAGLibStatic STATIC ${SRC_FILES} ${HDR_FILES})
target_link_libraries (SPTAGLibStatic DistanceUtils libzstd_static ${NUMA_LIBRARY_STATIC} ${FAISS_LIBRARY})
if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    target_compile_options(SPTAGLibStatic PRIVATE -fPIC)
endif()

install(TARGETS SPTAGLib SPTAGLibStatic
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)

if (NOT LIBRARYONLY)
    file(GLOB SERVER_HDR_FILES ${AnnService}/inc/Server/*.h ${AnnService}/inc/Socket/*.h)
    file(GLOB SERVER_FILES ${AnnService}/src/Server/*.cpp ${AnnService}/src/Socket/*.cpp)
    add_executable (server ${SERVER_FILES} ${SERVER_HDR_FILES})
    target_link_libraries(server ${Boost_LIBRARIES} SPTAGLibStatic)

    file(GLOB CLIENT_HDR_FILES ${AnnService}/inc/Client/*.h ${AnnService}/inc/Socket/*.h)
    file(GLOB CLIENT_FILES ${AnnService}/src/Client/*.cpp ${AnnService}/src/Socket/*.cpp)
    add_executable (client ${CLIENT_FILES} ${CLIENT_HDR_FILES})
    target_link_libraries(client ${Boost_LIBRARIES} SPTAGLibStatic)

    file(GLOB AGG_HDR_FILES ${AnnService}/inc/Aggregator/*.h ${AnnService}/inc/Socket/*.h ${AnnService}/inc/Server/QueryParser.h)
    file(GLOB AGG_FILES ${AnnService}/src/Aggregator/*.cpp ${AnnService}/src/Socket/*.cpp ${AnnService}/src/Server/QueryParser.cpp)
    add_executable (aggregator ${AGG_FILES} ${AGG_HDR_FILES})
    target_link_libraries(aggregator ${Boost_LIBRARIES} SPTAGLibStatic)

    file(GLOB BUILDER_FILES ${AnnService}/src/IndexBuilder/*.cpp)
    add_executable (indexbuilder ${BUILDER_FILES})
    target_link_libraries(indexbuilder ${Boost_LIBRARIES} SPTAGLibStatic)

    file(GLOB SEARCHER_FILES ${AnnService}/src/IndexSearcher/*.cpp)
    add_executable (indexsearcher ${SEARCHER_FILES})
    target_link_libraries(indexsearcher ${Boost_LIBRARIES} SPTAGLibStatic)
    
    file(GLOB QUANTIZER_HDR_FILES ${AnnService}/inc/Quantizer/*.h)
    file(GLOB QUANTIZER_FILES ${AnnService}/src/Quantizer/*.cpp)
    add_executable (quantizer ${QUANTIZER_FILES} ${QUANTIZER_HDR_FILES})
    target_link_libraries(quantizer ${Boost_LIBRARIES} SPTAGLibStatic DistanceUtils ${FAISS_LIBRARY})


    install(TARGETS server client aggregator indexbuilder indexsearcher quantizer
      RUNTIME DESTINATION bin
      ARCHIVE DESTINATION lib
      LIBRARY DESTINATION lib)
endif()

file(GLOB_RECURSE SSD_SERVING_HDR_FILES ${AnnService}/inc/SSDServing/*.h)
file(GLOB_RECURSE SSD_SERVING_FILES ${AnnService}/src/SSDServing/*.cpp)

add_executable(ssdserving ${SSD_SERVING_HDR_FILES} ${SSD_SERVING_FILES})
target_link_libraries(ssdserving SPTAGLibStatic ${Boost_LIBRARIES})
target_compile_definitions(ssdserving PRIVATE _exe)

# for Test
add_library(ssdservingLib ${SSD_SERVING_HDR_FILES} ${SSD_SERVING_FILES})
target_link_libraries(ssdservingLib SPTAGLibStatic ${Boost_LIBRARIES})

find_package(MPI)
if (MPI_FOUND)
    message (STATUS "Found MPI.")
    message (STATUS "MPI Include Path: ${MPI_CXX_INCLUDE_PATH}")
    message (STATUS "MPI Libraries: ${MPI_CXX_LIBRARIES}")
    include_directories (${MPI_CXX_INCLUDE_PATH})
    file(GLOB PARTITION_FILES ${AnnService}/src/BalancedDataPartition/*.cpp)
    add_executable(balanceddatapartition ${PARTITION_FILES})
    target_link_libraries(balanceddatapartition SPTAGLibStatic ${Boost_LIBRARIES} ${MPI_CXX_LIBRARIES})
endif()

install(TARGETS ssdservingLib ssdserving
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)
