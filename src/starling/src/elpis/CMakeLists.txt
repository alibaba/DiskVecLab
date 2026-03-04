# ELPIS library CMakeLists.txt

set(ELPIS_SOURCES
    BufferManager.cpp
    calc_utils.cpp
    Index.cpp
    Node.cpp
    pqueue.cpp
    QueryEngine.cpp
    Setting.cpp
)

add_library(elpis ${ELPIS_SOURCES})

# Include directories for ELPIS
target_include_directories(elpis PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/elpis
)

# Link with required libraries
target_link_libraries(elpis PUBLIC stdc++fs)