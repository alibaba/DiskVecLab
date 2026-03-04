include_directories(include)

add_executable(new_mincut new_mincut.cpp)

find_package(OpenMP REQUIRED)
target_link_libraries(new_mincut OpenMP::OpenMP_CXX ${PROJECT_NAME})