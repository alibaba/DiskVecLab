#include <iostream>
#include <mincut.h>
#include <string>


int main( int argc, char *argv[]) {

    if( argc != 3) {
        std::cerr << "wrong input parameters" << std::endl;
        return 0;
    }
    std::string index_path;
    std::string weight_path;
    std::string disk_path;
    std::string partition_path;
    std::string reverse_path;
    std::string undirect_path;
    std::string edge_path;
    index_path = (std::string)argv[1] + "_mem.index";
    weight_path = (std::string)argv[1] + "_mem.index.weights";
    disk_path = (std::string)argv[1] + "_disk.index";
    partition_path = (std::string)argv[1] + "_partition.bin";
    reverse_path = (std::string)argv[1] + "_reverse.index";
    undirect_path = (std::string)argv[1] +"_undirect.index";
    edge_path = (std::string)argv[1] + "_sorted.edge";

    std::string base_path;
    base_path = (std::string)argv[2];

    auto begin_time = std::chrono::high_resolution_clock::now();
    mincut *instance = new mincut();

    instance->load_meta_data( disk_path);
    instance->load_index_graph( index_path);
    instance->partition( base_path, 256);
    instance->save_partition( partition_path);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>( end_time - begin_time);
    std::cout << "mincut partition cost " << duration.count() << "s" << std::endl;
    delete instance;

    return 0;
}