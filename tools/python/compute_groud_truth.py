import sys
import utils
import numpy as np
import faiss

if __name__ == "__main__":
    data_path = sys.argv[1]
    query_path = sys.argv[2]
    data_type = sys.argv[3]
    K = sys.argv[4]
    output_path = sys.argv[5]

    if data_type == 'fvecs':
        data = utils.read_fvecs(data_path)
        query = utils.read_fvecs(query_path)
    elif data_type == 'ivecs':
        data = utils.read_ivecs(data_path)
        query = utils.read_ivecs(query_path)
    elif data_type == 'bvecs':
        data = utils.mmap_bvecs(data_path)
        query = utils.mmap_bvecs(query_path)
    elif data_type == 'uint8' or data_type == 'uint32' or data_type == 'float' :
        if data_type == "float":
            dtype = np.float32
        if data_type == "uint32":
            dtype = np.uint32
        if data_type == "uint8":
            dtype = np.uint8
        data = utils.bin_to_numpy(dtype, data_path)
        query = utils.bin_to_numpy(dtype, query_path)
    else:
        print(f"Invalid data type: {data_type}")
        exit(1)

    dim = data.shape[1]

    remember_nt = faiss.omp_get_max_threads()
    print(f'remember_nt: {remember_nt}')
    faiss.omp_set_num_threads(remember_nt)

    index = faiss.IndexFlatL2(dim)
    index.verbose = True
    index.add(data)

    _, ids = index.search(query, int(K))
    utils.numpy_to_vecs(output_path, ids, np.uint32)
