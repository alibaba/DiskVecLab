import sys
import utils
import numpy as np

def print_row(data, K):
    n, d = data.shape
    print(f"Total points: {n}, Dim: {d}")
    if K < 0:
        print(f"Top {-K} data: ")
        for i in range(int(-K)):
            print(data[i])
    else:
        print(f"Num {K} data: ")
        print(data[K])

if __name__ == "__main__":
    data_path = sys.argv[1]
    data_type = sys.argv[2]
    K = sys.argv[3]

    if data_type == 'fvecs':
        data = utils.read_fvecs(data_path)
    elif data_type == 'ivecs':
        data = utils.read_ivecs(data_path)
    elif data_type == 'bvecs':
        data = utils.mmap_bvecs(data_path)
    elif data_type == 'uint8' or data_type == 'uint32' or data_type == 'float' :
        if data_type == "float":
            dtype = np.float32
        if data_type == "uint32":
            dtype = np.uint32
        if data_type == "uint8":
            dtype = np.uint8
        data = utils.bin_to_numpy(dtype, data_path)
    elif data_type == 'gt':
        ids, dists = utils.read_gt_bin_file(data_path)
    else:
        print(f"Invalid data type: {data_type}")
        exit(1)

    if data_type == 'gt':
        print("Ids of ground truth:")
        print_row(ids, int(K))
        print("Distances of ground truth:")
        print_row(dists, int(K))
    else:
        print_row(data, int(K))
