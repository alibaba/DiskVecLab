import sys
import utils

if __name__ == "__main__":
    data_path = sys.argv[1]
    data_type = sys.argv[2]
    output_path = sys.argv[3]

    if data_type == 'fvecs':
        data = utils.read_fvecs(data_path)
    elif data_type == 'ivecs':
        data = utils.read_ivecs(data_path)
    elif data_type == 'bvecs':
        data = utils.mmap_bvecs(data_path)
    else:
        print(f"Invalid data type: {data_type}")
        exit(1)

    n, d = data.shape
    print(f"Total points: {n}, Dim: {d}")
    utils.numpy_to_bin(data, output_path)