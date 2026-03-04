import sys
import utils
import numpy as np

if __name__ == "__main__":
    data_path = sys.argv[1]
    data_type = sys.argv[2]
    output_path = sys.argv[3]

    if data_type == 'uint8' or data_type == 'uint32' or data_type == 'float' :
        if data_type == "float":
            dtype = np.float32
        if data_type == "uint32":
            dtype = np.uint32
        if data_type == "uint8":
            dtype = np.uint8
        data = utils.bin_to_numpy(dtype, data_path)
    else:
        print(f"Invalid data type: {data_type}")
        exit(1)

    n, d = data.shape
    print(f"Total points: {n}, Dim: {d}")
    utils.numpy_to_vecs(output_path, data, dtype)