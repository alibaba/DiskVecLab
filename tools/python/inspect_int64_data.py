import sys
import numpy as np


if __name__ == "__main__":
    data_path = sys.argv[1]
    K = sys.argv[2]
    array = np.fromfile(file=data_path, dtype=np.int64)
    n = array.shape
    print(f"Total points: {n}")
    for i in range(int(K)):
        print(array[i])