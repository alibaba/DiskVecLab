# MARGO

This is the source code of MARGO.

Our paper, "Select Edges Wisely: Monotonic Path Aware Graph Layout Optimization for Disk-based ANN Search", is submitted to VLDB2025.

## Quick Start
### Install Dependencies
```bash
apt install build-essential libboost-all-dev make cmake g++ libaio-dev libgoogle-perftools-dev clang-format libboost-all-dev libmkl-full-dev
```
### Index Construction
Construct index on SIFT100K dataset, which is provided in `data/sift100k` directory. Go to `scripts` directory and run
```bash
bash ./run_benchmark.sh release build
```
### Graph Layout Optimization
After index construction, go to `my_gp` directory and run
```bash
bash ./mcrun.sh
```
### ANN Search
After graph layout optimization, go to `scripts` and run
```bash
bash ./run_benchmark.sh search knn
```

### Configuration
For index construction and ANN search, go to `script` directory and modify the configurations in `config_dataset.sh` and `config_local.sh`. For graph layout optimization, the only parameter `nlist` can be set in `my_gp/new_mincut.cpp`. For example, `instance->partition( base_path, 256);`
means `nlist` is set to 256. Set `nlist` to 1 for the greedy algorithm.

### Reference
Please cite the following reference when you use MARGO in your research or development.
```bibtex
@article{DBLP:journals/pvldb/YueZXXZDGZJ25,
  author       = {Ziyang Yue and
                  Bolong Zheng and
                  Ling Xu and
                  Kanru Xu and
                  Shuhao Zhang and
                  Yajuan Du and
                  Yunjun Gao and
                  Xiaofang Zhou and
                  Christian S. Jensen},
  title        = {Select Edges Wisely: Monotonic Path Aware Graph Layout Optimization
                  for Disk-based {ANN} Search},
  journal      = {Proc. {VLDB} Endow.},
  volume       = {18},
  number       = {11},
  pages        = {4337--4349},
  year         = {2025}
}
