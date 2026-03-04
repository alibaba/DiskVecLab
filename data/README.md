# Used Datasets

All datasets are publicly available and the links are provided as follows:
| Dataset | Dimensionality | Download Link | Note |
| --- | --- | --- | --- |
| LAION-T2I/I2I | 512/768 | [Official Site (512 dim.)](https://laion.ai/blog/laion-400-open-dataset/)/[Official Site (768 dim.)](https://laion.ai/blog/laion-5b/) | Both 512 and 768-dimensional versions provides text based vectors and image based vectors. In experiments we use 512 dimensional version for **text to image** search (out-of-distribution) and 768 dimensional version for **image to image** search (in-distribution). |
| DEEP | 96 | [Link from Yandex](https://research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search/)/[Base Set](https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.1B.fbin)/[Query Set](https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/query.public.10K.fbin) | |
| Text2Image | 200 | [Link from Yandex](https://research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search/)/[Base Set](https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/base.1B.fbin)/[Query Set](https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.public.100K.fbin) | A **text to image** search (out-of-distribution) dataset where base sets are image vectors and query sets are text vectors. |
| SIFT | 128 | [Official Site](http://corpus-texmex.irisa.fr/) | |
| SpaceV | 100 | [Official Repo](https://github.com/microsoft/SPTAG/tree/b2748d982c68be70240285b0e222acea62d6c08e/datasets) | |

## Formats

Datasets are available in multiple formats. 
In our experiments, we use the bin format, either float32 (fbin) or uint8 (bin), for index construction and search. 
Some methods require other formats (e.g., fvecs); those repositories typically provide conversion tools.
We also provide some conversion scripts in `../tools` for your reference.