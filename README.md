# Randomized SVD

Implementation and benchmarking of Randomized Singular Value Decomposition algorithms for efficient low-rank matrix approximation, as described in our paper [Randomized Singular Value Decomposition](files/Randomized_SVD_paper.pdf).

## Quick Start

```bash
git clone https://github.com/IMRUNya/rSVD.git
cd rSVD
pip install -r requirements.txt
```

## Usage

```python
import numpy as np
from synthetic_matrix_benchmark import randomized_svd

# Your matrix
A = np.random.randn(1000, 1000)

# Compute rank-50 approximation
(U, S, Vt), runtime = randomized_svd(A, rank=50, oversampling=10, n_iter=2)
A_approx = (U * S) @ Vt
```

## Experiments

Run benchmarks:
```bash
python synthetic_matrix_benchmark.py
```

Explore image compression:
```bash
jupyter notebook rsvd_image_compression.ipynb
```