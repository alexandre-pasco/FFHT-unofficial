import numpy as np
import ffht
from time import perf_counter
import sys

reps = 1
d = 20
n = 2**d
k = 16
nthreads = 4

# a = np.random.randn(n).astype(np.float32)
x1 = np.random.normal(size=n)
x2 = np.random.normal(size=(k, n))

# Out of place fht on a 1d array
t1 = perf_counter()
for i in range(reps):
    _ = ffht.fht(x1)
t1d = (perf_counter() - t1) / reps

# Out of place fht on a 2d array
t1 = perf_counter()
for i in range(reps):
    _ = ffht.fht(x2, nthreads=nthreads)
t2d = (perf_counter() - t1) / reps


if sys.version_info[0] == 2:
    print("Mean of {} runs on a vector of dimension 2**{}".format(reps, d))
    print("Out of place FHT on 1d array in {} sec".format(t1d))
    print("Out of place FHT on 2d array in {} sec for {} vectors with {} threads".format(t2d, k, nthreads))
if sys.version_info[0] == 3:
    print(f"Mean of {reps} runs on a vector of dimension 2**{d}")
    print(f"Out of place FHT on 1d array in {t1d:.3e} sec")
    print(f"Out of place FHT on 2d array in {t2d:.3e} sec for {k} vectors with {nthreads} threads")
