import numpy as np
import time
import sys
import logging
import os
import datetime
from h2o4gpu.decomposition import TruncatedSVDSklearn as sklearnsvd
from h2o4gpu.solvers import TruncatedSVDH2O

print(sys.path)

logging.basicConfig(level=logging.DEBUG)

def func(m=5000000, n=10, k=9, convert_to_float32=False):

    np.random.seed(1234)

    X = np.random.rand(m, n)

    if convert_to_float32:
        print("Converting input matrix to float32")
        X = X.astype(np.float32)

    # Warm start
    W = np.random.rand(1000, 5)
    print('h2o4gpu Cusolver Warm Start')
    h2o4gpu_tsvd_cusolver_warm = TruncatedSVDH2O(n_components=3, algorithm="cusolver", random_state=42)
    h2o4gpu_tsvd_cusolver_warm.fit(W)
    print('h2o4gpu Power Warm Start')
    h2o4gpu_tsvd_power_warm = TruncatedSVDH2O(n_components=3, algorithm="power", tol=1e-5, n_iter=100, random_state=42,
                                         verbose=True)
    h2o4gpu_tsvd_power_warm.fit(W)

    # Exact scikit impl
    sklearn_tsvd_arpack = sklearnsvd(algorithm="arpack", n_components=k)

    # Randomized scikit impl
    sklearn_tsvd_random = sklearnsvd(algorithm="randomized", n_components=k)

    #Cusolver h2o4gpu impl
    print("Cusolver SVD on " + str(X.shape[0]) + " by " + str(X.shape[1]) + " matrix")
    print("Original X Matrix")
    print(X)
    print("\n")
    print("h2o4gpu cusolver tsvd run")
    h2o4gpu_tsvd_cusolver = TruncatedSVDH2O(n_components=k)
    start_time_gpu_cusolver = time.time()
    h2o4gpu_tsvd_cusolver.fit(X)
    end_time_gpu_cusolver = time.time() - start_time_gpu_cusolver
    print("Total time for h2o4gpu cusolver tsvd is " + str(end_time_gpu_cusolver))
    print("h2o4gpu tsvd cusolver Singular Values")
    print(h2o4gpu_tsvd_cusolver.singular_values_)
    print("h2o4gpu tsvd cusolver Components (V^T)")
    print(h2o4gpu_tsvd_cusolver.components_)
    print("h2o4gpu tsvd cusolver Explained Variance")
    print(h2o4gpu_tsvd_cusolver.explained_variance_)
    print("h2o4gpu tsvd cusolver Explained Variance Ratio")
    print(h2o4gpu_tsvd_cusolver.explained_variance_ratio_)

    print("Sleep before Power")
    time.sleep(5)

    #Power h2o4gpu impl
    print("Power SVD on " + str(X.shape[0]) + " by " + str(X.shape[1]) + " matrix")
    print("Original X Matrix")
    print(X)
    print("\n")
    print("h2o4gpu tsvd power method run")
    h2o4gpu_tsvd_power = TruncatedSVDH2O(n_components=k, algorithm="power")
    start_time_gpu_power = time.time()
    h2o4gpu_tsvd_power.fit(X)
    end_time_gpu_power = time.time() - start_time_gpu_power
    print("Total time for h2o4gpu tsvd is " + str(end_time_gpu_power))
    print("h2o4gpu tsvd power Singular Values")
    print(h2o4gpu_tsvd_power.singular_values_)
    print("h2o4gpu tsvd power Components (V^T)")
    print(h2o4gpu_tsvd_power.components_)
    print("h2o4gpu tsvd power Explained Variance")
    print(h2o4gpu_tsvd_power.explained_variance_)
    print("h2o4gpu tsvd power Explained Variance Ratio")
    print(h2o4gpu_tsvd_power.explained_variance_ratio_)

    print("Sleep before Sklearn ARPACK")
    time.sleep(5)

    #ARPACK sklearn impl
    print("\n")
    print("ARPACK sklearn run")
    start_sk_arpack = time.time()
    sklearn_tsvd_arpack.fit(X)
    end_sk_arpack = time.time() - start_sk_arpack
    print("Total time for sklearn is " + str(end_sk_arpack))
    print("Sklearn ARPACK Singular Values")
    print(sklearn_tsvd_arpack.singular_values_)
    print("Sklearn ARPACK Components (V^T)")
    print(sklearn_tsvd_arpack.components_)
    print("Sklearn ARPACK Explained Variance")
    print(sklearn_tsvd_arpack.explained_variance_)
    print("Sklearn ARPACK Explained Variance Ratio")
    print(sklearn_tsvd_arpack.explained_variance_ratio_)

    print("Sleep before Sklearn Randomized")
    time.sleep(5)

    #Randomized sklearn impl
    print("\n")
    print("Randomized sklearn randomized run")
    start_sk_random = time.time()
    sklearn_tsvd_random.fit(X)
    end_sk_randomized = time.time() - start_sk_random
    print("Total time for sklearn is " + str(end_sk_randomized))
    print("Sklearn Random Singular Values")
    print(sklearn_tsvd_random.singular_values_)
    print("Sklearn Random Components (V^T)")
    print(sklearn_tsvd_random.components_)
    print("Sklearn Random Explained Variance")
    print(sklearn_tsvd_random.explained_variance_)
    print("Sklearn Random Explained Variance Ratio")
    print(sklearn_tsvd_random.explained_variance_ratio_)

    return end_time_gpu_cusolver, end_sk_arpack, end_time_gpu_power, end_sk_randomized

def run_bench(m=5000000, n=10, k=9, convert_to_float32=False):
    dtype = "float32"
    if not convert_to_float32:
        dtype = "double"

    results = func(m, n, k, convert_to_float32=convert_to_float32)
    if results[0] <= results[1]:
        print("h2o4gpu tsvd cusolver is faster than sklearn arpack for m = %s and n = %s" % (m,n))

    if results[2] <= results[3]:
        print("h2o4gpu tsvd power is faster than sklearn randomized for m = %s and n = %s" % (m,n))

    filename = 'bench_results.csv'

    if not os.path.exists(filename):
        append_write = 'w'  # make a new file if not
        timings = open(filename, append_write)
        timings.write("timestamp" + "," + "m" + "," + "n" + "," + "k" + "," + "data_precision" + "," + "h2o4gpu_cusolver" + "," + "sklearn_arpack" + ","
                      "h2o4gpu_power" + "," + "sklearn_random" + '\n')
        timings.close()

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    timings = open(filename, 'a')
    timings.write(str(st) + "," + str(m)+","+str(n)+","+str(k)+","+str(dtype)+","+str(results[0])+","+str(results[1])+","+str(results[2])+","+str(results[3])+'\n')
    timings.close()

# def test_tsvd_error_k2(): run_bench(m=5000000, n=10, k=2)
# def test_tsvd_error_k5(): run_bench(m=5000000, n=100, k=5)
# def test_tsvd_error_k10(): run_bench(m=1000000, n=1000, k=10)
# def test_tsvd_error_k100(): run_bench(m=100000, n=10000, k=100)
def test_tsvd_error_k2_float32(): run_bench(m=5000000, n=10, k=2, convert_to_float32=True)
def test_tsvd_error_k5_float32(): run_bench(m=5000000, n=100, k=5, convert_to_float32=True)
def test_tsvd_error_k10_float32(): run_bench(m=1000000, n=1000, k=10, convert_to_float32=True)
def test_tsvd_error_k100_float32(): run_bench(m=100000, n=10000, k=100, convert_to_float32=True)
# def test_tsvd_error_k2_n10_m5k(): run_bench(m=5000, n=10, k=2)
# def test_tsvd_error_k5_n100_m5k(): run_bench(m=5000, n=100, k=5)
# def test_tsvd_error_k10_n50_m50k(): run_bench(m=50000, n=50, k=10)
# def test_tsvd_error_k100_n500_m50k(): run_bench(m=50000, n=500, k=100)
# def test_tsvd_error_k10_n50_m500k(): run_bench(m=500000, n=50, k=10)
# def test_tsvd_error_k100_n500_m500k(): run_bench(m=500000, n=500, k=100)
def test_tsvd_error_k2_n10_m5k_float32(): run_bench(m=5000, n=10, k=2, convert_to_float32=True)
def test_tsvd_error_k5_n100_m5k_float32(): run_bench(m=5000, n=100, k=5, convert_to_float32=True)
def test_tsvd_error_k10_n50_m50k_float32(): run_bench(m=50000, n=50, k=10, convert_to_float32=True)
def test_tsvd_error_k100_n500_m50k_float32(): run_bench(m=50000, n=500, k=100, convert_to_float32=True)
def test_tsvd_error_k10_n50_m500k_float32(): run_bench(m=500000, n=50, k=10, convert_to_float32=True)
def test_tsvd_error_k100_n500_m500k_float32(): run_bench(m=500000, n=500, k=100, convert_to_float32=True)



