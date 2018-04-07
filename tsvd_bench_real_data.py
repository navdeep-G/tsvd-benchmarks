import numpy as np
import time
import sys
import logging
import os
import datetime
import datasets
from h2o4gpu.decomposition import TruncatedSVDSklearn as sklearnsvd
from h2o4gpu.solvers import TruncatedSVDH2O

print(sys.path)

logging.basicConfig(level=logging.DEBUG)

def func(k=9, dataset="higgs"):

    #Get higgs dataset
    if dataset == "higgs":
        print("Getting Higgs dataset")
        X = datasets.get_higgs()
    elif dataset == "covtype":
        print("Getting covtype dataset")
        X = datasets.get_cover_type()
    elif dataset == "regression":
        print("Getting synthetic regression data")
        X = datasets.get_synthetic_regression()
    elif dataset == "year":
        print("Getting YearPredictionMSD data")
        X = datasets.get_year()
    else:
        raise Exception("Unrecognized dataset " + dataset)

    # Warm start
    W = np.random.rand(1000, 5)
    print('h2o4gpu Cusolver Warm Start')
    h2o4gpu_tsvd_cusolver_warm = TruncatedSVDH2O(n_components=3, algorithm="cusolver", tol=1e-5, n_iter=100, random_state=42,
                                         verbose=True)
    h2o4gpu_tsvd_cusolver_warm.fit(W)
    print('h2o4gpu Power Warm Start')
    h2o4gpu_tsvd_power_warm = TruncatedSVDH2O(n_components=3, algorithm="power", tol=1e-5, n_iter=100, random_state=42,
                                         verbose=True)
    h2o4gpu_tsvd_power_warm.fit(W)

    print('sklearn ARPACK Warm Start')
    sklearn_tsvd_arpack_warm = sklearnsvd(n_components=3, algorithm="arpack", n_iter=5, random_state=42)
    sklearn_tsvd_arpack_warm.fit(W)
    print('sklearn Randomized Warm Start')
    sklearn_tsvd_random_warm = sklearnsvd(n_components=3, algorithm="randomized", tol=1e-5, n_iter=5, random_state=42)
    sklearn_tsvd_random_warm.fit(W)

    # Exact scikit impl
    sklearn_tsvd_arpack = sklearnsvd(algorithm="arpack", n_components=k, tol=1e-5, n_iter=5, random_state=42)

    # Randomized scikit impl
    sklearn_tsvd_random = sklearnsvd(algorithm="randomized", n_components=k, tol=1e-5, n_iter=5, random_state=42)

    #Cusolver h2o4gpu impl
    print("Cusolver SVD on " + str(X.shape[0]) + " by " + str(X.shape[1]) + " matrix")
    print("Original X Matrix")
    print(X)
    print("\n")
    print("h2o4gpu cusolver tsvd run")
    h2o4gpu_tsvd_cusolver = TruncatedSVDH2O(n_components=k, algorithm="cusolver", tol=1e-5, n_iter=100, random_state=42)
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
    h2o4gpu_tsvd_power = TruncatedSVDH2O(n_components=k, algorithm="power", tol=1e-5, n_iter=100, random_state=42)
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

    return end_time_gpu_cusolver, end_sk_arpack, end_time_gpu_power, end_sk_randomized, X.shape[0], X.shape[1]

def run_bench(k=9, dataset="higgs"):
    results = func(k, dataset)
    dtype="double"
    directory="results"

    if results[0] <= results[1]:
        print("h2o4gpu tsvd cusolver is faster than sklearn arpack for m = %s and n = %s" % (results[4],results[5]))

    if results[2] <= results[3]:
        print("h2o4gpu tsvd power is faster than sklearn randomized for m = %s and n = %s" % (results[4],results[5]))

    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = directory + "/bench_results_real_data.csv"

    if not os.path.exists(filename):
        append_write = 'w'  # make a new file if not
        timings = open(filename, append_write)
        timings.write("timestamp" + "," + "dataset" + "," + "m" + "," + "n" + "," + "k" + "," + "data_precision" + "," + "h2o4gpu_cusolver" + "," + "sklearn_arpack" + ","
                      "h2o4gpu_power" + "," + "sklearn_random" + '\n')
        timings.close()

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    timings = open(filename, 'a')
    timings.write(str(st) + "," + str(dataset) + "," + str(results[4])+","+str(results[5])+","+str(k)+","+str(dtype)+","+str(results[0])+","+str(results[1])+","+str(results[2])+","+str(results[3])+'\n')
    timings.close()

def test_tsvd_k2_covtype(): run_bench(k=2, dataset="covtype")
def test_tsvd_k3_covtype(): run_bench(k=3, dataset="covtype")
def test_tsvd_k4_covtype(): run_bench(k=4, dataset="covtype")
def test_tsvd_k5_covtype(): run_bench(k=5, dataset="covtype")
def test_tsvd_k6_covtype(): run_bench(k=6, dataset="covtype")
def test_tsvd_k7_covtype(): run_bench(k=7, dataset="covtype")
def test_tsvd_k8_covtype(): run_bench(k=8, dataset="covtype")
def test_tsvd_k9_covtype(): run_bench(k=9, dataset="covtype")
def test_tsvd_k10_covtype(): run_bench(k=10, dataset="covtype")

def test_tsvd_k2_year(): run_bench(k=2, dataset="year")
def test_tsvd_k3_year(): run_bench(k=3, dataset="year")
def test_tsvd_k4_year(): run_bench(k=4, dataset="year")
def test_tsvd_k5_year(): run_bench(k=5, dataset="year")
def test_tsvd_k6_year(): run_bench(k=6, dataset="year")
def test_tsvd_k7_year(): run_bench(k=7, dataset="year")
def test_tsvd_k8_year(): run_bench(k=8, dataset="year")
def test_tsvd_k9_year(): run_bench(k=9, dataset="year")
def test_tsvd_k10_year(): run_bench(k=10, dataset="year")

def test_tsvd_k2_regression(): run_bench(k=2, dataset="regression")
def test_tsvd_k3_regression(): run_bench(k=3, dataset="regression")
def test_tsvd_k4_regression(): run_bench(k=4, dataset="regression")
def test_tsvd_k5_regression(): run_bench(k=5, dataset="regression")
def test_tsvd_k6_regression(): run_bench(k=6, dataset="regression")
def test_tsvd_k7_regression(): run_bench(k=7, dataset="regression")
def test_tsvd_k8_regression(): run_bench(k=8, dataset="regression")
def test_tsvd_k9_regression(): run_bench(k=9, dataset="regression")
def test_tsvd_k10_regression(): run_bench(k=10, dataset="regression")

def test_tsvd_k2_higgs(): run_bench(k=2, dataset="higgs")
def test_tsvd_k3_higgs(): run_bench(k=3, dataset="higgs")
def test_tsvd_k4_higgs(): run_bench(k=4, dataset="higgs")
def test_tsvd_k5_higgs(): run_bench(k=5, dataset="higgs")
def test_tsvd_k6_higgs(): run_bench(k=6, dataset="higgs")
def test_tsvd_k7_higgs(): run_bench(k=7, dataset="higgs")
def test_tsvd_k8_higgs(): run_bench(k=8, dataset="higgs")
def test_tsvd_k9_higgs(): run_bench(k=9, dataset="higgs")
def test_tsvd_k10_higgs(): run_bench(k=10, dataset="higgs")



