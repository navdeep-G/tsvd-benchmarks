import gzip
import os
import pandas as pd
import numpy as np
import shutil
import sys
import tarfile
import zipfile
from scipy.sparse import vstack
from sklearn import datasets
from sklearn.externals.joblib import Memory

if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve

mem = Memory("./mycache")


@mem.cache
def get_higgs(num_rows=None):
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz'
    filename = 'HIGGS.csv'
    if not os.path.isfile(filename):
        urlretrieve(url, filename + '.gz')
        with gzip.open(filename + '.gz', 'rb') as f_in:
            with open(filename, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    higgs = pd.read_csv(filename)
    if num_rows is not None:
        higgs = higgs[0:num_rows]

    return higgs.as_matrix()

@mem.cache
def get_cover_type(num_rows=None):
    data = datasets.fetch_covtype()
    data = data.data
    if num_rows is not None:
        data = data[0:num_rows]

    return data

@mem.cache
def get_synthetic_regression(num_rows=None):
    if num_rows is None:
        num_rows = 10000000
    X, y = datasets.make_regression(n_samples=num_rows, bias=100, noise=1.0)
    X = X.astype(np.float32)

    return X


@mem.cache
def get_year(num_rows=None):
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip'
    filename = 'YearPredictionMSD.txt'
    if not os.path.isfile(filename):
        urlretrieve(url, filename + '.zip')
        zip_ref = zipfile.ZipFile(filename + '.zip', 'r')
        zip_ref.extractall()
        zip_ref.close()

    year = pd.read_csv('YearPredictionMSD.txt', header=None)
    if num_rows is not None:
        year = year[0:num_rows]

    return year.as_matrix()


@mem.cache
def get_url(num_rows=None):
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/url/url_svmlight.tar.gz'
    filename = 'url_svmlight.tar.gz'
    if not os.path.isfile(filename):
        urlretrieve(url, filename)
        tar = tarfile.open(filename, "r:gz")
        tar.extractall()
        tar.close()

    num_files = 120
    files = ['url_svmlight/Day{}.svm'.format(day) for day in range(num_files)]
    data = datasets.load_svmlight_files(files)
    X = vstack(data[::2])

    if num_rows is not None:
        X = X[0:num_rows]

    return X
