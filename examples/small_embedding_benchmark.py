# -*- coding: utf-8 -*-
r"""
Small Ordinal Embedding Benchmark
=================================

In this example, we generate an artificial set of triplets to fit an ordinal embedding algorithm
like SOE. We vary the data dimension and measure the average fit's duration and accuracy.
"""

import time

import numpy as np

from cblearn import embedding
from cblearn.datasets import make_random_triplets


def benchmark_embedding(embedding, n_dims):
    estimator.n_components = n_dims
    X = np.random.normal(size=(n_samples, n_dims))
    T = make_random_triplets(X, size=n_triplets, result_format='list-order')
    T_test = make_random_triplets(X, size=10_000, result_format='list-order')

    delta_times = []
    train_accs = []
    test_accs = []
    for _ in range(n_repeat):
        start_time = time.time()
        estimator.fit(T)
        end_time = time.time()
        delta_times.append(end_time - start_time)
        train_accs.append(estimator.score(T))
        test_accs.append(estimator.score(T_test))

    print(f"{estimator}: time      {np.mean(delta_times):.2f} (sd: {np.std(delta_times):.2f})")
    print(f"{estimator}: train acc {np.mean(train_accs):.2f} (sd: {np.std(train_accs):.2f})")
    print(f"{estimator}: test  acc {np.mean(test_accs):.2f} (sd: {np.std(test_accs):.2f})")


n_samples, n_triplets = 10, 100
#  n_samples, n_triplets = 100, 1_000   # uncomment to use 10x more data
n_repeat = 10
estimator = embedding.SOE(1, n_init=1)

print(f"samples={n_samples} triplets={n_triplets} benchmark-repetitions={n_repeat}")
estimator = embedding.SOE(1, n_init=1)
benchmark_embedding(estimator, n_dims=1)
benchmark_embedding(estimator, n_dims=3)
benchmark_embedding(estimator, n_dims=10)

estimator = embedding.SOE(1, n_init=10)
benchmark_embedding(estimator, n_dims=1)
benchmark_embedding(estimator, n_dims=3)
benchmark_embedding(estimator, n_dims=10)
