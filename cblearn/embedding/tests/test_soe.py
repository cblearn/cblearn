import pytest
import numpy as np
from scipy.optimize import check_grad
from scipy.stats import ttest_rel
from scipy.spatial import distance_matrix

from cblearn.datasets import make_random_triplets
from cblearn.embedding import SOE


def test_soe_repeat():
    """ Test that the repeated initialization leads to stable results. """
    n, d = 20, 1  # optimization is most unstable for d = 1
    X = np.random.randn(n, d)

    # does repeated initialization lead to stable results?
    score_one = []
    score_repeat = []
    for i in range(10):
        T = make_random_triplets(X, size=2 * d * n * np.log(n), result_format='list-order',
                                 noise='normal', noise_options=dict(scale=0.1))
        T_test = make_random_triplets(X, size=10000, result_format='list-order',
                                      noise='normal', noise_options=dict(scale=0.1))
        soe_one = SOE(n_components=d, n_init=1)
        soe_repeat = SOE(n_components=d, n_init=10)
        score_one.append(soe_one.fit(T).score(T_test))
        score_repeat.append(soe_repeat.fit(T).score(T_test))

    stat = ttest_rel(score_one, score_repeat, alternative='less')
    assert stat.pvalue < 0.05


@pytest.mark.parametrize('n,d', [(20, 1), (25, 2), (30, 3)])
def test_soe_gradient(n, d):
    """ Test the gradient of the SOE loss function by comparing with local loss differences. """
    soe = SOE(n_components=d)

    def fun(x, x_shape, T, *args):
        X = x.reshape(x_shape)
        dist_pos = distance_matrix(X, X)[T[:, 0], T[:, 1]]
        dist_neg = distance_matrix(X, X)[T[:, 2], T[:, 3]]
        return soe._scipy_loss(X, T, dist_pos, dist_neg, *args)[0]

    def grad(x, x_shape, T, *args):
        X = x.reshape(x_shape)
        dist_pos = distance_matrix(X, X)[T[:, 0], T[:, 1]]
        dist_neg = distance_matrix(X, X)[T[:, 2], T[:, 3]]
        return soe._scipy_loss(X, T, dist_pos, dist_neg, *args)[1]

    X = np.random.randn(n, d)
    T = make_random_triplets(X, size=2*d*n*np.log(n), result_format='list-order')

    for ix_permut in ([1, 0, 0, 2], [0, 1, 0, 2], [0, 1, 2, 0], [1, 0, 2, 0]):
        for i in range(5):  # test at different points
            init = 4 * np.random.rand(n, d) - 2  # [-2, 2]
            args = [T[:, ix_permut]]
            assert check_grad(fun, grad, init.ravel(), init.shape, *args) < 1e-6
