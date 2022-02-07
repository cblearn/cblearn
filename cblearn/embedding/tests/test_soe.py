import pytest
import numpy as np
from scipy.optimize import check_grad
from cblearn.datasets import make_random_triplets

from cblearn.embedding._soe import SOE, _soe_loss


def test_soe_repeat():
    """ Test that the repeated initialization leads to stable results. """
    n, d = 20, 1  # optimization is most unstable for d = 1
    X = np.random.randn(n, d)
    T = make_random_triplets(X, size=2*d*n*np.log(n), result_format='list-order',
                             noise='normal', noise_options=dict(scale=0.1))
    T_test = make_random_triplets(X, size=10000, result_format='list-order',
                             noise='normal', noise_options=dict(scale=0.1))
    accuracy_threshold = 0.8  # empirically determined

    # is this an unstable condition? Then single inititalization should sometimes have low accuracy.
    with pytest.raises(AssertionError):
        soe_one = SOE(n_components=d, n_init=1)
        for _ in range(3):
            soe_one.fit(T)
            assert soe_one.score(T_test) > accuracy_threshold

    # does repeated initialization lead to stable results?
    soe_repeat = SOE(n_components=d, n_init=10)
    for _ in range(3):
        stress = soe_repeat.fit(T).stress_
        assert soe_repeat.score(T_test) > accuracy_threshold


@pytest.mark.parametrize('n,d', [(20, 1), (50, 2), (100, 3)])
def test_soe_gradient(n, d):
    """ Test the gradient of the SOE loss function by comparing with local loss differences. """
    def fun(x, *args):
        return _soe_loss(x, *args)[0]

    def grad(x, *args):
        return _soe_loss(x, *args)[1]

    X = np.random.randn(n, d)
    T = make_random_triplets(X, size=2*d*n*np.log(n), result_format='list-order')
    for i in range(10):  # test at 10 different points in the param space
        init = 4 * np.random.rand(n, d) - 2  # [-2, 2]
        args = [init.ravel(), X.shape, T[:, [1, 0, 0, 2]], 0.1]
        assert check_grad(fun, grad, *args) < 1e-6
