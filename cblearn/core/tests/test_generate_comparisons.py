import numpy as np
from cblearn.core import uniform_index_tuples


def test_uniform_index_tuples():
    o = 10
    n = 1_000_000
    c = 6
    a = uniform_index_tuples(o, n, c)
    assert a.shape == (n, c)
    for c in range(a.shape[1]):
        binc = np.bincount(a[:, c].ravel().astype(int), minlength=o)
        np.testing.assert_array_less(n // o * 0.99, binc)
        np.testing.assert_array_less(binc, n // o * 1.01)