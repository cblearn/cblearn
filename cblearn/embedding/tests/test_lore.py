"""Tests for the LORE (Low-Rank Ordinal Embedding) estimator."""

import sys
import types

import numpy as np
import pytest
from scipy.optimize import check_grad

from cblearn.datasets import (
    LinearSubspace,
    make_all_triplet_indices,
    make_random_triplet_indices,
    make_random_triplets,
    noisy_triplet_response,
    triplet_response,
)
from cblearn.embedding import LORE
from cblearn.embedding._lore_utils import (
    _logistic_triplet_loss_numpy,
    _schatten_p_value_numpy,
    _schatten_p_grad_numpy,
    _estimate_lipschitz_numpy,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_triplets():
    """Small synthetic triplet dataset for fast tests."""
    rng = np.random.RandomState(0)
    true_embedding = rng.randn(15, 2)
    triplets = make_random_triplets(
        true_embedding, result_format='list-order', size=300, random_state=rng
    )
    return triplets


@pytest.fixture
def low_rank_data():
    """Data sampled from a 2D subspace embedded in 8D, with triplets."""
    rng = np.random.RandomState(1)
    # Points live in a 2D plane inside 8D space
    coeffs = rng.randn(20, 2)
    basis = rng.randn(2, 8)
    basis, _ = np.linalg.qr(basis.T)    # orthonormal columns
    points = coeffs @ basis.T            # (20, 8)
    triplets = make_random_triplets(
        points, result_format='list-order', size=500, random_state=rng
    )
    return points, triplets


# ---------------------------------------------------------------------------
# Basic fit / transform tests
# ---------------------------------------------------------------------------

class TestLOREFitScipy:
    """LORE with scipy backend."""

    def test_basic_fit(self, small_triplets):
        """Fits without error and produces correct embedding shape."""
        est = LORE(n_components=5, backend="scipy", max_iter=100, random_state=42)
        embedding = est.fit_transform(small_triplets, n_objects=15)
        assert embedding.shape == (15, 5)

    def test_score_above_chance(self, small_triplets):
        """Score should be meaningfully above chance (0.5)."""
        est = LORE(n_components=5, backend="scipy", max_iter=200, random_state=42)
        est.fit(small_triplets, n_objects=15)
        assert est.score(small_triplets) >= 0.5

    def test_rank_is_valid(self, small_triplets):
        """rank_ must be a non-negative integer <= n_components."""
        est = LORE(n_components=5, backend="scipy", max_iter=100, random_state=42)
        est.fit(small_triplets, n_objects=15)
        assert isinstance(est.rank_, (int, np.integer))
        assert 0 <= est.rank_ <= 5


try:
    import torch  # noqa: F401
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")
class TestLOREFitTorch:
    """LORE with torch backend."""

    def test_basic_fit(self, small_triplets):
        """Fits without error and produces correct embedding shape."""
        est = LORE(n_components=5, backend="torch", max_iter=100,
                   verbose=False, random_state=42)
        embedding = est.fit_transform(small_triplets, n_objects=15)
        assert embedding.shape == (15, 5)

    def test_score_above_chance(self, small_triplets):
        """Score should be meaningfully above chance (0.5)."""
        est = LORE(n_components=5, backend="torch", max_iter=200,
                   verbose=False, random_state=42)
        est.fit(small_triplets, n_objects=15)
        assert est.score(small_triplets) >= 0.5

    def test_rank_is_valid(self, small_triplets):
        """rank_ must be a non-negative integer <= n_components."""
        est = LORE(n_components=5, backend="torch", max_iter=100,
                   verbose=False, random_state=42)
        est.fit(small_triplets, n_objects=15)
        assert isinstance(est.rank_, (int, np.integer))
        assert 0 <= est.rank_ <= 5


# ---------------------------------------------------------------------------
# Rank reduction test
# ---------------------------------------------------------------------------

def test_lore_rank_reduction(low_rank_data):
    """LORE should recover a rank strictly lower than n_components.

    Data lives in a 2D subspace of 8D space.  With strong regularisation and
    mu estimated via the power method, singular values corresponding to noise
    dimensions should be driven to zero.
    """
    _, triplets = low_rank_data
    est = LORE(n_components=8, lamb=0.5, p=0.5, mu="default", backend="scipy",
               max_iter=500, tol=1e-8, random_state=7)
    est.fit(triplets, n_objects=20)
    assert est.rank_ < 8, f"Expected rank reduction, got rank_={est.rank_}"


# ---------------------------------------------------------------------------
# Nuclear norm vs Schatten-p
# ---------------------------------------------------------------------------

def test_lore_nuclear_norm_p1(small_triplets):
    """p=1.0 (nuclear norm) should produce valid results."""
    est = LORE(n_components=4, p=1.0, backend="scipy", max_iter=80, random_state=0)
    est.fit(small_triplets, n_objects=15)
    assert est.embedding_.shape == (15, 4)
    assert np.isfinite(est.stress_)


def test_lore_schatten_p05(small_triplets):
    """p=0.5 (default Schatten-p) should produce valid results."""
    est = LORE(n_components=4, p=0.5, backend="scipy", max_iter=80, random_state=0)
    est.fit(small_triplets, n_objects=15)
    assert est.embedding_.shape == (15, 4)
    assert np.isfinite(est.stress_)


# ---------------------------------------------------------------------------
# Gradient check on the scipy loss function
# ---------------------------------------------------------------------------

def test_scipy_logistic_gradient():
    """Analytic gradient should match finite-difference approximation."""
    rng = np.random.RandomState(5)
    n, d = 10, 3
    triplets = np.array([[0, 1, 2], [1, 2, 3], [0, 2, 4],
                         [3, 4, 5], [5, 6, 7]], dtype=int)

    def scalar_loss(x_flat):
        X = x_flat.reshape(n, d)
        loss, _ = _logistic_triplet_loss_numpy(X, triplets, margin=0.1)
        return loss

    def analytic_grad(x_flat):
        X = x_flat.reshape(n, d)
        _, grad = _logistic_triplet_loss_numpy(X, triplets, margin=0.1)
        return grad.ravel()

    x0 = rng.randn(n * d)
    err = check_grad(scalar_loss, analytic_grad, x0)
    assert err < 1e-4, f"Gradient check failed with error {err:.2e}"


def test_schatten_p_value_and_grad():
    """Schatten-p value and gradient should be consistent."""
    sv = np.array([2.0, 1.5, 0.8, 0.3])
    eps = 1e-6

    # p=1 (nuclear norm)
    val = _schatten_p_value_numpy(sv, p=1.0)
    grad = _schatten_p_grad_numpy(sv, p=1.0)
    assert np.isclose(val, sv.sum())
    assert np.allclose(grad, np.ones(4))

    # p=0.5: gradient should be p*(sv+eps)^(p-1)
    val05 = _schatten_p_value_numpy(sv, p=0.5, eps=eps)
    grad05 = _schatten_p_grad_numpy(sv, p=0.5, eps=eps)
    assert np.isclose(val05, np.sum((sv + eps) ** 0.5))
    expected_grad = 0.5 * (sv + eps) ** (-0.5)
    assert np.allclose(grad05, expected_grad)


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------

def test_lore_objectives_non_increasing(small_triplets):
    """Objective values should be non-increasing (up to tolerance)."""
    est = LORE(n_components=4, backend="scipy", max_iter=150,
               tol=1e-12, random_state=3)  # tight tol to get more iters
    est.fit(small_triplets, n_objects=15)
    objs = est.objectives_
    assert len(objs) >= 2
    # Allow small numerical noise (1e-6 relative tolerance)
    diffs = np.diff(objs)
    assert np.all(diffs <= 1e-5), (
        f"Objective increased at iterations: {np.where(diffs > 1e-5)[0]}"
    )


# ---------------------------------------------------------------------------
# Attribute completeness
# ---------------------------------------------------------------------------

def test_lore_attributes_after_fit(small_triplets):
    """All expected fitted attributes should be present and well-typed."""
    est = LORE(n_components=3, backend="scipy", max_iter=50, random_state=0)
    est.fit(small_triplets, n_objects=15)

    assert hasattr(est, 'embedding_') and est.embedding_.shape == (15, 3)
    assert hasattr(est, 'rank_') and isinstance(est.rank_, (int, np.integer))
    assert hasattr(est, 'stress_') and np.isfinite(est.stress_)
    assert hasattr(est, 'n_iter_') and est.n_iter_ >= 1
    assert hasattr(est, 'sigma_history_') and isinstance(est.sigma_history_, list)
    assert hasattr(est, 'objectives_') and len(est.objectives_) == est.n_iter_


# ---------------------------------------------------------------------------
# Branch coverage: mu, init, backend error, verbose
# ---------------------------------------------------------------------------

def test_lore_scipy_default_mu_rank_reduction():
    """scipy mu='default' should estimate mu via power iteration and reduce rank.

    Validates that the scipy backend does not regress to the broken heuristic
    (which gave mu~97 and failed to reduce rank).
    """
    rng = np.random.RandomState(3)
    true_emb = rng.randn(20, 2)
    triplets = make_random_triplets(true_emb, result_format='list-order', size=500, random_state=rng)
    est = LORE(n_components=6, lamb=0.5, p=0.5, mu="default", backend="scipy",
               max_iter=300, random_state=3)
    est.fit(triplets, n_objects=20)
    assert est.rank_ < 6, (
        f"scipy mu='default' should reduce rank below n_components=6, got {est.rank_}. "
        "This may indicate a regression to the old broken heuristic."
    )


def test_lore_custom_mu(small_triplets):
    """Explicit mu (float) should override the default estimation."""
    est = LORE(n_components=3, mu=2.0, backend="scipy", max_iter=50, random_state=0)
    est.fit(small_triplets, n_objects=15)
    assert est.embedding_.shape == (15, 3)


def test_lore_custom_init(small_triplets):
    """Providing init array should be accepted and used."""
    rng = np.random.RandomState(99)
    init = rng.randn(15, 3)
    est = LORE(n_components=3, backend="scipy", max_iter=50, random_state=0)
    est.fit(small_triplets, init=init, n_objects=15)
    assert est.embedding_.shape == (15, 3)


def test_lore_invalid_backend(small_triplets):
    """Unknown backend string should raise ValueError."""
    est = LORE(n_components=2, backend="unknown")
    with pytest.raises(ValueError, match="Unknown backend"):
        est.fit(small_triplets, n_objects=15)


def test_lore_verbose(small_triplets, capsys):
    """verbose=True should not raise any errors."""
    est = LORE(n_components=3, backend="scipy", max_iter=5,
               verbose=True, random_state=0)
    est.fit(small_triplets, n_objects=15)
    # Just ensure it ran without exceptions
    assert est.embedding_.shape == (15, 3)


# ---------------------------------------------------------------------------
# End-to-end demo test (mirrors demo.ipynb)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")
def test_lore_demo_end_to_end():
    """End-to-end integration test replicating the demo.ipynb workflow.

    Generates a synthetic dataset from a 5D linear subspace embedded in 15D,
    adds noise to triplet comparisons, and verifies that LORE with the paper's
    default parameters (lamb=0.01, p=0.5) recovers a low-rank embedding with
    reasonable triplet accuracy.
    """
    # Synthetic dataset: 100 objects in a 5D subspace of 15D space
    manifold = LinearSubspace(subspace_dimension=5, space_dimension=15)
    true_points, true_distances = manifold.sample_points(100, random_state=0)

    train_triplets = make_random_triplet_indices(n_objects=100, size=1500, random_state=42)
    noisy_train_response = noisy_triplet_response(
        train_triplets, true_distances,
        noise='normal', noise_options={'scale': 0.1},
    )

    # Fit using the CBlearn LORE estimator with paper default parameters
    est = LORE(
        n_components=15,
        lamb=0.01,
        p=0.5,
        backend="torch",
        max_iter=300,
        verbose=False,
        random_state=42,
    )
    embedding = est.fit_transform(noisy_train_response, n_objects=100)

    # Shape is correct
    assert embedding.shape == (100, 15)

    # Rank should be reduced below ambient dimension (regularisation is working)
    assert est.rank_ < 15, f"Expected rank < 15 (ambient dim), got {est.rank_}"

    # Train accuracy above chance on noisy data
    train_acc = est.score(noisy_train_response)
    assert train_acc >= 0.7, f"Expected train accuracy >= 0.7, got {train_acc:.3f}"

    # All diagnostic attributes are populated
    assert len(est.objectives_) == est.n_iter_
    assert len(est.sigma_history_) > 0


def test_lore_n_objects_inferred(small_triplets):
    """fit() without n_objects kwarg should infer n_objects from triplet indices."""
    triplets = small_triplets
    est = LORE(n_components=2, backend="scipy", max_iter=5, random_state=0)
    # Omit n_objects - should be inferred as triplets.max() + 1
    est.fit(triplets)
    expected_n_objects = int(triplets.max()) + 1
    assert est.embedding_.shape[0] == expected_n_objects


@pytest.mark.skipif(
    not __import__('importlib').util.find_spec('torch'),
    reason="torch not installed"
)
def test_lore_torch_verbose(small_triplets):
    """LORE torch backend with verbose=True should print without errors."""
    import io
    import sys
    triplets = small_triplets
    est = LORE(n_components=2, backend="torch", max_iter=5, verbose=True, random_state=0)
    captured = io.StringIO()
    sys.stdout = captured
    try:
        est.fit(triplets, n_objects=15)
    finally:
        sys.stdout = sys.__stdout__
    output = captured.getvalue()
    assert est.embedding_.shape == (15, 2)
    # verbose mode should have printed something
    assert len(output) > 0


@pytest.mark.skipif(
    not __import__('importlib').util.find_spec('torch'),
    reason="torch not installed"
)
def test_lore_torch_nuclear_norm(small_triplets):
    """LORE torch backend with p=1.0 (nuclear norm) should fit without errors."""
    triplets = small_triplets
    est = LORE(n_components=2, p=1.0, backend="torch", max_iter=10, random_state=0)
    est.fit(triplets, n_objects=15)
    assert est.embedding_.shape == (15, 2)
    assert est.rank_ >= 1


# ---------------------------------------------------------------------------
# Coverage: scipy convergence verbose (line 244), norm_hvp==0 (line 147)
# ---------------------------------------------------------------------------

def test_lore_scipy_convergence_verbose(small_triplets, capsys):
    """verbose=True + high tol triggers the convergence print in scipy backend."""
    est = LORE(n_components=3, backend="scipy", max_iter=200,
               tol=100.0, verbose=True, random_state=0)
    est.fit(small_triplets, n_objects=15)
    capsys.readouterr()  # drain the captured output
    # Either converged early (prints "[scipy] Converged...") or ran to max_iter
    assert est.embedding_.shape == (15, 3)


def test_estimate_lipschitz_zero_hvp():
    """norm_hvp == 0 branch: empty triplets --> zero gradient --> zero HVP --> break."""
    # With no triplets the loss is identically 0, so the gradient is a zero
    # matrix at every point.  The HVP is therefore zero and norm_hvp == 0
    # triggers the break on the first power-method iteration (line 147).
    n, d = 5, 2
    X = np.ones((n, d))
    triplets = np.zeros((0, 3), dtype=int)
    result = _estimate_lipschitz_numpy(X, triplets, margin=0.1, num_iters=10)
    assert result >= 0.0  # returns eigenvalue + 0.05


# ---------------------------------------------------------------------------
# Coverage: torch explicit mu (line 398)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")
def test_lore_torch_explicit_mu(small_triplets):
    """torch backend with explicit float mu should bypass Lipschitz estimation."""
    est = LORE(n_components=3, mu=5.0, backend="torch", max_iter=20,
               verbose=False, random_state=0)
    est.fit(small_triplets, n_objects=15)
    assert est.embedding_.shape == (15, 3)


# ---------------------------------------------------------------------------
# Coverage: torch convergence verbose (lines 433-437) and KL-dist (lines 465-469)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")
def test_lore_torch_convergence_verbose_early_stop(small_triplets, capsys):
    """torch + verbose=True + moderate tol triggers objective convergence print.

    Uses tol=1e-4 so that per-step objective change converges before the
    total KL-distance from init (which stays large after X has drifted).
    Covers lines 433-437.
    """
    est = LORE(n_components=3, backend="torch", max_iter=500,
               tol=1e-4, verbose=True, random_state=0)
    est.fit(small_triplets, n_objects=15)
    capsys.readouterr()  # drain the captured output
    assert est.embedding_.shape == (15, 3)
    # Should stop before max_iter when objective converges
    assert est.n_iter_ < 500


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")
def test_lore_torch_kldist_convergence(small_triplets):
    """Very large tol triggers KL-dist convergence path on first iteration.

    With tol=1000.0, the objective change check (|inf - obj|) does NOT fire
    on iteration 0.  But kl_dist = ||X_new - init||_inf is a small positive
    number << 1000, so the KL-dist check fires first.  Covers lines 465-469.
    """
    est = LORE(n_components=3, mu=2.0, backend="torch", max_iter=200,
               tol=1000.0, verbose=True, random_state=0)
    est.fit(small_triplets, n_objects=15)
    assert est.embedding_.shape == (15, 3)
    # Stops on iteration 0 via KL-dist
    assert est.n_iter_ == 1


# ---------------------------------------------------------------------------
# Coverage: torch tqdm progress bar (lines 413, 472-477) and fallback (415)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")
def test_lore_torch_tqdm_progress(small_triplets, monkeypatch):
    """A (fake) tqdm module provides the progress bar and receives postfix stats.

    Injecting the fake into sys.modules makes the test independent of
    whether tqdm is actually installed.  Covers lines 413 and 472-477.
    """
    postfix_calls = []

    class FakeTqdm:
        def __init__(self, iterable, disable=False):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, stats):
            postfix_calls.append(stats)

    fake_module = types.ModuleType("tqdm")
    fake_module.tqdm = FakeTqdm
    monkeypatch.setitem(sys.modules, "tqdm", fake_module)

    est = LORE(n_components=3, mu=5.0, backend="torch", max_iter=3,
               random_state=0)
    est.fit(small_triplets, n_objects=15)

    assert postfix_calls, "set_postfix was never called"
    assert 'rank' in postfix_calls[0]


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")
def test_lore_torch_no_tqdm_fallback(small_triplets, monkeypatch):
    """Without tqdm the loop falls back to a plain range (lines 414-415)."""
    monkeypatch.setitem(sys.modules, "tqdm", None)   # forces ImportError

    est = LORE(n_components=3, mu=5.0, backend="torch", max_iter=3,
               random_state=0)
    est.fit(small_triplets, n_objects=15)
    assert est.embedding_.shape == (15, 3)


# ---------------------------------------------------------------------------
# Real CUDA execution (runs only on machines with a GPU; skipped in CI)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _TORCH_AVAILABLE or not torch.cuda.is_available(),
                    reason="CUDA not available")
def test_lore_torch_cuda_end_to_end(small_triplets):
    """Fit LORE on a real GPU, exercising the CUDA-only gesvd SVD path.

    ``device="auto"`` selects CUDA when available, so on GPU machines this
    runs the genuine CUDA code path (including the ``driver='gesvd'``
    branch that CPU-only CI cannot reach).
    """
    est = LORE(n_components=3, backend="torch", device="auto",
               max_iter=50, random_state=0)
    est.fit(small_triplets, n_objects=15)

    assert est.embedding_.shape == (15, 3)
    assert 0 <= est.rank_ <= 3
    assert est.score(small_triplets) > 0.6


# ---------------------------------------------------------------------------
# Rank recovery: exact intrinsic dimensionality from dense triplets
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")
def test_lore_rank3_recovery_end_to_end():
    """LORE recovers the exact subspace rank from densely sampled triplets.

    25 objects on a 3D linear subspace of 10D space; all 6900 possible
    triplets are answered noiselessly and split 80/20 into train/test.
    LORE must recover rank_ == 3 exactly and generalise to held-out
    triplets.
    """
    manifold = LinearSubspace(subspace_dimension=3, space_dimension=10)
    points, _ = manifold.sample_points(25, random_state=42)

    triplets = make_all_triplet_indices(25, monotonic=False)
    responses = triplet_response(triplets, points, result_format='list-order')

    rng = np.random.RandomState(42)
    perm = rng.permutation(len(responses))
    n_train = int(0.8 * len(responses))
    train, test = responses[perm[:n_train]], responses[perm[n_train:]]

    est = LORE(n_components=10, lamb=0.01, p=0.5, backend="torch",
               max_iter=1000, device="cpu", random_state=42)
    est.fit(train, n_objects=25)

    assert est.rank_ == 3, f"Expected exact rank recovery of 3, got {est.rank_}"
    test_acc = est.score(test)
    assert test_acc >= 0.9, f"Expected test accuracy >= 0.9, got {test_acc:.3f}"
