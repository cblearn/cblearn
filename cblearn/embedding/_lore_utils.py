"""Training utilities for the LORE embedding algorithm.

This module is self-contained: it does **not** import from any external
``lore`` package.  All algorithm logic (logistic triplet loss, Schatten-p
regularisation, proximal gradient training loop) is implemented directly here
for both backends.

Two backends are supported:

- ``torch``: Uses PyTorch autograd for gradient and Lipschitz-constant
  computation via the power method.  Supports CUDA GPUs.
- ``scipy``: Pure NumPy/SciPy re-implementation of the same proximal gradient
  algorithm, with analytically derived gradients.

Both produce identical mathematical results up to numerical precision.
"""

import numpy as np
from scipy.optimize import OptimizeResult


# ---------------------------------------------------------------------------
# Scipy backend helpers (pure NumPy)
# ---------------------------------------------------------------------------

def _logistic_triplet_loss_numpy(X, triplets, margin, eps=1e-3):
    """Logistic triplet loss and its gradient w.r.t. the embedding X.

    Uses the smooth distance ``d(x, y) = sqrt(||x - y||^2 + epsilon^2)`` so that the
    gradient is consistent with the finite-difference approximation used by
    ``scipy.optimize.check_grad``.

    Args:
        X: Embedding matrix of shape (n_objects, n_dims).
        triplets: Integer array of shape (n_triplets, 3).  Column order is
            [anchor, closer, farther].
        margin: Scalar margin added to the distance difference.
        eps: Smoothing constant inside the square root (default 1e-3).

    Returns:
        Tuple ``(loss, gradient)`` where loss is a scalar float and gradient
        has the same shape as X.
    """
    anchors = X[triplets[:, 0]]
    positives = X[triplets[:, 1]]
    negatives = X[triplets[:, 2]]

    diff_ap = anchors - positives          # (n_triplets, n_dims)
    diff_an = anchors - negatives

    # Smooth Euclidean distance: sqrt(||.||^2 + epsilon^2)
    dist_ap = np.sqrt(np.sum(diff_ap ** 2, axis=1) + eps ** 2)   # (n_triplets,)
    dist_an = np.sqrt(np.sum(diff_an ** 2, axis=1) + eps ** 2)

    # Logistic triplet loss: mean log(1 + exp(d_ap - d_an + margin))
    t = dist_ap - dist_an + margin
    loss = float(np.mean(np.logaddexp(0.0, t)))   # numerically stable

    # Gradient ---------------------------------------------------------------
    # d(loss)/d(t_i) = sigmoid(t_i) / N
    sigmoid_t = 1.0 / (1.0 + np.exp(-np.clip(t, -500.0, 500.0)))
    scale = sigmoid_t / len(triplets)           # (n_triplets,)

    # d(smooth_dist_ap)/d(anchor) = diff_ap / dist_ap  (and similarly for an)
    d_loss_dap = scale / dist_ap      # (n_triplets,)
    d_loss_dan = -scale / dist_an

    grad = np.zeros_like(X)
    np.add.at(grad, triplets[:, 0], d_loss_dap[:, None] * diff_ap)
    np.add.at(grad, triplets[:, 1], -d_loss_dap[:, None] * diff_ap)
    np.add.at(grad, triplets[:, 0], d_loss_dan[:, None] * diff_an)
    np.add.at(grad, triplets[:, 2], -d_loss_dan[:, None] * diff_an)

    return loss, grad


def _schatten_p_value_numpy(sv, p, eps=1e-6):
    """Compute the (unnormed) Schatten-p value of a singular value vector.

    Args:
        sv: 1-D array of singular values.
        p: Exponent.  ``p=1`` is the nuclear norm.
        eps: Regularisation term for non-integer p (avoids 0^p issues).

    Returns:
        Scalar value.
    """
    if p == 1.0:
        return float(np.sum(sv))
    return float(np.sum((sv + eps) ** p))


def _schatten_p_grad_numpy(sv, p, eps=1e-6):
    """Gradient of the (unnormed) Schatten-p value w.r.t. singular values.

    Args:
        sv: 1-D array of singular values.
        p: Exponent.  ``p=1`` gives a vector of ones (nuclear norm).
        eps: Regularisation term, matching ``_schatten_p_value_numpy``.

    Returns:
        Array of the same shape as sv.
    """
    if p == 1.0:
        return np.ones_like(sv)
    return p * (sv + eps) ** (p - 1.0)


# ---------------------------------------------------------------------------
# Scipy backend - proximal gradient training loop
# ---------------------------------------------------------------------------

def _estimate_lipschitz_numpy(X, triplets, margin, num_iters=100, tol=1e-6, fd_eps=1e-5):
    """Estimate the Lipschitz constant of the logistic triplet loss gradient.

    Uses the power method with finite-difference Hessian-vector products,
    mirroring ``_estimate_lipschitz_torch`` for the scipy backend.

    Args:
        X: Embedding matrix of shape (n_objects, n_dims).
        triplets: Integer array of shape (n_triplets, 3).
        margin: Triplet loss margin.
        num_iters: Maximum power-method iterations.
        tol: Convergence tolerance on eigenvalue change.
        fd_eps: Finite-difference step size for HVP approximation.

    Returns:
        Estimated Lipschitz constant (float) with a small safety margin.
    """
    rng = np.random.default_rng(0)
    v = rng.standard_normal(X.shape)
    v /= np.linalg.norm(v)

    _, grad0 = _logistic_triplet_loss_numpy(X, triplets, margin)
    eigenvalue = 0.0

    for _ in range(num_iters):
        # Finite-difference Hessian-vector product: (grad_f(X + epsilon*V) - grad_f(X)) / epsilon
        _, grad_plus = _logistic_triplet_loss_numpy(X + fd_eps * v, triplets, margin)
        hvp = (grad_plus - grad0) / fd_eps

        v_flat, hvp_flat = v.ravel(), hvp.ravel()
        new_eigenvalue = float(np.dot(v_flat, hvp_flat) / np.dot(v_flat, v_flat))

        norm_hvp = np.linalg.norm(hvp)
        if norm_hvp == 0:
            break
        v = hvp / norm_hvp

        if abs(new_eigenvalue - eigenvalue) < tol:
            break
        eigenvalue = abs(new_eigenvalue)

        # Update base gradient to current point for next iteration
        _, grad0 = _logistic_triplet_loss_numpy(X, triplets, margin)

    return eigenvalue + 0.05


def _lore_train_scipy(init, triplets, lamb, p, margin, mu,
                      max_iter, tol, zero, seed, verbose):
    """Proximal gradient training loop for LORE (pure NumPy/SciPy backend).

    Mirrors the algorithm described in Anand et al. (ICLR 2026) but uses
    NumPy SVD and analytically derived gradients instead of PyTorch autograd.

    Args:
        init: Initial embedding, numpy array of shape (n_objects, n_components).
        triplets: Integer array of shape (n_triplets, 3) in list-order format.
        lamb: Regularisation strength lambda.
        p: Schatten-p exponent.
        margin: Logistic triplet loss margin.
        mu: Lipschitz step-size parameter.  ``"default"`` estimates via the
            power method using finite-difference Hessian-vector products.
        max_iter: Maximum number of iterations.
        tol: Convergence tolerance on the change in objective value.
        zero: Threshold below which singular values are set to zero.
        seed: Integer random seed (reserved for future stochastic extensions).
        verbose: If True, print progress every 100 iterations.

    Returns:
        scipy.optimize.OptimizeResult with attributes:
            x: Final embedding array (n_objects, n_components).
            fun: Final objective value.
            nit: Number of iterations performed.
            success: True if converged before max_iter.
            message: Convergence message.
            ranks, objectives, f_losses, g_losses, sigma_history: lists
                tracking algorithm progress.
    """
    n_objects, n_components = init.shape
    X = init.copy()

    if mu == "default":
        mu = _estimate_lipschitz_numpy(X, triplets, margin)
        if verbose:
            print(f"[scipy] Estimated Lipschitz constant (mu): {mu:.4f}")
    else:
        mu = float(mu)

    objectives = []
    f_losses = []
    g_losses = []
    ranks = []
    sigma_history = []

    prev_objective = np.inf
    converged = False
    message = f"Reached max_iter={max_iter}"

    for iteration in range(max_iter):
        f_loss, f_grad = _logistic_triplet_loss_numpy(X, triplets, margin)

        U, sv, Vt = np.linalg.svd(X - f_grad / mu, full_matrices=False)

        g_val = _schatten_p_value_numpy(sv, p)
        g_grad = _schatten_p_grad_numpy(sv, p)
        g_loss = lamb * g_val

        current_objective = f_loss + g_loss
        objectives.append(float(current_objective))
        f_losses.append(float(f_loss))
        g_losses.append(float(g_loss))

        # Proximal / singular-value thresholding step
        new_sv = sv - (lamb / mu) * g_grad
        new_sv = np.maximum(new_sv, 0.0)
        new_sv[new_sv < zero] = 0.0

        rank = int(np.sum(new_sv > 0))
        ranks.append(rank)
        sigma_history.append(new_sv.copy())

        X = U @ np.diag(new_sv) @ Vt

        if verbose and iteration % 100 == 0:
            print(f"  iter {iteration:4d} | obj={current_objective:.6f} "
                  f"| f={f_loss:.6f} | g={g_loss:.6f} | rank={rank}")

        if abs(prev_objective - current_objective) < tol:
            converged = True
            message = f"Converged after {iteration + 1} iterations"
            if verbose:
                print(f"[scipy] {message}")
            break
        prev_objective = current_objective

    return OptimizeResult(
        x=X,
        fun=objectives[-1] if objectives else np.nan,
        nit=len(objectives),
        success=converged,
        message=message,
        ranks=ranks,
        objectives=objectives,
        f_losses=f_losses,
        g_losses=g_losses,
        sigma_history=sigma_history,
    )


# ---------------------------------------------------------------------------
# Torch backend - self-contained proximal gradient training loop
# ---------------------------------------------------------------------------

def _estimate_lipschitz_torch(embedding, triplets_t, margin, num_iters=100, tol=1e-6):
    """Estimate the Lipschitz constant of the logistic triplet loss gradient.

    Uses the power method on Hessian-vector products via PyTorch autograd.
    This is equivalent to the largest eigenvalue of the Hessian matrix.

    Args:
        embedding: Torch tensor of shape (n_objects, n_dims), requires_grad.
        triplets_t: Long tensor of shape (n_triplets, 3).
        margin: Triplet loss margin.
        num_iters: Maximum power-method iterations.
        tol: Convergence tolerance on eigenvalue change.

    Returns:
        Estimated Lipschitz constant (float) with a small safety margin added.
    """
    import torch  # lazy import

    v = torch.rand_like(embedding)
    v = v / v.norm()
    largest_eigenvalue = torch.tensor(0.0, device=embedding.device)

    for _ in range(num_iters):
        embedding = embedding.detach().requires_grad_(True)
        loss = _logistic_triplet_loss_torch(embedding, triplets_t, margin)
        grad = torch.autograd.grad(loss, embedding, create_graph=True)[0]
        hvp = torch.autograd.grad((grad * v).sum(), embedding, retain_graph=False)[0]

        new_eigenvalue = (
            torch.dot(v.flatten(), hvp.flatten())
            / torch.dot(v.flatten(), v.flatten())
        )
        v = (hvp / hvp.norm()).detach()

        if torch.abs(new_eigenvalue - largest_eigenvalue) < tol:
            break
        largest_eigenvalue = new_eigenvalue.abs()

    return largest_eigenvalue.item() + 0.05


def _logistic_triplet_loss_torch(embedding, triplets_t, margin, eps=1e-3):
    """Logistic triplet loss for the torch backend.

    Uses the smooth distance ``sqrt(||x - y||^2 + epsilon^2)`` matching the scipy
    backend, so that both backends produce consistent results.

    Args:
        embedding: Float tensor of shape (n_objects, n_dims).
        triplets_t: Long tensor of shape (n_triplets, 3).
        margin: Scalar margin.
        eps: Smoothing constant for the distance (default 1e-3).

    Returns:
        Scalar loss tensor.
    """
    import torch  # lazy import

    X = embedding[triplets_t]
    anchor, positive, negative = X[:, 0, :], X[:, 1, :], X[:, 2, :]
    dist_ap = torch.sqrt(torch.sum((anchor - positive) ** 2, dim=1) + eps ** 2)
    dist_an = torch.sqrt(torch.sum((anchor - negative) ** 2, dim=1) + eps ** 2)
    t = dist_ap - dist_an + margin
    return torch.log1p(torch.exp(t)).mean()


def _schatten_p_torch(sv, p, eps=1e-6):
    """Compute the (unnormed) Schatten-p value of a singular value tensor.

    Args:
        sv: 1-D float tensor of singular values.
        p: Exponent.  ``p=1`` is the nuclear norm.
        eps: Regularisation term for non-integer p.

    Returns:
        Scalar tensor.
    """
    import torch  # lazy import

    if p == 1.0:
        return torch.sum(sv)
    eps_t = torch.tensor(eps, dtype=sv.dtype, device=sv.device)
    return torch.sum((sv + eps_t) ** p)


def _lore_train_torch(init, triplets, lamb, p, margin, mu,
                      max_iter, tol, zero, seed, verbose, device):
    """Proximal gradient training loop for LORE (PyTorch backend).

    Implements the algorithm from Anand et al. (ICLR 2026) using PyTorch
    autograd for gradient computation.  Supports CPU and CUDA GPU execution.

    This function is self-contained: it does **not** import from any external
    ``lore`` package.

    Args:
        init: Initial embedding, numpy array of shape (n_objects, n_components).
        triplets: Integer array of shape (n_triplets, 3) in list-order format.
        lamb: Regularisation strength lambda.
        p: Schatten-p exponent.
        margin: Logistic triplet loss margin.
        mu: Lipschitz step-size.  ``"default"`` estimates via the power method.
        max_iter: Maximum number of iterations.
        tol: Convergence tolerance.
        zero: Singular-value zeroing threshold.
        seed: Integer random seed.
        verbose: If True, show a tqdm progress bar.
        device: PyTorch device string (``"auto"``, ``"cpu"``, ``"cuda"``).

    Returns:
        scipy.optimize.OptimizeResult with the same attributes as
        ``_lore_train_scipy``.
    """
    import torch  # lazy import - keeps torch optional for scipy-only users
    from cblearn.embedding._torch_utils import torch_device

    dev = torch_device(device)

    if seed is not None:
        torch.manual_seed(seed)   # also seeds the CUDA generators

    triplets_t = torch.tensor(triplets, dtype=torch.long, device=dev)
    X = torch.tensor(init, dtype=torch.float32, device=dev)
    init_tensor = X.clone()   # kept for KL-distance convergence check

    # --- Lipschitz constant estimation --------------------------------------
    if mu == "default":
        X_clone = X.clone().requires_grad_(True)
        mu_val = _estimate_lipschitz_torch(X_clone, triplets_t, margin)
        if verbose:
            print(f"Estimated Lipschitz constant (mu): {mu_val:.4f}")
    else:
        mu_val = float(mu)

    # --- Training loop ------------------------------------------------------
    objectives = []
    f_losses = []
    g_losses = []
    ranks = []
    sigma_history = []

    prev_objective = torch.tensor(float('inf'))
    converged = False
    message = f"Reached max_iter={max_iter}"

    try:
        from tqdm import tqdm as _tqdm
        pbar = _tqdm(range(max_iter), disable=not verbose)
    except ImportError:
        pbar = range(max_iter)

    for iteration in pbar:
        X = X.detach().requires_grad_(True)

        # Forward pass
        f_loss = _logistic_triplet_loss_torch(X, triplets_t, margin)
        sv = torch.linalg.svdvals(X)
        g_sv = _schatten_p_torch(sv, p)
        g_loss = lamb * g_sv
        current_objective = f_loss + g_loss

        objectives.append(current_objective.item())
        f_losses.append(f_loss.item())
        g_losses.append(g_loss.item())

        # Convergence check on objective change
        if torch.abs(prev_objective - current_objective) < tol:
            converged = True
            message = f"Converged after {iteration + 1} iterations"
            if verbose:
                print(f"\n{message}")
            break
        prev_objective = current_objective.detach()

        # Compute gradients
        f_grad = torch.autograd.grad(f_loss, X, retain_graph=False)[0]
        g_grad = torch.autograd.grad(g_sv, sv, retain_graph=False)[0]

        # Proximal / singular-value thresholding step
        with torch.no_grad():
            svd_input = X - f_grad / mu_val
            svd_kwargs = {'full_matrices': False}
            if svd_input.is_cuda:  # pragma: no cover - CUDA-only; exercised by the GPU-gated test
                svd_kwargs['driver'] = 'gesvd'
            U, S, Vt = torch.linalg.svd(svd_input, **svd_kwargs)
            new_S = S - (lamb / mu_val) * g_grad
            rank_mask = new_S > zero
            masked_S = new_S * rank_mask

            # Keep singular values sorted for the next g_grad computation
            sorted_S, _ = torch.sort(masked_S, descending=True)
            sigma_history.append(sorted_S.cpu().numpy().copy())

            X = U @ torch.diag(masked_S) @ Vt
            ranks.append(int(rank_mask.sum().item()))

            # Secondary convergence: KL distance from initial embedding
            kl_dist = torch.linalg.matrix_norm(X - init_tensor, ord=float('inf'))
            if kl_dist < tol:
                converged = True
                message = f"KL-dist converged after {iteration + 1} iterations"
                if verbose:
                    print(f"\n{message}")
                break

        if hasattr(pbar, 'set_postfix'):
            pbar.set_postfix({
                'f': f"{f_loss.item():.4f}",
                'g': f"{g_loss.item():.4f}",
                'obj': f"{current_objective.item():.4f}",
                'rank': ranks[-1] if ranks else '?',
            })

    return OptimizeResult(
        x=X.detach().cpu().numpy(),
        fun=objectives[-1] if objectives else float('nan'),
        nit=len(objectives),
        success=converged,
        message=message,
        ranks=ranks,
        objectives=objectives,
        f_losses=f_losses,
        g_losses=g_losses,
        sigma_history=sigma_history,
    )
