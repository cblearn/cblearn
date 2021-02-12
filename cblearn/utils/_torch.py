from typing import Sequence, Callable

import numpy as np
import scipy


def assert_torch_is_available() -> None:
    """ Check if the torch module is installed and raise error otherwise.

        The error message contains instructions how to install the pytorch package.

        Raises:
            ModuleNotFoundError: If torch package is not found.
    """
    try:
        import torch  # noqa: F401  We do not use torch here on purpose
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(f"{e}.\n\n"
                                  "This function of cblearn requires the installation of the pytorch package.\n"
                                  "To install pytorch, visit https://pytorch.org/get-started/locally/\n"
                                  "or run either of the following commands:\n"
                                  "    pip install cblearn[torch]\n"
                                  "    pip install torch\n"
                                  "    conda install -c=conda-forge pytorch\n")


def torch_minimize_lbfgs(objective: Callable, init: np.ndarray, args: Sequence, device: str, max_iter: int
                         ) -> 'scipy.optimize.OptimizeResult':
    """ Pytorch minimization routine using L-BFGS.

        This function is aims to be a pytorch version of :func:`scipy.optimize.minimize`.

        Args:
            objective:
                Loss function to minimize.
                Function argument is the current parameters and optional additional argument.
            init:
                The initial parameter values.
            args:
                Sequence of additional arguments, passed to the objective.
            device:
                Device to run the minimization on, usually "cpu" or "cuda".
                "auto" uses "cuda", if available.
            max_iter:
                The maximum number of optimizer iteration.
        Returns:
            Dict-like object containing status and result information
            of the optimization.
    """
    import torch  # pytorch is an optional dependency of the library

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    X = torch.tensor(init, requires_grad=True).to(device)
    args = [torch.tensor(a).to(device) for a in args]
    factr = 1e7 * np.finfo(float).eps

    optimizer = torch.optim.LBFGS([X])
    loss = float("inf")
    success, message = True, ""
    for n_iter in range(max_iter):
        def closure():
            optimizer.zero_grad()
            loss = objective(X, *args)
            loss.backward()
            return loss

        optimizer.step(closure)
        prev_loss = loss
        loss = optimizer.state[X]['prev_loss']
        if abs(prev_loss - loss) / max(abs(loss), abs(prev_loss), 1) < factr:
            break
    else:
        success = False
        message = "LBFGS did not converge."

    return scipy.optimize.OptimizeResult(
        x=X.cpu().detach().numpy(), fun=loss, nit=n_iter,
        success=success, message=message)
