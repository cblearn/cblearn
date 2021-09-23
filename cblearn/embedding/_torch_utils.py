from typing import Sequence, Callable, Optional

import numpy as np
import scipy


def torch_device(device: str) -> str:
    import torch

    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"


def torch_minimize_kernel(method, objective, init, **kwargs):
    import torch

    kernel_init = init @ init.T * .1
    dim = init.shape[1]

    def kernel_objective(K, *args, **kwargs):
        with torch.no_grad():
            D, U = torch.linalg.eigh(K)
            D = D[-dim:].clamp(min=0)
            K[:] = U[:, -dim:].mm(D.diag()).mm(U[:, -dim:].transpose(0, 1))
        return objective(K, *args, **kwargs)

    result = torch_minimize(method, kernel_objective, kernel_init, **kwargs)
    U, s, _ = np.linalg.svd(result.x)

    result.x = U[:, :dim] @ np.sqrt(np.diag(s[:dim]))
    return result


def torch_minimize(method,
                   objective: Callable, init: np.ndarray, data: Sequence, args: Sequence = [],
                   seed: Optional[int] = None, device: str = 'auto', max_iter: int = 100,
                   batch_size=50_000, shuffle=True, **kwargs) -> 'scipy.optimize.OptimizeResult':
    """ Pytorch minimization routine using L-BFGS.

        This function is aims to be a pytorch version of :func:`scipy.optimize.minimize`.

        Args:
            objective:
                Loss function to minimize.
                Function argument is the current parameters and optional additional argument.
            init:
                The initial parameter values.
            data:
                Sequence of data arrays.
            args:
                Sequence of additional arguments, passed to the objective.
            seed:
                Manual seed of randomness in data sampling. Use to preserve reproducability.
            device:
                Device to run the minimization on, usually "cpu" or "cuda".
                "auto" uses "cuda", if available.
            max_iter:
                The maximum number of optimizer iteration (also called epochs).
        Returns:
            Dict-like object containing status and result information
            of the optimization.
    """
    import torch

    device = torch_device(device)

    data = [torch.tensor(d).to(device) for d in data]
    args = [torch.tensor(a).to(device) for a in args]
    factr = 1e7 * np.finfo(float).eps
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)
    dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*data),
                                             batch_size=batch_size, shuffle=shuffle, generator=g)

    method = method.lower()
    optimizers = {
        'l-bfgs-b': torch.optim.LBFGS,
        'adam': torch.optim.Adam,
    }
    X = torch.tensor(init, requires_grad=True, device=device)
    if method.lower() in optimizers:
        optimizer = optimizers[method.lower()]([X], **kwargs)
    else:
        raise ValueError(f"Expects method in {optimizers.keys()}, got {method}.")

    loss = float("inf")
    success, message = True, ""
    for n_iter in range(max_iter):
        prev_loss = loss
        loss = 0
        for batch in dataloader:
            def closure():
                optimizer.zero_grad()
                loss = objective(X, *batch, *args)
                loss.backward()
                return loss

            step_loss = optimizer.step(closure)
            loss += len(batch) * step_loss / len(dataloader.dataset)

        if abs(prev_loss - loss) / max(abs(loss), abs(prev_loss), 1) < factr:
            break
    else:
        success = False
        message = f"{method} did not converge."

    return scipy.optimize.OptimizeResult(
        x=X.cpu().detach().numpy(), fun=loss.cpu().detach().numpy(),
        nit=n_iter + 1, success=success, message=message)


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
