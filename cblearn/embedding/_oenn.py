from typing import Optional, Union

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state

from cblearn import utils
from cblearn.embedding._base import TripletEmbeddingMixin
from cblearn.embedding._torch_utils import assert_torch_is_available, torch_device


class OENN(BaseEstimator, TripletEmbeddingMixin):
    """ Ordinal Embedding Neural Network (OENN).

        OENN [1]_ learns a dense neural network, that maps from the
        object indices to an embedding, that satisfies the triplet constraints.

        OENN requires the optional *torch* dependency
        and uses the ADAM optimizer and backpropagation.
        It can executed on CPU, but also CUDA GPUs.

        .. note::
            The *torch* backend requires the *pytorch* python package (see :ref:`extras_install`).

        Attributes:
            embedding_: Final embedding, shape (n_objects, n_components)
            encoding_: Input encoding of the objects, shape (n_objects, n_input_dim)
            stress_: Final value of the loss corresponding to the embedding.
            stress_progess_: Loss per optimization iteration over the full triplet dataset.
            n_iter_: Final number of optimization steps.

        Examples:

        >>> from cblearn import datasets
        >>> seed = np.random.RandomState(42)
        >>> true_embedding = seed.rand(15, 2)
        >>> triplets = datasets.make_random_triplets(true_embedding, result_format='list-order',
        ...                                          size=1000, random_state=seed)
        >>> triplets.shape, np.unique(triplets).shape
        ((1000, 3), (15,))
        >>> estimator = OENN(n_components=2, random_state=seed)
        >>> embedding = estimator.fit_transform(triplets, n_objects=15)
        >>> embedding.shape
        (15, 2)
        >>> estimator.score(triplets) > 0.8
        True

        References
        ----------
        .. [1] Vankadara, L. C., Haghiri, S., Lohaus, M., Wahab, F. U., & von Luxburg, U. (2020).
               Insights into Ordinal Embedding Algorithms: A Systematic Evaluation. ArXiv:1912.01666 [Cs, Stat].
        """

    def __init__(self, n_components=2, verbose=False, random_state: Union[None, int, np.random.RandomState] = None,
                 max_iter=1000, learning_rate=0.005, layer_width='auto', batch_size=50_000,  device: str = "auto"):
        r""" Initialize the estimator.

        Args:
            n_components :
                The dimension of the embedding.
            verbose: boolean, default=False
                Enable verbose output.
            random_state:
             The seed of the pseudo random number generator used to initialize the optimization.
            max_iter:
                Maximum number of optimization iterations.
            learning_rate: Learning rate of the gradient-based optimizer.
            layer_width: Width of the hidden layers. If 'auto', then the width w depends on the
                         input dimension d and the number of objects n: :math:`w = \max(120, 2d\log n)`.
            batch_size: Batch size of stochastic optimization.
            device: The device on which pytorch computes. {"auto", "cpu", "cuda"}
                "auto" chooses cuda (GPU) if available, but falls back on cpu if not.
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.verbose = verbose
        self.random_state = random_state
        self.layer_width = layer_width
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device

    def _make_emb_net(self, input_dim: int, layer_width: int, hidden_layers: int = 3):
        import torch

        return torch.nn.Sequential(
            torch.nn.Linear(input_dim, layer_width),
            torch.nn.ReLU(),
            *sum([(torch.nn.Linear(layer_width, layer_width), torch.nn.ReLU())
                  for _ in range(hidden_layers - 1)], tuple()),
            torch.nn.Linear(layer_width, self.n_components)
        ).double()

    def fit(self, X: utils.Query, y: np.ndarray = None,
            encoding: Optional[np.ndarray] = None, n_objects: Optional[int] = None, eps: float = 1e-6) -> 'OENN':
        """Computes the embedding.

        Args:
            X: The training input samples, shape (n_samples, 3)
            y: Ignored
            encoding: Encoding of the objects. If none, we use a random input encoding.
            n_objects: Number of objects in the embedding.
            eps: Largest loss difference to stop optimization.
        Returns:
            self.
        """
        triplets = utils.check_query_response(X, y, result_format='list-order')
        random_state = check_random_state(self.random_state)
        if n_objects is None:
            n_objects = triplets.max() + 1
        if encoding is None:
            input_dim = int(np.ceil(np.log(n_objects))) + 1
            self.encoding_ = np.random.normal(0, 1, (n_objects, input_dim))
        else:
            self.encoding_ = encoding

        layer_width = self.layer_width
        if layer_width == 'auto':
            layer_width = max(120, 2 * self.n_components * np.log(n_objects))

        assert_torch_is_available()
        import torch  # torch is an optional dependency - import at runtime

        seed = random_state.randint(1)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        device = torch_device(self.device)
        emb_net = self._make_emb_net(self.encoding_.shape[1], layer_width).to(device)
        optimizer = torch.optim.Adam(emb_net.parameters(), lr=self.learning_rate)
        criterion = torch.nn.TripletMarginLoss(margin=1, p=2).to(device)
        triplets = torch.tensor(triplets.astype(int), device=device)
        encoding = torch.tensor(self.encoding_, dtype=float, device=device)
        dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(triplets),
                                                 batch_size=self.batch_size, shuffle=True)

        loss = float("inf")
        losses = []
        for n_iter in range(self.max_iter):
            prev_loss = loss
            loss = 0
            for batch in dataloader:
                def closure():
                    optimizer.zero_grad()
                    X = encoding[triplets.long()]
                    loss = criterion(emb_net(X[:, 0, :]), emb_net(X[:, 1, :]), emb_net(X[:, 2, :]))
                    loss.backward()
                    return loss

                step_loss = optimizer.step(closure)
                loss += len(batch) * step_loss / len(dataloader.dataset)

            losses.append(loss.cpu().detach().numpy())
            if abs(prev_loss - loss) / max(abs(loss), abs(prev_loss), 1) < eps:
                break
        else:
            success = False
            message = "Adam optimizer did not converge."

        if self.verbose and not success:
            print(f"OENN's optimization failed with reason: {message}.")

        self.embedding_ = emb_net(encoding).cpu().detach().numpy()
        self.stress_ = losses[-1]
        self.stress_progress_ = losses
        self.n_iter_ = n_iter + 1

        return self