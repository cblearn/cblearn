from typing import Union

import scipy
import sparse
import numpy as np


Query = Union[np.ndarray, sparse.COO, scipy.sparse.spmatrix]
Response = Union[Query, np.ndarray]
