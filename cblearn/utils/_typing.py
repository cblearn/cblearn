from typing import Union, Tuple

import scipy
import sparse
import numpy as np


Questions = Union[np.ndarray, sparse.COO, scipy.sparse.spmatrix]
Answers = Union[Questions, Tuple[Questions, np.ndarray]]
