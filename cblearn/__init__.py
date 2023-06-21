from cblearn import _version
from cblearn.core import Comparison, SparseComparison
from cblearn.core import issparse, asdense, assparse, canonical_X_y
from cblearn.core import check_triplets, check_quadruplets
from cblearn.core import check_pivot_comparisons, check_pairwise_comparisons
from cblearn.core import uniform_index_tuples, all_index_tuples

__version__ = _version.get_versions()['version']
