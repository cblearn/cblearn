# -*- coding: utf-8 -*-
r"""
Triplet Formats
===============

OrdComp supports triplet input data in two
formats: As a triplet array (or matrix with three columns) or as a sparse matrix.
"""
import time

from ordcomp import datasets
from ordcomp.utils import check_triplets

triplets_implicit = datasets.make_random_triplets(n_objects=1000, size=1000000, repeat=False)
print(f"'triplets_implicit' is a numpy array of shape {triplets_implicit.shape}.")

# %%
# Triplet Array
# -------------
# In the array format, the constraints are encoded by the index order.
triplet = triplets_implicit[0]
print(f"The triplet {triplet} means, that object {triplet[0]} (1st) should be "
      f"embedded closer to object {triplet[1]} (2nd) than to object {triplet[0]} (3th).")

# %%
# Alternatively, the triplet array can be complemented by a response array.
triplets_boolean, responses_boolean = check_triplets(triplets_implicit, format='array', response_type='boolean')
print(f"Is object {triplets_boolean[0, 0]} closer to object {triplets_boolean[0, 1]} "
      f"than to object {triplets_boolean[0, 2]}? {responses_boolean[0]}.")

triplets_numeric, responses_numeric = check_triplets(triplets_implicit, format='array', response_type='numeric')
print(f"Is object {triplets_numeric[0, 0]} closer to object {triplets_numeric[0, 1]} "
      f"than to object {triplets_numeric[0, 2]}? {responses_numeric[0]}.")


# %%
# Sparse Matrix
# -------------
# In the sparse matrix format the object indices of the triplet constraints correspond to the
# row / column indices of a sparse matrix.
triplet_spmatrix = check_triplets(triplets_implicit, format='spmatrix')
print(f"triplet_spmatrix[i, j, k]="
      f"{triplet_spmatrix[triplets_numeric[0, 0], triplets_numeric[0, 1], triplets_numeric[0, 2]]} "
      f"is the same as response(i,j,k)={responses_numeric[0]}.")

# %%
# Conversation Time
# -----------------
# Converting between triplet and response formats is not free, let's measure the process time.


def time_convert_triplet(triplets, responses, to_format, to_response_type):
    time_start = time.process_time()
    check_triplets(triplets, responses=responses, format=to_format, response_type=to_response_type)
    return (time.process_time() - time_start)


data = [(triplets_implicit, None), (triplets_boolean, responses_boolean),
        (triplets_numeric, responses_numeric), (triplet_spmatrix, None)]
formats = [("array", "implicit"), ("array", "boolean"), ("array", "numeric"), ("spmatrix", None)]

timings = [
    (time_convert_triplet(triplets, responses, to_format, to_response),
     f"{from_format}[{from_response}]->{to_format}[{to_response}]")
    for (from_format, from_response), (triplets, responses) in zip(formats, data)
    for to_format, to_response in formats
]

for seconds, desc in sorted(timings):
    print(f"{seconds * 1000:.2f}ms {desc}")
