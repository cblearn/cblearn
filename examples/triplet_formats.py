# -*- coding: utf-8 -*-
r"""
Triplet Formats
===============

cblearn supports triplet input data in two
formats: As a triplet array (or matrix with three columns) or as a sparse matrix.
"""
import time

from cblearn import datasets
from cblearn.utils import check_query_response

triplets_ordered = datasets.make_random_triplet_indices(n_objects=1000, size=1000000, repeat=False)
print(f"'triplets_ordered' is a numpy array of shape {triplets_ordered.shape}.")

# %%
# Triplet Array
# -------------
# In the array format, the constraints are encoded by the index order.
triplet = triplets_ordered[0]
print(f"The triplet {triplet} means, that object {triplet[0]} (1st) should be "
      f"embedded closer to object {triplet[1]} (2nd) than to object {triplet[0]} (3th).")

# %%
# Alternatively, the triplet array can be complemented by a answer array.
triplets_boolean, answers_boolean = check_query_response(triplets_ordered, result_format='list-boolean')
print(f"Is object {triplets_boolean[0, 0]} closer to object {triplets_boolean[0, 1]} "
      f"than to object {triplets_boolean[0, 2]}? {answers_boolean[0]}.")

triplets_numeric, answers_numeric = check_query_response(triplets_ordered, result_format='list-count')
print(f"Is object {triplets_numeric[0, 0]} closer to object {triplets_numeric[0, 1]} "
      f"than to object {triplets_numeric[0, 2]}? {answers_numeric[0]}.")


# %%
# Sparse Matrix
# -------------
# In the sparse matrix format the object indices of the triplet constraints correspond to the
# row / column indices of a sparse matrix.
triplet_spmatrix = check_query_response(triplets_ordered, result_format='tensor-count')
print(f"triplet_spmatrix[i, j, k]="
      f"{triplet_spmatrix[triplets_numeric[0, 0], triplets_numeric[0, 1], triplets_numeric[0, 2]]} "
      f"is the same as answer(i,j,k)={answers_numeric[0]}.")

# %%
# Conversation Time
# -----------------
# Converting between triplet and answer formats is not free, let's measure the process time.


def time_convert_triplet(triplets, to_format):
    time_start = time.process_time()
    if len(triplets) == 2:
        triplets, answers = triplets
        check_query_response(triplets, answers, result_format=to_format)
    else:
        check_query_response(triplets, result_format=to_format)
    return (time.process_time() - time_start)


data = [triplets_ordered, (triplets_boolean, answers_boolean),
        (triplets_numeric, answers_numeric), triplet_spmatrix]
formats = ["list-order", "list-boolean", "list-count", "tensor-count"]

timings = [
    (time_convert_triplet(triplets, to_format),
     f"{from_format}->{to_format}")
    for from_format, triplets in zip(formats, data)
    for to_format in formats
]

for seconds, desc in sorted(timings):
    print(f"{seconds * 1000:.2f}ms {desc}")
