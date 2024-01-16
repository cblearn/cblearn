# -*- coding: utf-8 -*-
r"""
.. _example_ordinal_embedding:
Ordinal Embedding
=================

Ordinal embedding algorithms are used to estimate d-dimensional Euclidean coordinates from comparison data.

In this example, we first generate some triplet comparisons from Euclidean points.
If you want to use triplets from a human experiment, these ground-truth points are actually unknown.

We try to reconstruct the points from the comparisons and quantify. As we know the ground-truth points here,
we can evaluate how good this works.
"""
from sklearn.datasets import make_blobs
from cblearn.datasets import make_random_triplets

# sample points from 3-dimensional Gaussian
true_embedding, __ = make_blobs(n_samples=100, n_features=3, centers=1)
print(f"Embedding: {true_embedding.shape}")

# sample triplet comparisons from
triplets = make_random_triplets(true_embedding, size=10000, result_format='list-order')
print(f"Triplet comparisons: {triplets.shape}")

# %%
# The ordinal embedding estimators in cblearn follow the interface of scikit-learn's transformers.
# Let's estimate coordinates in a 2-dimensional and in a 3-dimensional Euclidean space.
from cblearn.embedding import SOE  # noqa: E402 linter ignore import not at top of file


transformer_2d = SOE(n_components=2)
pred_embedding_2d = transformer_2d.fit_transform(triplets)

transformer_3d = SOE(n_components=3)
pred_embedding_3d = transformer_3d.fit_transform(triplets)

print(f"Predicted 2D embedding: {pred_embedding_2d.shape}")
print(f"Predicted 3D embedding: {pred_embedding_3d.shape}")


# %%
# The estimated embedding can be evaluated from different perspectives.
#
#   1. The procrustes distance is a square distance between the true and the
#      estimated embeddings, where scale, rotation and translation transformations
#      are ignored. This is only possible, if the true embedding is known
#      and the embeddings have the same dimensionality.
#   2. The training triplet error is the fraction of training comparisons,
#      which do not comply with the estimated embedding.
#   3. The cross-validation triplet error indicates the fraction of unknown triplets
#      which do not comply with the estimated embedding.
#      Note, that 5-fold cross validation requires refitting the model 5 times.
from sklearn.model_selection import cross_val_score  # noqa: E402 linter ignore import not at top of file
from cblearn.metrics import procrustes_distance  # noqa: E402


distance_3d = procrustes_distance(true_embedding, pred_embedding_3d)
print(f"Procrustes distance: {distance_3d:.5f} in 3d")

error_2d = 1 - transformer_2d.score(triplets)
error_3d = 1 - transformer_3d.score(triplets)
print(f"Training triplet error: {error_2d:.3f} in 2d vs {error_3d:.3f} in 3d.")

cv_error_2d = 1 - cross_val_score(transformer_3d, triplets, cv=5, n_jobs=-1).mean()
cv_error_3d = 1 - cross_val_score(transformer_3d, triplets, cv=5, n_jobs=-1).mean()
print(f"CV triplet error: {cv_error_2d:.3f} in 2d vs {cv_error_3d:.3f} in 3d.")
