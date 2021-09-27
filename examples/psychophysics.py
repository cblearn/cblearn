# -*- coding: utf-8 -*-
r"""
Psychophysics with ordinal embedding
====================================


"""
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from cblearn import datasets
from cblearn import embedding
from cblearn import utils

# %%
materials = datasets.fetch_material_similarity()
materials.keys()

# %%
materials.material_name

# %%
triplets = materials.triplet
triplets[:10], triplets.shape

# %% [markdown]
# Triplet (i, j, k) means:
#
# $$d(\psi(i), \psi(j)) \le d(\psi(i), \psi(k))$$

# %%
# triplets indices to artists:
materials.material_name[triplets]

# %%
# alternative representation
utils.check_query_response(triplets, result_format='tensor-count')

# %%
oe = embedding.TSTE(n_components=2)
X = oe.fit_transform(triplets)
oe.n_iter_

# %%
import matplotlib.pyplot as plt


plt.plot(X[:, 0], X[:, 1], 'o')

# %%
X = embedding.SOE(n_components=2).fit_transform(triplets)
plt.plot(X[:, 0], X[:, 1], 'o')

# %%
import numpy as np
from sklearn.model_selection import cross_val_score
from scipy.spatial import procrustes

##############################
# play with these parameters
n_objects = 50
dim = 2
dim_embedding = 2
noise = 0.5
triplet_factor = 10
##############################


X_true = np.random.normal(0, 1, (n_objects, dim))

n_triplets = int(triplet_factor * n_dim * n_objects * np.log(n_objects))
print(f"# Triplets = {n_triplets}")
triplets = datasets.make_random_triplets(X_true, size=n_triplets, noise='normal', noise_options={'scale': noise}, result_format='list-order')
triplets_test = datasets.make_random_triplets(X_true, size=10000, noise='normal', noise_options={'scale': noise}, result_format='list-order')

soe = embedding.SOE(dim_embedding)
X_est = soe.fit_transform(triplets)
print(f"""\
Train Accuracy = {soe.score(triplets):.2f}
Test  Accuracy = {soe.score(triplets_test):.2f}
CV    Accuracy = {cross_val_score(soe, triplets, cv=10).mean():.2f}""")

if X_true.shape == X_est.shape:
    # align scale and rotation
    X_true, X_est, disparity = procrustes(X_true, X_est)
    print(f"Mean Square Difference = {disparity / n_objects:.5f}")
    
    plt.plot(X_est[:, 0], X_est[:, 1], 'ob', label='Truth', ms=10)
    plt.plot(X_true[:, 0], X_true[:, 1], 'or', label='SOE', ms=10)
    text_args = dict(fontsize=9, horizontalalignment='center', verticalalignment='center')
    for i in range(n_objects):
        plt.text(*X_est[i] , str(i), **text_args)
        plt.text(*X_true[i] , str(i), **text_args)
    plt.legend();
