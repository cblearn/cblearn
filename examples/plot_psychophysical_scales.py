# -*- coding: utf-8 -*-
r"""
Psychophysical scaling with ordinal embedding
=============================================

In this example, we how to estimate psychophysical scales with ordinal embedding algorithms, using triplets (triad) data.
"""

# %%
from cblearn import datasets
from cblearn import embedding
from cblearn import utils

# %%
# Loading triplets a triplet dataset
# ----------------------------------
#
# cblearn provides functions to download and parse publicly available psychophysical datasets.
# Here we will load triplets of rendered materials (Lagunas et al., 2019, ACM Transactions on Graphics):
materials = datasets.fetch_material_similarity()
print(materials.keys())
print(materials.material_name[:10])
triplets = materials.triplet
print(triplets[:10], triplets.shape)


# %% [markdown]
# A triplet (i, j, k) is the fundamental datatype
# and describes the distances in a psychophysical scale :math:`\psi`, here called embedding, by
#
# :math:`d(\psi(i), \psi(j)) \le d(\psi(i), \psi(k)).`

# %%
# The triplet entries i, j, k are indices of the shown material stimuli.
print(materials.material_name[triplets[:10]])

# %%
# Estimating psychophysical scales from triplets
# ----------------------------------------------
# The psychophysical scale can be constructed from triplets using one of the so called ordinal embedding algorithms,
# implemented in cblearn:

oe = embedding.SOE(n_components=2, n_init=1)  # increasing n_init provides better results but requires more computation time.
X = oe.fit_transform(triplets)
print(oe.n_iter_)

# %%
# Let's plot this scale:
import matplotlib.pyplot as plt

plt.plot(X[:, 0], X[:, 1], 'o')
plt.show()

# %%
# Simulating triplets and measuring the fit
# -----------------------------------------
# Now, we will show some advanced usage where we simulate (artificial) triplets
# and analyse the fit of the scale with cross-validation.

import numpy as np
from sklearn.model_selection import cross_val_score
from scipy.spatial import procrustes

##############################
# you can play with these parameters
n_objects = 30
dim = 2
dim_embedding = 2
noise = 0.5
triplet_factor = 8
##############################

# %%
# Simulating responses
# ####################
# Sometimes it is useful to simulate triplets from an assumed scale
# where we freely can define the number of stimuli and their perceived dimensionality.
#
# First, we generate this ground-truth scale and then sample triplets with a normal distributed noise model.
# The scale's fit requires a certain number of triplets, that depends on the ground truth dimensionality and number of samples.

X_true = np.random.normal(0, 1, (n_objects, dim))

n_triplets = int(triplet_factor * dim * n_objects * np.log(n_objects))
print(f"# Triplets = {n_triplets}")
triplets = datasets.make_random_triplets(X_true, size=n_triplets, noise='normal', noise_options={'scale': noise}, result_format='list-order')
triplets_test = datasets.make_random_triplets(X_true, size=10000, noise='normal', noise_options={'scale': noise}, result_format='list-order')

# %%
# Analysing the fit: Triplet accuracy and procrustes distance
# ###########################################################
# The fit of a scale can be evaluated with different metrics:
#
# 1. The triplet accuracy is the fraction of triplets, agreeing with the distances in the scale estimate.
# 2. The procrustes distance is the squared error between the ground truth and estimated scales, which are aligned
#    in terms of rotation, translation, flip and scale, that cannot be reconstructed from triplets.
#
# The accuracy on the triplets used for fitting (train accuracy) describes how well the scale memorizes
# these triplets. In contrast, accuracy on separate test triplets describes how well the scale generalizes to unseen
# triplets. This test accuracy can be approximated without additional triplets by using cross-validation.

soe = embedding.SOE(dim_embedding)
X_est = soe.fit_transform(triplets)
print(f"""
Train Accuracy = {soe.score(triplets):.2f}
Test  Accuracy = {soe.score(triplets_test):.2f}
CV    Accuracy = {cross_val_score(soe, triplets, cv=10).mean():.2f}""")

if X_est.shape[1] == 2:
    # align scale and rotation
    X_true, X_est, disparity = procrustes(X_true, X_est)
    print(f"Mean Square Difference = {disparity / n_objects:.5f}")
    
    plt.plot(X_est[:, 0], X_est[:, 1], 'ob', label='Truth', ms=10)
    plt.plot(X_true[:, 0], X_true[:, 1], 'or', label='SOE', ms=10)
    text_args = dict(fontsize=9, horizontalalignment='center', verticalalignment='center')
    for i in range(n_objects):
        plt.text(X_est[i, 0], X_est[i, 1], str(i), **text_args)
        plt.text(X_true[i, 0], X_true[i, 1], str(i), **text_args)
    plt.legend();
    plt.show()
