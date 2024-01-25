==========
User Guide
==========

Most Machine Learning algorithms use numerical training data (features) for inference,
either representing points in a Euclidean space, similarities, or distances.
The are settings, e.g. in human studies, when metric points are not available but only ordinal comparisons.
Comparison-based learning algorithms are the machine learning algorithms applicable in this setting.

-------------------
Triplet comparisons
-------------------

Triplet comparisons are the most common form of ordinal comparisons. For the triplet of objects :math:`(i, j, k)`
one can ask, "Is the object i more similar to the object j or k?".
For the unknown points :math:`(x_i, x_j, x_k)` and the distance metric :math:`\delta`, the question corresponds to the following
inequality:

.. math::

    \delta(x_i, x_j) \le \delta(x_i, x_k).

This library supports two representation formats of triplets in an array or a sparse matrix form.
The array form uses 2d ``numpy`` arrays representing a triplet per row and columns for ``i,j,k``.
Alternatively to the ordering, an additional response array containing 1 or -1 can specify if  ``(i,j,k)`` is correct or wrong.
The sparse matrix is an alternative representation, where triplets are naturally specified as the matrix indices, containing entries 1 or -1.

--------------------------
Scikit-learn compatibility
--------------------------

All estimators in this library are compatible with the ``scikit-learn`` API and can be used in ``scikit-learn`` pipelines
if comparisons are represented in the array format.
The ``scikit-learn`` compatibility is achieved by implementing the ``fit``, ``predict``, and ``score`` methods of the ``BaseEstimator`` class.

The ``fit`` method is used to train the model, the ``predict`` method is used to predict the labels of the test data,
and the ``score`` method is used to evaluate the model on the test data.
In the case of ordinal embedding, for example, the ``predict`` method returns the triplet response according to the embedding
and the ``score`` method returns the triplet accuracy (the fraction of correct triplet responses).

The :ref:`example_ordinal_embedding` example shows how to use a scikit-learn cross validation function with an ordinal embedding estimator.

-------------------------
Pytorch backend (CPU/GPU)
-------------------------

The default backend for computations is the ``scipy`` stack, optimized for fast CPU computations and minimal overhead in both compute and disk space.
However, this comes with limitations when implementing new methods and for calculations with very large data sets.

As an alternative for some estimators, a ``pytorch`` implementation exists.
 To use this implementation, ``pytorch`` must be installed (see :ref:`extras_install`) and, if necessary,
 the option ``backend='torch'`` must be set (see the respective function documentation).
These estimators take care automatically of the data transfer between numpy and torch (internal data representation) and
use a batched optimizer for faster convergence. If a CUDA GPU is available, the computations are automatically performed on the GPU.

``pytorch`` itself needs a lot of hard disk space and starting the optimization has a certain overhead
(automatic derivation, data transformation).
 It is therefore advisable to use the ``scipy`` backend by default and only change if necessary.



-------------------------
Dataset loading utilities
-------------------------

.. include:: ../../cblearn/datasets/descr/musician_similarity.rst
.. include:: ../../cblearn/datasets/descr/food_similarity.rst
.. include:: ../../cblearn/datasets/descr/car_similarity.rst
.. include:: ../../cblearn/datasets/descr/imagenet_similarity.rst
.. include:: ../../cblearn/datasets/descr/things_similarity.rst
.. include:: ../../cblearn/datasets/descr/nature_vogue_similarity.rst
.. include:: ../../cblearn/datasets/descr/material_similarity.rst
.. include:: ../../cblearn/datasets/descr/similarity_matrix.rst
