==========
User Guide
==========

Most Machine Learning algorithms use numerical training data (features) for inference,
either representing points in a Euclidean space, similarities, or distances.
The are settings, e.g. in human studies, when metric points are not available but only ordinal comparisons.
Comparison-based Learning are Machine Learning algorithms, applicable in this setting.

-------------------
Triplet comparisons
-------------------

Triplet comparisons are the most common form of ordinal comparisons. For the triplet of objects :math:`(i, j, k)`
one can ask "Is the object i more similar to the object j or to the object k?".
For the unknown points :math:`(x_i, x_j, x_k)` and the distance metric :math:`\delta`, the question corresponds to the following
inequality:

.. math::

    \delta(x_i, x_j) \le \delta(x_i, x_k).

This library supports two representation formats of triplets, in an array or in an sparse matrix.


-------------------------
Dataset loading utilities
-------------------------

.. include:: ../../cblearn/datasets/descr/musician_similarity.rst
.. include:: ../../cblearn/datasets/descr/food_similarity.rst
