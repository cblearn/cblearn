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
