.. _api_ref:

=============
API Reference
=============

This is the class and function reference of cblearn.


:mod:`cblearn.datasets` Datasets
================================

.. automodule:: cblearn.datasets

Loaders
-------

.. currentmodule:: cblearn

.. autosummary::
    :toctree: generated/

    datasets.fetch_musician_similarity
    datasets.fetch_food_similarity


Simulations
-----------

.. currentmodule:: cblearn

.. autosummary::
    :toctree: generated/

    datasets.make_random_triplets



Low-level Dataset Utility
-------------------------

.. currentmodule:: cblearn

.. autosummary::
    :toctree: generated/

    datasets.make_all_triplet_indices
    datasets.make_random_triplet_indices
    datasets.triplet_answers
    datasets.noisy_triplet_answers


:mod:`cblearn.embedding` Embedding
==================================

.. automodule:: cblearn.embedding

.. currentmodule:: cblearn

.. autosummary::
    :toctree: generated/

    embedding.SOE

Wrapper
-------

.. currentmodule:: cblearn.embedding


.. autosummary::
    :toctree: generated/

    wrapper.MLDS
    wrapper.SOE


:mod:`cblearn.metrics` Metrics
==============================

.. automodule:: cblearn.metrics

.. currentmodule:: cblearn

.. autosummary::
    :toctree: generated/

    metrics.triplet_error
    metrics.procrustes_distance
    metrics.TripletScorer

:mod:`cblearn.utils` Utility
============================

.. automodule:: cblearn.utils

.. currentmodule:: cblearn

.. autosummary::
    :toctree: generated/

    utils.data_format
    utils.check_format
    utils.check_triplet_questions
    utils.check_triplet_answers
    utils.check_size

Pytorch
-------

.. autosummary::
    :toctree: generated/

    utils.assert_torch_is_available
    utils.torch_minimize_lbfgs
