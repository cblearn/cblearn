.. _api_ref:

=============
API Reference
=============

This is the class and function reference of OrdComp.


:mod:`ordcomp.datasets` Datasets
================================

.. automodule:: ordcomp.datasets

Loaders
-------

.. currentmodule:: ordcomp

.. autosummary::
    :toctree: generated/

    datasets.fetch_musician_similarity
    datasets.fetch_food_similarity


Simulations
-----------

.. currentmodule:: ordcomp

.. autosummary::
    :toctree: generated/

    datasets.make_random_triplets



Low-level Dataset Utility
-------------------------

.. currentmodule:: ordcomp

.. autosummary::
    :toctree: generated/

    datasets.make_all_triplet_indices
    datasets.make_random_triplet_indices
    datasets.triplet_answers
    datasets.noisy_triplet_answers


:mod:`ordcomp.embedding` Embedding
==================================

.. automodule:: ordcomp.embedding

.. currentmodule:: ordcomp

.. autosummary::
    :toctree: generated/

Wrapper
-------

.. currentmodule:: ordcomp.embedding


.. autosummary::
    :toctree: generated/

    wrapper.MLDS
    wrapper.SOE


:mod:`ordcomp.metrics` Metrics
==============================

.. automodule:: ordcomp.metrics

.. currentmodule:: ordcomp

.. autosummary::
    :toctree: generated/

    metrics.triplet_error
    metrics.procrustes_distance
    metrics.TripletScorer

:mod:`ordcomp.utils` Utility
============================

.. automodule:: ordcomp.utils

.. currentmodule:: ordcomp

.. autosummary::
    :toctree: generated/

    utils.data_format
    utils.check_format
    utils.check_triplet_questions
    utils.check_triplet_answers
    utils.check_size
