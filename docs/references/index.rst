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

    datasets.fetch_car_similarity
    datasets.fetch_food_similarity
    datasets.fetch_imagenet_similarity
    datasets.fetch_nature_scene_similarity
    datasets.fetch_material_similarity
    datasets.fetch_musician_similarity
    datasets.fetch_vogue_cover_similarity
    datasets.fetch_things_similarity
    datasets.fetch_similarity_matrix


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
    datasets.triplet_response
    datasets.noisy_triplet_response


:mod:`cblearn.embedding` Embedding
==================================

.. automodule:: cblearn.embedding

.. currentmodule:: cblearn

.. autosummary::
    :toctree: generated/

    embedding.CKL
    embedding.FORTE
    embedding.GNMDS
    embedding.SOE
    embedding.STE
    embedding.TSTE
    embedding.OENN
    embedding.MLDS

Utility
-------

.. currentmodule:: cblearn.embedding

.. autosummary::
    :toctree: generated/

    estimate_dimensionality_cv
    DimensionEstimationResult


.. _references_embedding_wrapper:
Wrapper
-------

.. currentmodule:: cblearn.embedding


.. autosummary::
    :toctree: generated/

    wrapper.MLDS
    wrapper.SOE

:mod:`cblearn.cluster` Cluster
==========================================

.. automodule:: cblearn.cluster

.. currentmodule:: cblearn

.. autosummary::
    :toctree: generated/

    cluster.ComparisonHC



:mod:`cblearn.metrics` Metrics
==============================

.. automodule:: cblearn.metrics

.. currentmodule:: cblearn

.. autosummary::
    :toctree: generated/

    metrics.query_accuracy
    metrics.query_error
    metrics.procrustes_distance
    metrics.QueryScorer

:mod:`cblearn.preprocessing` Preprocessing
==========================================

.. automodule:: cblearn.preprocessing

.. currentmodule:: cblearn

.. autosummary::
    :toctree: generated/

    preprocessing.query_from_columns
    preprocessing.triplets_from_multiselect
    preprocessing.triplets_from_oddoneout
    preprocessing.triplets_from_mostcentral
    preprocessing.SharedColumnEncoder
    preprocessing.MultiColumnLabelEncoder

:mod:`cblearn.utils` Utility
============================

.. automodule:: cblearn.utils

.. currentmodule:: cblearn

.. autosummary::
    :toctree: generated/

    utils.data_format
    utils.check_format
    utils.check_query
    utils.check_query_response
    utils.check_response
    utils.check_size
