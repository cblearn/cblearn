.. _nature_vogue_dataset:

Nature and Vogue datasets
---------------------------

The nature and vogue datasets consist of odd-one-out triplets of the form
"Out of three shown items pick one that appears to be
different from the two others".

The items were either images of natural scenes (forests, beaches, mountaints, etc.)
or covers of the Vogue magazine.

**Data Set Characteristics:**

    ===================   =====================
    Triplets (Covers)                      1107
    Objects  (Covers)                        60
    Triplets (Scenes)                      3355
    Objects  (Scenes)                       120
    ===================   =====================

This datasets can be downloaded using :func:`cblearn.datasets.fetch_nature_scene_similarity` and
:func:`cblearn.datasets.fetch_vogue_cover_similarity`
To use the odd-one-out triplets with e.g. ordinal embedding algorithms, convert them to standard triplets
with :func:`cblearn.dataset.triplets_from_oddoneout`.

Please cite the following paper if you use this dataset in publications.

.. topic:: References

    - Heikinheimo, H., & Ukkonen, A. (2013). The crowd-median algorithm.
      In Proceedings of the AAAI Conference on Human Computation and Crowdsourcing (Vol. 1, No. 1).
