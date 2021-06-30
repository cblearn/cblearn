.. _things_similarity_dataset:

Things Similarity dataset
-------------------------

`This dataset`_ contains odd-one-out trials of images from the Things image database.
In an crowd sourced experiments, subjects were asked to choose one of three images, that is the odd-one.
Note: The trials used here, are the test trials of the original paper. Their train trials are not published.

The data is shared under CC-BY-4.0 by Hebart, M. N., Zheng, C. Y., Pereira, F., and Baker, C. I.

.. _This dataset: https://osf.io/z2784/


**Data Set Characteristics:**

    ===================   =====================
    Trials                              146,012
    Objects (Things)                      1,854
    Query                 3 images, odd one out
    ===================   =====================

This dataset can be downloaded using the :func:`cblearn.datasets.fetch_things_similarity`.
To use the odd-one-out trials with e.g. ordinal embedding algorithms, they can be converted to standard triplets
with :func:`cblearn.preprocessing.triplets_from_oddoneout`.

Please cite the following paper if you use this dataset in publications.

.. topic:: References

    - Hebart, M. N., Zheng, C. Y., Pereira, F., & Baker, C. I. (2020).
      Revealing the multidimensional mental representations of natural objects underlying human similarity judgements.
      Nature Human Behaviour, 4(11), 1173–1185. https://doi.org/10.1038/s41562-020-00951-3
