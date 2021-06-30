.. _musician_similarity_dataset:

Musician Similarity dataset
---------------------------

`This dataset contains triplets`_ gathered during the `MusicSeer similarity survey`_ in October 2002.

In a web-based survey or game, the user was presented a target musician and multiple others to select the
most similar to the target.
Such, for each user judgement multiple triplets were created with the remaining others.

.. _This dataset contains triplets: https://labrosa.ee.columbia.edu/projects/musicsim/musicseer.org/results/
.. _MusicSeer similarity survey: http://musicseer.com

**Data Set Characteristics:**

    ===================   =====================
    Triplets                             224792
    Objects (Musicians)                     413
    Dimensionality                      unknown
    ===================   =====================

.. Note:
    The original dataset, published 2002-10-15, contains 224793 triplets.
    We omit in this dataset the triplets with missing values for the last triplet index.

This dataset can be downloaded using the :func:`cblearn.datasets.fetch_musician_similarity`.

When using these triplets, please give credit to the original authors.

.. topic:: References

    - Ellis, D. P., Whitman, B., Berenzweig, A., & Lawrence, S. (2002).
      The quest for ground truth in musical artist similarity.
