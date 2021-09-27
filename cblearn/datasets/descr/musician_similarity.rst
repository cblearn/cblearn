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
    Triplets                            131.970
    Objects (Artists)                       448
    Dimensionality                      unknown
    ===================   =====================

This is is based on the original dataset, that was used in the ISMIR paper that is referenced below with 138.338 triplets
and 413 artists, but make some modifications. We drop triplets that are missing the third (other) entry.
Some artists in the triplets are missing in the provided name list, we call them 'unknown_0', 'unknown_1', etc.

This dataset can be downloaded using the :func:`cblearn.datasets.fetch_musician_similarity`.

When using these triplets, please give credit to the original authors.

.. topic:: References

    - Ellis, D. P., Whitman, B., Berenzweig, A., & Lawrence, S. (2002).
      The quest for ground truth in musical artist similarity.
