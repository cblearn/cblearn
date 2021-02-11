.. _food_similarity_dataset:

Food Similarity dataset
-----------------------

`The food dataset contains triplets`_ collected from Amazon Mechanical Turk in 2014.

The crowd workers were presented a target and multiple other of the 100 food images.
They selected a fixed number of other images, which taste more similar to the target than the remaining.
Per user selection, multiple triplet constraints were created.

.. _The food dataset contains triplets: https://vision.cornell.edu/se3/projects/cost-effective-hits/

**Data Set Characteristics:**

    ===================   =====================
    Triplets                             190376
    Objects                                 100
    Dimensionality                      unknown
    ===================   =====================

This dataset can be downloaded using the :func:`cblearn.datasets.fetch_food_similarity`.

.. License statement from the original homepage

    This dataset contains copyrighted material under the educational fair use exemption to the U.S. copyright law.

When using this data, please consider the fair use statement above and give credit to the original authors.

.. topic:: References

    - Wilber, M. J., Kwak, I. S., & Belongie, S. J. (2014).
      Cost-effective hits for relative similarity comparisons. arXiv preprint arXiv:1404.3291.