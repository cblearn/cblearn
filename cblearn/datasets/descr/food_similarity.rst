.. _food_similarity_dataset:

Food Similarity dataset
-----------------------

`The food dataset contains triplets`_ collected from Amazon Mechanical Turk in 2014.

The crowd workers were presented a target and multiple other of the 100 food images.
They selected a fixed number of other images, which taste more similar to the target than the remaining.
Per user selection, multiple triplet constraints were created.

.. _The food dataset contains triplets: https://web.archive.org/web/20250605221918/https://vision.cornell.edu/se3/projects/cost-effective-hits/

**Data Set Characteristics:**

    ===================   =====================
    Triplets                             190376
    Objects                                 100
    Dimensionality                      unknown
    ===================   =====================

This dataset can be downloaded using the :func:`cblearn.datasets.fetch_food_similarity`.

.. note::
    The original download host, vision.cornell.edu, is no longer online, so the link
    above points to an archived copy of the project page.
    :func:`cblearn.datasets.fetch_food_similarity` downloads the dataset from an
    `OSF mirror <https://osf.io/tqpmw/>`_ of the original archive instead.
    The mirrored file is byte-identical to the original, which its unchanged
    checksum confirms.

.. License statement from the original homepage

    This dataset contains copyrighted material under the educational fair use exemption to the U.S. copyright law.

When using this data, please consider the fair use statement above and give credit to the original authors.

.. topic:: References

    - Wilber, M. J., Kwak, I. S., & Belongie, S. J. (2014).
      Cost-effective hits for relative similarity comparisons. arXiv preprint arXiv:1404.3291.