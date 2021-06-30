.. _imagenet_similarity_dataset:

Imagenet Similarity dataset
---------------------------

This dataset contains comparison trials of images from the ImageNet validation dataset (ILSVRC-2012).
In an crowd sourced experiments, subjects ranked two out of 8 images that appeared most similar to a reference image.
The trials where selected in an active learning routine, such that they already are not too dissimilar within a trial.

There are two versions of this dataset: Version "0.2" has trials for
all 50 ImageNet validation images per class, version "0.1" has trials for a single image per class.

The whole `dataset`_ is published under CC-By Attribution 4.0 International by Brett Roads.

.. _dataset: https://osf.io/cn2s3/

**Data Set Characteristics:**

    ===================   =====================
    Trials    v0.1/v0.2        25,273 / 384,277
    Objects (Images)             1,000 / 50,000
    Classes                               1,000
    Query                         rank 2 from 8
    ===================   =====================

This dataset can be downloaded using the :func:`cblearn.datasets.fetch_imagenet_similarity`.
To use the 8-rank-2 trials with e.g. ordinal embedding algorithms, they can be converted to standard triplets
with :func:`cblearn.preprocessing.triplets_from_multiselect`.

Please cite the following paper if you use this dataset in publications.

.. topic:: References

    - Roads, B. D., & Love, B. C. (2020). Enriching ImageNet with Human Similarity Judgments
      and Psychological Embeddings. ArXiv:2011.11015 [Cs]. http://arxiv.org/abs/2011.11015
