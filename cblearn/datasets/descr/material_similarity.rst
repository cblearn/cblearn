.. _material_similarity_dataset:

Material Similarity dataset
---------------------------

This dataset contains triplets of 100 material images, gathered in a crowd sourced experiment.
The subjects chose for triplets of one reference and two candidate images
"Which of these two candidates has a more similar appearance to the reference?".
The trials where actively chosen such that they maximize the information gain (CKL algorithm).

Experimental code and the material images are available at the dataset author's _`Github repository`.
.. _Github repository: https://github.com/mlagunas/material-appearance-similarity

**Data Set Characteristics:**

    ===================   =====================
    Triplets Train/Test            22801 / 3000
    Responses                     92892 / 11800
    Objects (Materials)                     100
    ===================   =====================

This dataset can be downloaded using the :func:`cblearn.datasets.fetch_material_similarity`.
Most triplets where responded multiple times, often contradictory.

Please cite the following paper if you use this dataset in publications.

.. topic:: References

    - Lagunas, M., Malpica, S., Serrano, A., Garces, E., Gutierrez, D., & Masia, B. (2019).
      A Similarity Measure for Material Appearance. ACM Transactions on Graphics, 38(4), 1–12.
