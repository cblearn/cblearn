.. _central_car_dataset:

Car Similarity dataset
-----------------------

`This dataset contains triplets`_ of 60 car images, responsed in an online survey.
The people chose one car of three, such that the following statement is true:
"Object A is the most central object within the triple of objects (A,B,C)".

All images were found on Wikimedia Commons and are assigned to one of four classes:
ORDINARY CARS, SPORTS CARS, OFF-ROAD/SPORT UTILITY VEHICLES, and OUTLIERS.

The corresponding car images are available with the _`full dataset`.
.. _full dataset: http://www.tml.cs.uni-tuebingen.de/team/luxburg/code_and_data/index.php

**Data Set Characteristics:**

    ===================   =====================
    Triplets                               7097
    Objects (Cars)                           60
    Query                  3 cars, most-central
    ===================   =====================

This dataset can be downloaded using the :func:`cblearn.datasets.fetch_car_similarity`.
To use the most-central triplets with e.g. ordinal embedding algorithms, you should convert them to standard triplets
(:func:`cblearn.dataset.triplets_from_mostcentral`).

Please cite the following paper if you use this dataset in publications.

.. topic:: References

    - M. Kleindessner and U. von Luxburg. Lens depth function and k-relative neighborhood graph:
      Versatile tools for ordinal data analysis. JMLR, 18(58):1–52, 2017.
