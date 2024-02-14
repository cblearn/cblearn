.. _getting_started:

================
Getting Started
================

-----
Setup
-----

``cblearn`` requires Python 3.9 or newer.
The package is mainly tested on Linux, but Windows and Mac OS should work, too.

Python environment
==================
The easiest way to install Python and its dependencies is using a
 Anaconda_ environment or similar, because dependencies do not conflict with
 other Python packages you may have installed and the Python version can be specified.

.. _Anaconda: https://docs.anaconda.com/anaconda/install/

.. code-block:: bash

    conda create -n cblearn python==3.9
    conda activate cblearn


Install cblearn
===================

``cblearn`` and its dependencies can be installed using `pip`:

.. code-block:: bash

    pip install cblearn



.. _extras_install:

Install Extra Requirements
==========================

The installation routine above installs a minimal set of required packages, sufficient
for most uses.
However, some features require more packages that can be installed by adding
an `option` to the install command..

For example, to use estimators on GPU, based on ``pytorch``, and estimators
wrapping paper author's original implementation in ``R``-lang:

.. code-block:: bash

    $ pip install cblearn[torch,wrapper]

======= =============================================================
Extras  Description
======= =============================================================
torch   Estimators which can run on a GPU, implemented using pytorch.
wrapper Estimators which wrap implementations in R or Matlab/Octave.
tests   Test runner for unit tests (required to contribute).
docs    Build the documentation (required to contribute).
======= =============================================================

-----------
Quick Start
-----------

`cblearn` is designed to be easy to use.
The following example generates triplets from a point cloud and fits an ordinal embedding
model to the triplets.

.. literalinclude:: ./quickstart.py
   :language: python
   :linenos:

The output should show a trend similar to the following::

    Triplets | Error (SSE)
    ----------------------
          25 |       0.913
         100 |       0.278
         400 |       0.053
        1600 |       0.001

The procrustes distance measures the sum of squared errors between points and embedding,
after aligning the embedding to the points (i.e., by optimizing rotating, translation, and scaling).
The error approaches zero, demonstrating, that the relative distances of the point cloud can be reconstructed from triplets only.

The `result_format` option of the triplet generator can be used to explore the different
data formats that can be used to represent triplets.
Besides the `list-order` format, using a numpy array with shape `(n, 3)` (containing entries of indices of the points in the order *anchor*, *near*, *far*),
triplets can be represented along responses as a pair of numpy arrays `((n, 3), (n,))` (e.g., `estimator.fit((triplets, responses))`) or as positions in a 3-dimensional sparse matrix.

From here, you might look for more theoretical insight in the :ref:`user_guide`,
look at practical :ref:`examples`, or get an overview of the :ref:`API`.