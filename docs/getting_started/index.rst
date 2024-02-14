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

``cblearn`` and can be installed using `pip`:

.. code-block:: bash

    pip install cblearn


This will install the minimal set of required packages, sufficient for most uses and saving disk space.
However, some features require more packages that can be installed by adding an ``option`` to the install command.

.. _extras_install:

Install Extra Requirements
==========================

Extra requirements can be installed by adding an ``option`` to the install command and enable more advanced features.
Some of those extra dependencies need non-Python packages to be installed first.

.. code-block:: bash

    $ pip install cblearn[torch,wrapper] h5py


torch
    Most estimators provide an (optional) implementation using ``pytorch`` to run large datasets on CPU and GPU.
    This requires the ``pytorch`` package to be installed `manually <https://pytorch.org/get-started/locally/>`_
    or by adding the ``torch``` option to the install command.
    Note that ``pytorch`` might need about 1GB of disk space.

wrapper
    The estimators in :ref:`references_embedding_wrapper` provide an Python interface to the original implementation
    in ``R``-lang.
    This requires the ``rpy2`` package to be installed by adding the ``wrapper`` option to the install command.
    Additionally, this requires an installed ``R`` interpreter whit must available be in the ``PATH`` environment variable.
    The ``R`` packages are installed automatically upon the first use of the estimators.

h5py
    The function :func:`cblearn.datasets.fetch_imagenet_similarity` requires the ``h5py`` package to load the dataset.
    This can package can be installed with pip.
    Note that some platforms require additionally the ``hdf5`` libraries to be installed `manually <https://www.hdfgroup.org/downloads/hdf5/>`_.


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