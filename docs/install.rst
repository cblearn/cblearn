============
Installation
============

``cblearn`` requires Python 3.9 or newer.
We recommend using Anaconda_ to install Python and
dependencies in separated environments.
The package is mainly tested on Linux, but Windows and Mac OS should work, too.

.. _Anaconda: https://docs.anaconda.com/anaconda/install/

.. code-block:: bash

    conda create -n cblearn python==3.9
    conda activate cblearn


-----------------
User Installation
-----------------

``cblearn`` and can be installed using `pip`:

.. code-block:: bash

    pip install cblearn

This will install the minimal set of required packages, sufficient for most uses and saving disk space.
However, some features require more packages that can be installed by adding an ``option`` to the install command.

.. _extras_install:

Extra Requirements
===================

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
    Note that some platforms require additionally the ``hdf5`` libraries to be installed
     `manually <https://www.hdfgroup.org/downloads/hdf5/>`_.


.. _developer_install:

------------------------
Contributor Installation
------------------------


If you want to make changes to the code or documentation, you should
first download the repository and install the project in developer mode with
developer dependencies.  This way, changes in the code are directly considered without the need for re-installation.
Additionally, packages required to run the tests and build the documentation are installed.

.. code-block:: bash

    $ git clone git@github.com:cblearn/cblearn.git
    $ cd cblearn
    $ pip install -e.[tests,docs,torch,wrapper]

The ``-e`` option installs the package in developer mode such that changes in the code are considered directly without re-installation.

tests
    To run the unit tests, the ``pytest`` package is required, which
    can be installed by adding the ``tests`` option to the install command.

docs
    Building these docs requires the ``sphinx`` package, which can be installed by adding the `docs` option to the install command.


Now you can run the tests and build the documentation:

.. code-block:: bash

    $ python -m pytest --remote-data  # should run all tests; this can take a while.

    $ cd docs
    $ make html  # should generate docs/_build/html/index.html