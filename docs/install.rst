============
Installation
============

``cblearn`` requires Python 3.9 or newer.
We recommend using Anaconda_ to install Python and
dependencies in separated environments.
We mainly test on Linux (tested on Ubuntu 20.4), but Windows and Mac OS should also work.
Examples in this installation guide Linux shell commands.

.. _Anaconda: https://docs.anaconda.com/anaconda/install/

.. code-block:: bash

    conda create -n cblearn python==3.9
    conda activate cblearn


-----------------
User Installation
-----------------

``cblearn`` and its dependencies can be installed using `pip`:

.. code-block:: bash

    pip install cblearn


.. _extras_install:

Extras Requirements
===================

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


.. _developer_install:

------------------------
Contributor Installation
------------------------

If you want to make changes to the code or documentation, you should
first download the repository and install the project in developer mode with
developer dependencies.
This way, changes in the code are directly considered without the need for re-installation.

.. code-block:: bash

    $ git clone git@github.com/cblearn/cblearn.git
    $ cd cblearn
    $ pip install -e.[tests,docs]
