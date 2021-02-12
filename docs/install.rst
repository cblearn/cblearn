============
Installation
============

OrdComp requires Python 3.7 or newer.
We recommend using Anaconda_ to install Python and
dependencies in separated environments.
We support Linux (tested on Ubuntu 20.4), Windows and Mac OS.
Examples in this installation guide Linux shell commands.

.. _Anaconda: https://docs.anaconda.com/anaconda/install/

-----------------
User Installation
-----------------

OrdComp and its dependencies can be installed using `pip`:

.. code-block:: bash

    $ pip install git+ssh://git@github.com/dekuenstle/ordcomp.git

.. _extras_install:

Extras Requirements
===================

The installation routine above tries to only install a minimal set of required packages.
Some of OrdComp's features depend on extra dependencies.
These can be installed by adding them comma-separated in squared brackets to the pip command.

For example, to use estimators on GPU, based on pytorch, and estimators
wrapping paper author's original implementation in R-lang:

.. code-block:: bash

    $ pip install git+ssh://git@github.com/dekuenstle/ordcomp.git[torch,wrapper]

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
This way, changes in the code are directly considered without the need of re-installation.

.. code-block:: bash

    $ git clone git@github.com/dekuenstle/ordcomp.git
    $ cd ordcomp
    $ pip install -e.[tests,docs]
