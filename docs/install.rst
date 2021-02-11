============
Installation
============

cblearn requires Python 3.7 or newer.
We recommend using [Anaconda](https://docs.anaconda.com/anaconda/install/) to install Python and
dependencies in separated environments.
We support Linux (tested on Ubuntu 20.4), Windows and Mac OS.
Examples in this installation guide Linux shell commands.

-----------------
User Installation
-----------------

cblearn and its dependencies can be installed using `pip`:

.. code-block:: bash

    $ pip install git+ssh://git@github.com/dekuenstle/cblearn.git


.. _developer_install:

------------------------
Contributor Installation
------------------------

If you want to make changes to the code or documentation, you should
first download the repository and install the project in developer mode with
developer dependencies.
This way, changes in the code are directly considered without the need of re-installation.

.. code-block:: bash

    $ git clone git@github.com/dekuenstle/cblearn.git
    $ cd cblearn
    $ pip install -e.[tests,docs]
