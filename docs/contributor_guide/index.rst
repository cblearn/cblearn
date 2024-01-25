=================
Contributor Guide
=================

There are multiple ways to contribute to this project.
You can report bugs in this library or propose new ideas via `Github issues`_.
This guide describes how to contribute code or documentation.

.. _Github issues: https://github.com/dekuenstle/cblearn/issues


---------------
Getting Started
---------------

We assume you downloaded and installed ``cblearn`` as described in :ref:`developer_install`.

The project directory contains the code directory ``cblearn/`` and the documentation ``docs/``.
In addition, the folder contains a readme, license, multiple configuration files, and an examples folder.

-------------
Changing Code
-------------

The Python code is structured in :ref:`modules`. Each module contains
a `tests` folder with unit tests.
There should be such a test for every method and function.
Use ``pytest --cov`` to run these tests and to measure the coverage; no tests should fail.
The coverage indicates the tested fraction of code and should be close to *100%*.

All Python code follows the `PEP8 Style Guide`_. The style
of all code can be checked, running ``flake8 .`` and should print no warnings.

Every class, method, and function should have a docstring describing the functionality and parameters.
Please follow the `Google Docstring Style`_.
The docstring will be added to the :ref:`api_ref` by adding the function name in ``docs/references/index.rst``.
Check the syntax of the docstring by running ``make html`` in the ``docs/`` folder.

Types should not be added to the docstring but in the code as `type hints`_.
Typechecks can be performed using ``mypy cblearn``.

.. _PEP8 Style Guide: https://www.python.org/dev/peps/pep-0008/
.. _Google Docstring Style: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html
.. _type hints: https://docs.python.org/3/library/typing.html

Remote data tests
-----------------
Tests that require remote data, for example fetching a dataset from the internet, are marked with ``@pytest.mark.remote_data``
or ``+REMOTE_DATA`` (docstring).
These tests are skipped by default but can be run by adding the ``--remote-data`` flag to ``pytest``.

Scikit-learn estimator tests
----------------------------
``scikit-learn`` provides a test suite that should ensure the compatibility of estimators.
We use this test suite to test our estimators, too, but have to skip some tests because they use artificial data incompatible
to comparison data. Typically, ``cblearn`` estimators are compatible with ``scikit-learn`` estimators
if comparisons are represented as ``numpy`` arrays. From an API perspective,
comparison arrays look like discrete features and class labels; however, not all discrete features and class labels are valid comparisons.

In the future scikit-learn might simplify the usage of custom data generation routines during the compatibility tests.
Otherwise, we might replace those incompatible tests with our own.

All sklearn estimator tests can be skipped with ``pytest -m "not sklearn``.

----------------------
Changing Documentation
----------------------

The documentation is contained in the `docs/` folder.
It can be built by running ``make html``.
Open ``docs/_build/html/index.html`` in a browser to view the local build of the documentation.

The documentation is structured in multiple folders and written as `reStructuredText`_.

.. _reStructuredText: https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html

-----------------------------------
Excursion: Run Github Tests Locally
-----------------------------------

Instead of running the different tests above independently, it is also possible
to run the whole testing workflow, which is used on Github, locally.

Install nektos' `act`_ and then run `act -P ubuntu-latest=nektos/act-environments-ubuntu:18.04-full`

`act` uses docker images with preinstalled software to provide almost the same test environment as Github.
If it is not yet so, you have to `install docker`_ and, optionally, make it accessible for non-root users.

.. note::
    The docker image requires about 18 GB disk space. The first start of act might take some time,
    because it downloads about 12 GB of image files.

.. _act: https://github.com/nektos/act
.. _`install docker`: https://docs-stage.docker.com/engine/install/
.. _`accessible for nonroot user`: https://docs.docker.com/engine/install/linux-postinstall/

------------------
Publish Changes
------------------

Most contributions will change files in the code or the documentation directory, as described in the
sections below. Commit your changes to a separate *git* branch (do **not** commit to ``master``).
After changing, push this branch to Github and open a pull request to the ``master`` branch there.
Once the request is opened, automated tests are run.
If these tests indicate a problem, you can fix this problem on your branch and push again.
Once the automated tests are successful, maintainers of ``cblearn`` will review the changes and provide feedback.
Usually, after some iterations, your changes will be merged into the ``main`` branch.

.. Note:

    If you state a pull request, your changes will be published under `this open source license`_.

.. _this open source license: https://github.com/dekuenstle/cblearn/blob/master/LICENSE


Versions should be semantic and follow PIP440_: The version indicates ``major.minor.fix``;
breaking changes are just allowed with major version steps.
A Github release tag indicates a new version, which triggers a continuous deployment to PyPI via Github Actions.

.. _PIP440: https://peps.python.org/pep-0440/
