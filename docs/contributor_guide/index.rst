=================
Contributor Guide
=================

There are multiple ways to contribute to this project.
You can report bugs in this library or propose new ideas via `Github issues`_.
This guide describes how to contribute code or documentation.

.. _Github issues: https://github.com/dekuenstle/ordcomp/issues


---------------
Getting Started
---------------

We assume, you downloaded and installed OrdComp as described in :ref:`developer_install`.

The project directory contains the code directory ``ordcomp/`` and the documentation ``docs/``.
In addition, there are readme, license, and a multiple configuration files as well as an examples folder.

-------------
Changing Code
-------------

The Python code is structured in :ref:`modules`. Each module contains
a `tests` folder with unit-tests.
There should be such a test for every method and function.
Use ``pytest --cov`` to run these tests and to measure the coverage, no tests should fail.
The coverage indicates the tested fraction of code and should be close to *100%*.

All Python code follows the `PEP8 Style Guide`_. The style
of all code can be checked, running ``flake8 .`` and should print no warnings.

Every class, method, and function should also have a docstring, describing the functionality and parameters.
Please follow the `Google Docstring Style`_.
The docstring will be added to the :ref:`api_ref` by adding the function name in ``docs/references/index.rst``.
Check the syntax of the docstring by running ``make html`` in the ``docs/`` folder.

Types should not be added to the docstring, but in the code as `type hints`_.
Typechecks can be performed using ``mypy ordcomp``.

.. _PEP8 Style Guide: https://www.python.org/dev/peps/pep-0008/
.. _Google Docstring Style: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html
.. _type hints: https://docs.python.org/3/library/typing.html

----------------------
Changing Documentation
----------------------

The documentation is contained in the `docs/` folder.
It can be build by running ``make html``.
Open ``docs/_build/html/index.html`` in a browser to view the local build of the documentation.

The documentation is structured in multiple folders and written as `reStructuredText`_.

.. _reStructuredText: https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html

------------------
Publish Changes
------------------

Most contributions will change files either in the code or in the documentation directory, as described in the
sections below. Commit your changes to a separate *git* branch (do **not** commit to ``master``).
After you finished changing push this branch to Github and open a pull request to the ``master`` branch there.
Once the request is opened, automated tests are run.
If these tests indicate a problem, you can fix this problem on your branch and push again.
Once the automated tests are successful, maintainers of OrdComp will review the changes and provide feedback.
Usually after some iterations, your changes will be merged to the ``master`` branch.

.. Note:

    If you state a pull request, your changes will be published under `this open source license`_.

.. _this open source license: https://github.com/dekuenstle/ordcomp/blob/master/LICENSE



