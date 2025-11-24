Installation
============

Is is recommended to create a separate Python environment for this package, e.g.

.. code-block:: bash

    micromamba create -n quvac python=3.12

After cloning the git repository and entering it, choose relevant optional dependencies

- ``[test]`` allows to run tests
- ``[plot]`` installs ``matplotlib``
- ``[optimization]`` installs Bayesian optimization package

To install all dependencies, run

.. code-block:: bash

    pip install .[all]

.. note::
    For example, if you do not require optimization capabilities, run
     
    .. code-block:: bash
        
        pip install .[test,plot]

After successfull installation with ``[all]`` or ``[test]`` option, run ``pytest`` to make sure the installation was
successfull (it takes some time).

.. code-block:: bash

    pytest

.. note::
    By default ``pytest`` runs only unit tests without any physics benchmark (files ``tests/test_*.py```).

    For minimal physics benchmarking run ``pytest -m 'not slow' -m 'benchmark'`` (files ``tests/bench_*.py``).

    Benchmarks have a couple of time-demanding tests, they could be launched with ``pytest -m 'slow' -m 'benchmark'``.