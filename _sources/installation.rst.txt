Installation
============

Is is recommended to create a separate Python environment for this package, e.g.

.. code-block:: bash

    micromamba create -n quvac python=3.12

After cloning the git repository it could be simply installed with

.. code-block:: bash

    pip install quvac/

To make sure the installation was successfull run ``pytest`` (it takes some time).

.. code-block:: bash

    pytest

.. note::
    By default ``pytest`` runs only unit tests without any physics benchmark (files ``tests/test_*.py```).

    For minimal physics benchmarking run ``pytest -m 'not slow' -m 'benchmark'`` (files ``tests/bench_*.py``).

    Benchmarks have a couple of time-demanding tests, they could be launched with ``pytest -m 'slow' -m 'benchmark'``.