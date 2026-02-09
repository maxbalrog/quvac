Installation
============

Is is recommended to create a separate Python environment for this package, e.g.

.. code-block:: bash

    micromamba create -n quvac python=3.12

After cloning the git repository and entering it, choose relevant optional dependencies

- ``[test]`` allows to run tests
- ``[plot]`` installs ``matplotlib`` and ``jupyterlab``
- ``[docs]`` installs ``sphinx`` and everything necessary for documentation generation
- ``[optimization]`` installs Bayesian optimization package
- ``[light]`` is a shorthand for ``[test,plot,docs]``

To install all dependencies, run

.. code-block:: bash

    pip install .[all]

.. note::
    For example, if you do not require optimization capabilities, run
     
    .. code-block:: bash
        
        pip install .[test,plot]

After successfull installation with ``[all]``, ``[light]`` or ``[test]`` options, run ``pytest`` to make sure the installation was
successfull (it takes some time).

.. code-block:: bash

    pytest

.. note::
    By default ``pytest`` runs only unit tests without any physics benchmark (files ``tests/test_*.py```).

    For minimal physics benchmarking run ``pytest -m 'not slow' -m 'benchmark'`` (files ``tests/bench_*.py``).

    Benchmarks have a couple of time-demanding tests, they could be launched with ``pytest -m 'slow' -m 'benchmark'``.


Using ``uv``
------------

If you prefer using ``uv`` package manager then the installation follows similar steps. After cloning the git repository and entering 
it, create the environment and install ``quvac``

.. code-block:: bash

    uv venv
    uv pip install .[light]

You can test the installation with

.. code-block:: bash

    uv run pytest

Launch the jupyterlab (e.g. tutorial notebooks) with

.. code-block:: bash

    uv run jupyter lab

Generate the documentation with

.. code-block:: bash
    
    uv run python -m sphinx -b html docs/source docs/build/html


