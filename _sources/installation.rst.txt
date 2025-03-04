Installation
============

Is is recommended to create a separate Python environment for this package, e.g.

.. code-block:: bash

    micromamba create -n quvac python=3.12

After cloning the git repository it could be simply installed with

.. code-block:: bash

    pip install quantum-vacuum

After successfull installation run ``pytest`` to make sure the installation was
successfull (it takes some time).

.. code-block:: bash

    pytest