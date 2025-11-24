quvac documentation
===================

.. .. image:: ../../images/logo.jpg
..    :alt: Quvac logo
..    :width: 300px
..    :align: center

Welcome to quvac documentation!

Quvac (from quantum vacuum, pronounced as qu-ack 🐸) allows to calculate quantum 
vacuum signals produced during light-by-light scattering. 

It uses the vacuum emission picture [1]_ to describe the corresponding transition amplitude
and the linear Maxwell solver to propagate external fields. The original idea of such code
was put forward in [2]_. Here we provide our implementation.

.. toctree::
   :maxdepth: 2

   installation
   usage
   input_file
   tutorials
   implementation
   for_developers


Acknowledgements
^^^^^^^^^^^^^^^^

If you use this code and/or consider it useful, please cite our article.

.. code-block:: bibtex
   
   @article{valialshchikov2025back,
   title={Back-reflection in dipole fields and beyond},
   author={Valialshchikov, Maksim and Karbstein, Felix and Seipt, Daniel and Zepf, Matt},
   journal={arXiv preprint arXiv:2510.11764},
   year={2025}
   }

References
^^^^^^^^^^^^^^^^

.. [1] F. Karbstein, and R. Shaisultanov. "Stimulated photon emission from the vacuum." 
   PRD 91.11 (2015): 113002 `(article) <https://arxiv.org/abs/1412.6050>`_.

.. [2] A. Blinne, et al. "All-optical signatures of quantum vacuum nonlinearities in 
   generic laser fields." PRD 99.1 (2019): 016006 `(article) <https://arxiv.org/abs/1811.08895>`_.