Introduction
============

This library aims to provide a unified interface for inverting
images captured by
`computed tomography imaging spectrographs <https://en.wikipedia.org/wiki/Computed_tomography_imaging_spectrometer>`_
(CTISs), particularly those designed for observing the Sun in extreme
ultraviolet.

Discussions
===========

Some explanations of the theory behind inversion

.. toctree::
    :maxdepth: 1

    discussions/mart-discussion
    discussions/richardson-lucy-analogy/richardson-lucy-analogy

Tutorials
=========

Examples on how to use this package.

.. toctree::
    :maxdepth: 1

    tutorials/ideal-instrument
    tutorials/simple-mart
    tutorials/parametric-fit

Citation
========

If you use :mod:`ctis` in your research, please cite it.
The citation metadata is kept in
`CITATION.cff <https://github.com/sun-data/ctis/blob/main/CITATION.cff>`_,
which the "Cite this repository" button on the
`GitHub page <https://github.com/sun-data/ctis>`_
can export as BibTeX or APA.
Please include the version of :mod:`ctis` that you used,
which is given by ``importlib.metadata.version("ctis")``.

.. code-block:: bibtex

    @software{ctis,
      author = {Smart, Roy T. and Parker, Jacob D. and Kankelborg, Charles C.},
      title = {ctis},
      version = {X.Y.Z},
      url = {https://github.com/sun-data/ctis},
    }

API Reference
=============

.. autosummary::
    :toctree: _autosummary
    :template: module_custom.rst
    :recursive:

    ctis


References
==========

.. bibliography::

|


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
