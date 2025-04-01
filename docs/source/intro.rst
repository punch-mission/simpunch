Introduction
=============

What is PUNCH?
--------------
PUNCH is a NASA Small Explorer (SMEX) mission to better understand how the mass and energy of
the Sun’s corona become the solar wind that fills the solar system.
Four suitcase-sized satellites will work together to produce images of the entire inner solar system around the clock.
You can learn more at the `PUNCH website <https://punch.space.swri.edu/>`_.

Where does `simpunch` fit in?
--------------------------------------
``simpunch`` is a framework for generating synthetic PUNCH data, a useful for testing the science reduction pipeline during development. We are opening this code as a tool for the scientific community for generating synthetic datasets from model outputs. This can prove useful for comparisons of actual PUNCH observations with simulated model data.

PUNCH and Python
----------------

The PUNCH framework is built using Python - an object-oriented language with a large user / code base in astronomy and solar physics. The pipeline and tools for querying / loading PUNCH data use the Python language, along with the SunPy and Astropy software libraries. A number of useful tutorials exist online, including the official `Python tutorial <https://docs.python.org/3/tutorial/index.html>`_ and the `Hitchhiker's Guide to Python <https://docs.python-guide.org>`_.

In addition to scripts and modules, Python notebooks provide a great way to execute and document a sequence of code cells, with visualizations directly in-line. It's a sort of analogue of the classic research notebook. The `SunPy example gallery <https://docs.sunpy.org/en/stable/generated/gallery/index.html>`_ provides a great series of example notebooks, which are an additional great tool for learning Python.
