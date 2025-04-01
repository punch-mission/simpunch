Usage
=============

How do I use simpunch?
----------------------
To begin, you'll need sample model data with which to pass through the pipeline. During development, we used this tool extensively with GAMERA model data passed through FORWARD to produce integrated values of total brightness and polarized brightness along a particular line of sight through the model. An outline of the overall flow, along with each individual processing level is given below.

Overall flow
------------

To run the entire simpunch processing pipeline with input model data, the ``simpunch.flow.generate_flow`` function can be called.

.. autofunction:: simpunch.flow.generate_flow
    :no-index:

Level 0
-------

To generate level 0 data in the mzp polarized framework, the ``simpunch.level0.generate_l0_pmzp`` function can be called.

.. autofunction:: simpunch.level0.generate_l0_pmzp
    :no-index:

To generate level 0 clear data, the ``simpunch.level0.generate_l0_cr`` function can be called.

.. autofunction:: simpunch.level0.generate_l0_cr
    :no-index:

Level 1
-------

To generate level 1 data in the mzp polarized framework, the ``simpunch.level1.generate_l1_pmzp`` function can be called.

.. autofunction:: simpunch.level1.generate_l1_pmzp
    :no-index:

To generate level 1 clear data, the ``simpunch.level1.generate_l1_cr`` function can be called.

.. autofunction:: simpunch.level1.generate_l1_cr
    :no-index:

Level 2
-------

To generate a level 2 polarized trefoil mosaic, the ``simpunch.level2.generate_l2_ptm`` function can be called.

.. autofunction:: simpunch.level2.generate_l2_ptm
    :no-index:

To generate a level 2 clear trefoil mosaic, the ``simpunch.level2.generate_l2_ctm`` function can be called.

.. autofunction:: simpunch.level2.generate_l2_ctm
    :no-index:

Level 3
-------

To generate a level 3 polarized trefoil mosaic, the ``simpunch.level2.generate_l3_ptm`` function can be called.

.. autofunction:: simpunch.level3.generate_l3_ptm
    :no-index:

To generate a level 3 clear trefoil mosaic, the ``simpunch.level2.generate_l3_ctm`` function can be called.

.. autofunction:: simpunch.level3.generate_l3_ctm
    :no-index:
