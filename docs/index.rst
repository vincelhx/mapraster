################################################
mapraster: raster-to-image mapping utilities
################################################

**mapraster** provides utilities to map a lon/lat raster onto an image grid
defined by 2D longitude/latitude coordinates.

It works with `xarray`_ datasets and supports antimeridian crossing and NaN handling.

Documentation
-------------

Getting Started
...............

:doc:`installing`
~~~~~~~~~~~~~~~~~

Examples
........

* :doc:`examples/how_to_use`

Help & Reference
................

:doc:`basic_api`
~~~~~~~~~~~~~~~~

Tests
.....

The latest automated test report (generated during docs build) is available here:

`Open test report <test-report.html>`_

----------------------------------------------

Last documentation build: |today|


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   installing

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Examples

   examples/how_to_use.ipynb

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   basic_api


.. _xarray: http://xarray.pydata.org

