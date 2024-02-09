API Reference
=============

This reference manual offers detailed explanations of the functions, modules, and objects contained within SeismUtils, outlining their purposes and functionalities. To learn how to apply SeismUtils, please refer to the `tutorials page <https://seismutils.readthedocs.io/en/latest/user_guide/tutorials.html>`_.

Module Structure
^^^^^^^^^^^^^^^^

SeismUtils is structured to meet the diverse demands of seismology experts, featuring two primary submodules for enhanced functionality:

.. list-table:: 
   :widths: 25 75
   :header-rows: 0

   * - ``seismutils.geo``
     - Tailored for geospatial data handling, it enables users to work with and understand geographical information, facilitating the mapping and exploration of areas related to seismic activities.
   * - ``seismutils.signal``
     - Focuses on the processing and analysis of seismic waveform data, providing tools for in-depth signal examination.

Functions
---------

.. toctree::
   :maxdepth: 1
   :caption: Geospatial Data

   geo/coordinate_transformations
   geo/data_querying_and_selection
   geo/seismic_visualization

.. toctree::
   :maxdepth: 1
   :caption: Waveform Analysis

   signal/signal_processing
   signal/spectral_analysis