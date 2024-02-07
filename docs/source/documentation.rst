API Overview
======================================

Designed to cater to the nuanced needs of seismology professionals, ``seismutils`` is divided into two main submodules:

.. list-table:: 
   :widths: 25 75
   :header-rows: 0

   * - ``seismutils.geo``
     - Provides functionalities for geospatial exploration and earthquake location mapping. 
   * - ``seismutils.signal``
     - Functions for processing and analyzing seismic waveform data.

Each submodule is tailored to facilitate specific aspects of seismic data manipulation, visualization, and analysis, ensuring an efficient and streamlined workflow for users.

Documentation
======================================

Here, detailed explanations of each function within the ``seismutils`` toolkit are provided, showcasing their use through practical examples and mini-tutorials.

.. toctree::
   :maxdepth: 3
   :caption: seismutils.geo

   geo/coordinate_transformations
   geo/data_querying_and_selection
   geo/seismic_visualization

.. toctree::
   :maxdepth: 2
   :caption: seismutils.signal

   signal/signal_processing
   signal/spectral_analysis