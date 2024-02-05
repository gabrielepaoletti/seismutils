Data querying and selection
======================================

.. autofunction:: seismutils.geo.select

**Parameter options**

Some parameters of the ``select()`` function accept multiple options, allowing you to customize the geometric shape used for data selection. Below is a brief overview of these parameters and their possible values:

- ``shape_type``: Specifies the type of geometric shape to use for selecting data within the dataset. Can be:

  - ``'circle'``: Selects data within a circular area. This shape is defined by a single radius value.
  - ``'oval'``: Selects data within an oval area. This shape requires specifying both the width and height.
  - ``'rectangle'``: Selects data within a rectangular area. Like the oval, this shape is defined by its width and height.

