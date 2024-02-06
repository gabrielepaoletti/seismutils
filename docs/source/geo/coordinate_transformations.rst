Coordinate transformations
======================================

.. autofunction:: seismutils.geo.convert_to_geographical


.. autofunction:: seismutils.geo.convert_to_utm

**Parameter options**

Some parameters of the ``convert_to_utm`` function accept multiple options, allowing you to customize the conversion process according to your needs. Below is a brief overview of these parameters and their possible values:

- ``units``: Specifies the units of the UTM coordinates passed as input. Can be:
  
  - ``'m'`` for meters.
  - ``'km'`` for kilometers.

**Example usage**

To use the ``convert_to_utm()`` function, you must import it from the ``seismutils.geo`` sub-module. 

Below is an example of converting geographical coordinates (longitude and latitude) from zone 33N, with a longitude of 13.271772 and a latitude of 38.836032, to UTM coordinates. These coordinates are assumed to be based on the WGS84 ellipsoid and datum, and the output UTM coordinates are expressed in kilometers (the ``units`` parameter).

.. code-block:: python

    import seismutils.geo as sug

    lon, lat = 13.271772, 38.836032  # Geographical coordinates

    utmx, utmy = sug.convert_to_utm(
        lon=lon,
        lat=lat,
        zone=33,
        units='km',
    )

    print(f'UTMX: {utmx}, UTMY: {utmy}')
    # Output: UTMX: 350, UTMY: 4300

.. note:: 
    The ``convert_to_utm()`` function is also designed to handle input data in the form of arrays or pandas Series. 

When either of these data types is provided as input for the geographical coordinates, the function will return an array containing the converted UTM coordinates for each set of geographical coordinates in the input array or Series.