Coordinate transformations
======================================

.. autofunction:: seismutils.geo.convert_to_geographical

Example usage
^^^^^^^^^^^^^

To use the ``convert_to_geographical`` function, you must import it from the ``seismutils.geo`` sub-module. 

Below is an example of converting UTM coordinates from zone 33N, with an easting of 350 and a northing of 4300, to geographical coordinates. These coordinates are assumed to be in kilometers (the ``units`` parameter) and based on the WGS84 ellipsoid and datum.

.. code-block:: python

    import seismutils.geo as sug

    utmx, utmy = 350, 4300 # Expressed in kilometers

    lon, lat = sug.convert_to_geographical(
        utmx=utmx,
        utmy=utmy,
        zone=33,
        northern=True,
        units='km',
    )

    print(f'Longitude: {lon}, Latitude: {lat}')
    # Output: Longitude: 13.27177, Latitude: 38.83603

.. note:: 
    The ``convert_to_geographical`` function is also designed to handle input data in the form of arrays or pandas Series. 

When either of these data types is provided as input for the UTM coordinates, the function will return an array containing the converted geographical coordinates (latitude and longitude) for each set of UTM coordinates in the input array or Series.

.. autofunction:: seismutils.geo.convert_to_utm