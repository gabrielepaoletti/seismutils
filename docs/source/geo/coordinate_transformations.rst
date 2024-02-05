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
    # Output: Longitude: 13.271772, Latitude: 38.836032

.. note:: 
    The ``convert_to_geographical`` function is also designed to handle input data in the form of arrays or pandas Series. 

    When either of these data types is provided as input for the UTM coordinates, the function will return an array containing the converted geographical coordinates (latitude and longitude) for each set of UTM coordinates in the input array or Series.

.. autofunction:: seismutils.geo.convert_to_utm

    Example usage
^^^^^^^^^^^^^

To use the ``convert_to_utm`` function, you must import it from the ``seismutils.geo`` sub-module. 

Below is an example of converting geographical coordinates (latitude and longitude) from zone 33N, with a latitude of 38.83603 and a longitude of 13.27177, to UTM coordinates. These coordinates are assumed to be based on the WGS84 ellipsoid and datum, and the output UTM coordinates are expressed in kilometers (the ``units`` parameter).

.. code-block:: python

    import seismutils.geo as sug

    import seismutils.seismutils.geo as sug

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
    The ``convert_to_utm`` function is also designed to handle input data in the form of arrays or pandas Series. 

    When either of these data types is provided as input for the geographical coordinates, the function will return an array containing the converted UTM coordinates for each set of geographical coordinates in the input array or Series.