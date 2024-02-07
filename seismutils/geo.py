import os
import pyproj

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from matplotlib.ticker import MultipleLocator
from typing import Tuple

def convert_to_geographical(utmx: float, utmy: float, zone: int, northern: bool, units: str, ellps: str='WGS84',datum: str='WGS84'):
    '''
    Converts UTM coordinates to geographical (longitude and latitude) coordinates.

    .. note::
        This function is capable of handling both individual floating-point numbers and bulk data in the form of arrays or pandas Series for the UTM coordinates. When provided with array or Series inputs, it returns an array containing the converted geographical coordinates (longitude and latitude) for each set of UTM coordinates.
    
    :param utmx: The UTM x coordinate (easting).
    :type utmx: float
    :param utmy: The UTM y coordinate (northing).
    :type utmy: float
    :param zone: UTM zone number.
    :type zone: int
    :param northern: True if the location is in the northern hemisphere; otherwise, False.
    :type northern: bool
    :param units: The unit of the UTM coordinates (``'m'`` for meters, ``'km'`` for kilometers).
    :type units: str
    :param ellps: The ellipsoid to use. Defaults to ``'WGS84'``.
    :type ellps: str, optional
    :param datum: The geodetic datum to use. Defaults to ``'WGS84'``.
    :type datum: str, optional
    :return: A tuple containing the longitude and latitude.
    :rtype: tuple(float, float)

    **Parameter details**

    - ``units``: This parameter allows the user to specify the units of the input UTM coordinates. Acceptable values are ``'m'`` for meters and ``'km'`` for kilometers. This flexibility lets the user work with the scale most relevant to their application or dataset.

    - ``ellps`` and ``datum:`` These parameters define the shape of the Earth (ellipsoid) and the datum for the conversion process. While the default is ``'WGS84'``, which is widely used for global mapping and satellite data, users can specify other ellipsoids or datums as needed for their specific geographic information system (GIS) applications.

    .. note::
        The conversion accuracy depends on the correctness of the input parameters, including the UTM zone and the hemisphere.

    **Usage example**

    .. code-block:: python

        import seismutils.geo as sug

        utmx, utmy = 350, 4300  # UTM coordinates
        
        lon, lat = sug.convert_to_geographical(
            lon=lon,
            lat=lat,
            zone=33,
            northern=True,
            units='km',
        )

        print(f'UTMX: {utmx}, UTMY: {utmy}')
        # Expected output: Latitude: 13.271772, Longitude: 38.836032
    '''
    # Define the geographic and UTM CRS based on the zone and hemisphere
    utm_crs = pyproj.CRS(f'+proj=utm +zone={zone} +{"+north" if northern else "+south"} +ellps={ellps} +datum={datum} +units={units}')
    geodetic_crs = pyproj.CRS('epsg:4326')
    
    # Create a Transformer object to convert between CRSs
    transformer = pyproj.Transformer.from_crs(utm_crs, geodetic_crs, always_xy=True)
    
    # Transform the coordinates
    lon, lat = transformer.transform(utmx, utmy)
    return lon, lat