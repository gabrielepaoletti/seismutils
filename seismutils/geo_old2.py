import os
import pyproj

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import List, Tuple
from matplotlib.ticker import MultipleLocator

def convert_to_geographical(utmx: float, utmy: float, zone: int, northern: bool, units: str, ellps: str='WGS84', datum: str='WGS84'):
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

def convert_to_utm(lon: float, lat: float, zone: int, units: str, ellps: str='WGS84', datum: str='WGS84'):
    '''
    Converts geographical (longitude and latitude) coordinates to UTM coordinates.

    .. note::
        This function is designed to handle both individual floating-point numbers and bulk data in the form of arrays or pandas Series for the geographical coordinates. When provided with array or Series inputs, it returns an array containing the converted UTM coordinates (easting and northing) for each set of geographical coordinates.

    :param lon: Longitude value(s).
    :type lon: float
    :param lat: Latitude value(s).
    :type lat: float
    :param zone: UTM zone number.
    :type zone: int
    :param units: The unit of the output UTM coordinates (``'m'`` for meters, ``'km'`` for kilometers).
    :type units: str
    :param ellps: The ellipsoid to use. Defaults to ``'WGS84'``.
    :type ellps: str, optional
    :param datum: The geodetic datum to use. Defaults to ``'WGS84'``.
    :type datum: str, optional
    :return: A tuple containing the UTM x (easting) and UTM y (northing) coordinates.
    :rtype: tuple(float, float)

    **Parameter details**

    - ``units``: Allows specifying the units for the output UTM coordinates. Supporting both meters ``'m'`` and kilometers ``'km'``, this parameter provides flexibility for various application scales.

    - ``ellps`` and ``datum``: Define the Earth's shape (ellipsoid) and the reference datum for the conversion. The default ``'WGS84'`` is commonly used, but alternative specifications can be used for different GIS needs.

    .. note::
        Conversion accuracy is influenced by the accuracy of the input longitude and latitude, as well as the chosen UTM zone,.

    **Usage example**

    .. code-block:: python

        import seismutils.geo as sug

        lon, lat = 13.271772, 38.836032  # Geographical coordinates

        utmx, utmy = sug.convert_to_utm(
            lon=lon,
            lat=lat,
            zone=33,
            units='km'
        )

        print(f'UTM X: {utmx}, UTM Y: {utmy}')
        # Expected output: UTM X: 350, UTM Y: 4300
    '''
    # Create a pyproj Proj object for UTM conversion using the given zone and ellipsoid.
    utm_converter = pyproj.Proj(proj='utm', zone=zone, units=units, ellps=ellps, datum=datum)

    # Transform the coordinates
    utmx, utmy = utm_converter(np.array(lon), np.array(lat))
    return utmx, utmy

def cross_sections(data: pd.DataFrame, center: Tuple[float, float], num_sections: Tuple[int, int], event_distance_from_section: int, strike: int, map_length: int, depth_range: Tuple[float, float], zone: int,section_distance: int=1, plot: bool=False, save_figure: bool=False, save_name: str='section', save_extension: str='jpg', return_dataframes: bool=True):
    '''
    Analyzes earthquake data relative to a geological structure's orientation, creating cross sections perpendicular to strike that showcase the spatial distribution of seismic events. Optionally plots these sections for visual inspection.

    This function segments the input earthquake data into cross sections based on their proximity and alignment with a specified geological strike. It can generate a series of parallel cross sections, allowing for a comprehensive analysis of seismic activity around a central point of interest.

    :param data: DataFrame containing earthquake event data, with essential columns like 'lon', 'lat' and 'depth'
    :type data: pd.DataFrame
    :param center: Coordinates (longitude, latitude) of the central point for the primary cross section.
    :type center: Tuple[float, float]
    :param num_sections: Tuple specifying the number of sections to the left and right of the primary section to analyze and plot.
    :type num_sections: Tuple[int, int]
    :param event_distance_from_section: Maximum distance from a section within which an earthquake event is considered for inclusion, in kilometers.
    :type event_distance_from_section: int
    :param strike: Geological strike direction in degrees from North, determining the orientation of cross sections.
    :type strike: int
    :param map_length: Length of the cross section lines in kilometers.
    :type map_length: int
    :param depth_range: Tuple specifying the minimum and maximum depths of earthquake events to include.
    :type depth_range: Tuple[float, float]
    :param zone: UTM zone for mapping coordinates.
    :type zone: int
    :param section_distance: Distance between adjacent cross sections, in kilometers. Default is 1 km.
    :type section_distance: int
    :param plot: If True, generates plots for each cross section with earthquake events.
    :type plot: bool, optional
    :param save_figure: If True, saves the generated plots in a directory.
    :type save_figure: bool, optional
    :param save_name: Name under which the figure will be saved. Default ``'section'``
    :type save_name: str, optional
    :param save_extension: Extension with which the image will be saved. Default ``'jpg'``
    :type save_extension: str, optional
    :param return_dataframes: If True, returns a list of DataFrames for each section. Each DataFrame contains earthquake events that fall within the section.
    :type return_dataframes: bool, optional
    :return: List of DataFrames corresponding to each cross section, containing relevant earthquake event data if 'return_dataframes' is True. Otherwise, returns None.
    :rtype: List[pd.DataFrame] or None
    
    **Usage example**

    .. code-block:: python

        import seismutils.geo as sug

        # Assume that data is a pd.DataFrame formatted in the following way:
        # index | lat | lon | depth | local_magnitude | momentum_magnitude | ID | time
        
        subset = sug.cross_sections(
            data=data,
            center=(13.12131, 42.83603),
            num_sections=(0,0),
            event_distance_from_section=3,
            strike=155,
            map_length=15,
            depth_range=(-10, 0),
            zone=33,
            plot=True
        )

    .. image:: https://imgur.com/0cufUSo.png
       :align: center
       :target: seismic_visualization.html#seismutils.geo.cross_section
    
    The catalog used to demonstrate how the function works, specifically the data plotted in the image above, is derived from the `Tan et al. (2021) earthquake catalog <https://zenodo.org/records/4736089>`_.

    .. note::
        Due to the complexity of using this function, it is recommended for users to consult a full tutorial on how to effectively plot cross sections. This tutorial will guide through the specifics of data preparation, parameter tuning, and interpretation of the results.
    '''

    # Function to calculate the distance of a point from a plane
    def distance_point_from_plane(x, y, z, normal, origin):
        d = -normal[0] * origin[0] - normal[1] * origin[1] - normal[2] * origin[2]
        dist = np.abs(normal[0] * x + normal[1] * y + normal[2] * z + d)
        dist = dist / np.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)
        return dist
    
    def section_center_positions(center_x, center_y, section_centers, strike):
        angle_rad = np.pi / 2 - np.radians(strike)
        return center_x + section_centers * np.cos(angle_rad), center_y + section_centers * np.sin(angle_rad)
    
    # Make sure all the depths are positive values
    data['depth'] = np.abs(data['depth'])

    # Convert earthquake data and center to UTM coordinates
    utmx, utmy = convert_to_utm(data['lon'], data['lat'], zone=zone, units='km', ellps='WGS84', datum='WGS84' )
    center_utmx, center_utmy = convert_to_utm(center[0], center[1], zone=zone, units='km', ellps='WGS84', datum='WGS84')
    
    # Set normal vector for the section based on the provided orientation
    normal_tostrike = strike - 90
    normal_ref = [np.cos(normal_tostrike * np.pi / 180), -np.sin(normal_tostrike * np.pi / 180), 0]
    
    # Calculate center coordinates for each section
    centers_distro = np.arange(-num_sections[0]*section_distance, num_sections[1]*section_distance+1, section_distance)
    centers_depths = -10 * np.ones(len(centers_distro))
    center_xs, center_ys = section_center_positions(center_utmx, center_utmy, centers_distro, strike)
    center_coords = np.array([center_xs, center_ys, centers_depths]).T
    
    # List to store dataframes for each section
    section_dataframes = []
    
    for section in range(len(centers_distro)):
        
        # Calculate distance of events from each section plane and filter by depth
        dist = distance_point_from_plane(utmx, utmy, -data['depth'], normal_ref, center_coords[section])
        in_depth_range = (data['depth'] >= -depth_range[1]) & (data['depth'] <= -depth_range[0])
        on_section_coords =+ (utmy - center_coords[section][1]) * normal_ref[0] - (utmx - center_coords[section][0]) * normal_ref[1]
        
        close_and_in_depth = np.where((dist < event_distance_from_section) & in_depth_range & (np.abs(on_section_coords) < map_length))
        
        if plot:
            # Plot sections
            fig = plt.figure(figsize=(15, 7))
            
            plt.scatter(on_section_coords[close_and_in_depth], -data.depth.iloc[close_and_in_depth], marker='.', color='black', s=0.25, alpha=0.75)
            plt.title(f'Section {section+1}', fontsize=14, fontweight='bold')
            
            # Format plot axis
            plt.gca().xaxis.set_major_locator(MultipleLocator(map_length/5))
            plt.gca().xaxis.set_major_formatter('{x:.0f}')
            plt.gca().xaxis.set_minor_locator(MultipleLocator(map_length/10))
            
            plt.gca().yaxis.set_major_locator(MultipleLocator(np.abs(depth_range).max()/5))
            plt.gca().yaxis.set_major_formatter('{x:.0f}')
            plt.gca().yaxis.set_minor_locator(MultipleLocator(np.abs(depth_range).max()/10))
            
            plt.gca().set_aspect('equal')
            plt.xlabel('Distance along strike [km]', fontsize=12)
            plt.ylabel('Depth [km]', fontsize=12)
            plt.xlim(-map_length, map_length)
            plt.ylim(*depth_range)
            
            if save_figure:
                os.makedirs('./seismutils_figures', exist_ok=True)
                fig_name = os.path.join('./seismutils_figures', f'{save_name}_{section+1}.{save_extension}')
                plt.savefig(fig_name, dpi=300, bbox_inches='tight', facecolor=None)
            
            plt.show()
        
        # Add the events of this section to the list if return_dataframes is True
        if return_dataframes:
            # Add on section coordinates to the dataframe
            section_df = data.iloc[close_and_in_depth].copy().reset_index(drop=True)
            section_df['on_section_coords'] = on_section_coords[close_and_in_depth]
            
            # Append section dataframes to a list
            section_dataframes.append(section_df)     
    
    return section_dataframes

def select(data: pd.DataFrame, coords: Tuple[pd.Series, pd.Series], center: Tuple[float, float], size: Tuple[int, int], rotation: int, shape_type: str, plot: bool=False, save_figure: bool=False, save_name: str='selection', save_extension: str='jpg', return_indices: bool=False):
    '''
    Selects a subset of data points that fall within a specified geometric shape, which is defined by its center, size, and rotation. This function can handle shapes like circles, ovals, and rectangles. It offers options to plot the selected points and return the subset as either indices or a DataFrame.

    This utility is particularly useful in spatial data analysis, where isolating data points within specific geometric boundaries can provide insights into patterns or distributions relative to a defined area of interest.

    :param data: DataFrame containing the dataset from which to select points.
    :type data: pd.DataFrame
    :param coords: A pair of Series representing the x (longitude) and y (latitude) coordinates of the data points.
    :type coords: Tuple[pd.Series, pd.Series]
    :param center: The (x, y) coordinates representing the center of the geometric shape.
    :type center: Tuple[float, float]
    :param size: Dimensions of the geometric shape (width, height) for rectangles and ovals, or radius for circles.
    :type size: Tuple[int, int]
    :param rotation: The rotation angle of the shape in degrees, counter-clockwise from the x-axis.
    :type rotation: int
    :param shape_type: The type of geometric shape ('circle', 'oval', 'rectangle').
    :type shape_type: str
    :param plot: If True, plots the original dataset points and the selected subset.
    :type plot: bool, optional
    :param return_indices: If True, returns the indices of the selected points; otherwise, returns a subset DataFrame.
    :type return_indices: bool, optional
    :param save_figure: If True, saves the generated plots in a directory.
    :type save_figure: bool, optional
    :param save_name: Name under which the figure will be saved. Default ``'selection'``
    :type save_name: str, optional
    :param save_extension: Extension with which the image will be saved. Default ``'jpg'``
    :type save_extension: str, optional
    :return: Indices of selected points or a DataFrame containing the selected subset, based on the return_indices parameter.
    :rtype: List[int] or pd.DataFrame

    **Parameter details**

    - ``shape_type``: This parameter defines the geometric shape for data selection, with the following options:
        - ``'circle'``: Selects data within a circular area centered at the specified coordinates. The ``size`` parameter represents the radius of the circle.
        - ``'oval'``: Selects data within an elliptical (oval) area. Requires the ``size`` parameter to specify both the major (width) and minor (height) axes of the ellipse.
        - ``'rectangle'``: Selects data within a rectangular area. The ``size`` parameter should define the rectangle's width and height.

    The choice of shape affects how the ``size`` and ``rotation`` parameters are interpreted, allowing for flexible data selection strategies based on geometric considerations.
    
    **Usage example**

    In the following example, we first use the :func:`cross_sections` function to isolate a subset of earthquake events based on their spatial relationship to a defined geological structure, preparing a targeted subset from which we then select a specific cluster of events visible on a section.
    
    .. code-block:: python

        import seismutils.geo as sug

        # Assume that data is a pd.DataFrame formatted in the following way:
        # index | lat | lon | depth | local_magnitude | momentum_magnitude | ID | time

        # Creating a subset from a cross-section
        subset = sug.cross_sections(
            data=data,
            center=(13.12131, 42.83603),
            num_sections=(0,0),
            event_distance_from_section=3,
            strike=155,
            map_length=15,
            depth_range=(-10, 0),
            zone=33,
            plot=False
        )
        
        # Select the the cluster we are interested in
        selection = sug.select(
            data=subset[0], # Using [0] because sug.cross_sections() returns List[pd.DataFrame]
            coords=(data['lon'], data['depth']),
            center=(10.2, 7.2),
            size=(0.6, 1.1),
            rotation=160,
            shape_type='oval',
            plot=True
        )

    .. image:: https://imgur.com/GmxrUlA.png
       :align: center
       :target: data_querying_and_selection.html#seismutils.geo.select
    
    The catalog used to demonstrate how the function works, specifically the data plotted in the image above, is derived from the `Tan et al. (2021) earthquake catalog <https://zenodo.org/records/4736089>`_.
    
    .. note::
        Due to the complexity of using this function, especially in the context of spatial data analysis and geometric selection, it is highly recommended for users to consult a full tutorial. This guide would cover the specifics of data preparation, parameter tuning, and result interpretation, ensuring users can effectively apply this function to their datasets.
    '''
    def rotate_point(point, center, angle):
        angle_rad = np.deg2rad(angle)
        ox, oy = center
        px, py = point

        qx = ox + np.cos(angle_rad) * (px - ox) - np.sin(angle_rad) * (py - oy)
        qy = oy + np.sin(angle_rad) * (px - ox) + np.cos(angle_rad) * (py - oy)
        return qx, qy

    selected_indices = []
    x_coords, y_coords = coords
    for index in range(len(x_coords)):
        point = (x_coords[index], y_coords[index])
        rotated_point = rotate_point(point, center, -rotation)

        if shape_type == 'circle':
            radius = size
            if np.hypot(rotated_point[0] - center[0], rotated_point[1] - center[1]) <= radius:
                selected_indices.append(index)
        
        elif shape_type == 'oval':
            rx, ry = size
            if ((rotated_point[0] - center[0])/rx)**2 + ((rotated_point[1] - center[1])/ry)**2 <= 1:
                selected_indices.append(index)
        
        elif shape_type == 'rectangle':
            width, height = size
            if (center[0] - width/2 <= rotated_point[0] <= center[0] + width/2 and
                    center[1] - height/2 <= rotated_point[1] <= center[1] + height/2):
                selected_indices.append(index)

    if plot:
        fig = plt.figure(figsize=(15, 7))
        plt.title(f'Selection', fontsize=14, fontweight='bold')
        
        plt.scatter(coords[0], coords[1] if coords[1].name != 'depth' else -coords[1], marker='.', color='grey', s=0.25, alpha=0.75)
        plt.scatter(coords[0].iloc[selected_indices], coords[1].iloc[selected_indices] if coords[1].name != 'depth' else -coords[1].iloc[selected_indices], marker='.', color='blue', s=0.25, alpha=0.75)
      
        # Format plot axis
        plt.gca().xaxis.set_major_locator(MultipleLocator(round(np.abs(coords[0]).max())/5))
        plt.gca().xaxis.set_major_formatter('{x:.0f}')
        plt.gca().xaxis.set_minor_locator(MultipleLocator(round(np.abs(coords[0]).max())/10))
        
        plt.gca().yaxis.set_major_locator(MultipleLocator(round(np.abs(coords[1]).max())/5))
        plt.gca().yaxis.set_major_formatter('{x:.0f}')
        plt.gca().yaxis.set_minor_locator(MultipleLocator(round(np.abs(coords[1]).max())/10))
      
        plt.gca().set_aspect('equal')
        plt.xlabel(f'{coords[0].name}', fontsize=12)
        plt.ylabel(f'{coords[1].name}', fontsize=12)
        plt.xlim(round(coords[0].min()), round(coords[0].max()))
        plt.ylim(round(coords[1].max()) if coords[1].name != 'depth' else -round(coords[1].max()), round(coords[1].min()) if coords[1].name != 'depth' else -round(coords[1].min()))
        
        if save_figure:
            os.makedirs('./seismutils_figures', exist_ok=True)
            fig_name = os.path.join('./seismutils_figures', f'{save_name}.{save_extension}')
            plt.savefig(fig_name, dpi=300, bbox_inches='tight', facecolor=None)
        
        plt.show()
    
    if return_indices:  
        return selected_indices
    else:
        return data.iloc[selected_indices].reset_index(drop=True)