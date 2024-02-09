import os
import pyproj

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from matplotlib.ticker import MultipleLocator
from typing import Tuple

def convert_to_geographical(utmx: float, utmy: float, zone: int, northern: bool, units: str, ellps: str='WGS84', datum: str='WGS84'):
    """
    Converts UTM coordinates to geographical (longitude and latitude) coordinates.

    Notes
    -----
    This function is capable of handling both individual floating-point numbers and bulk data in the form of arrays or pandas Series for the UTM coordinates. When provided with array or Series inputs, it returns an array containing the converted geographical coordinates (longitude and latitude) for each set of UTM coordinates. The conversion accuracy depends on the correctness of the input parameters, including the UTM zone and the hemisphere.

    Parameters
    ----------
    utmx : float
        The UTM x coordinate (easting).
    utmy : float
        The UTM y coordinate (northing).
    zone : int
        UTM zone number.
    northern : bool
        True if the location is in the northern hemisphere; otherwise, False.
    units : str
        The unit of the UTM coordinates. Acceptable values are 'm' for meters and 'km' for kilometers. This parameter allows the user to specify the units of the input UTM coordinates.
    ellps : str, optional
        The ellipsoid to use. Defaults to 'WGS84'. These parameters define the shape of the Earth (ellipsoid) for the conversion process.
    datum : str, optional
        The geodetic datum to use. Defaults to 'WGS84'. This parameter defines the datum for the conversion process.

    Returns
    -------
    tuple(float, float)
        A tuple containing the longitude and latitude.

    See Also
    --------
    - ``units`` : This parameter allows the user to work with the scale most relevant to their application or dataset.
    - ``ellps`` and ``datum`` : Specify other ellipsoids or datums as needed for specific geographic information system (GIS) applications.

    Examples
    --------
    .. code-block:: python

        import seismutils.geo as sug

        utmx, utmy = 350, 4300  # UTM coordinates
        
        lon, lat = sug.convert_to_geographical(
            utmx=utmx,
            utmy=utmy,
            zone=33,
            northern=True,
            units='km',
        )

        print(f'UTMX: {utmx}, UTMY: {utmy}')
        # Expected output: Latitude: 13.271772, Longitude: 38.836032
    """

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
    data.depth = np.abs(data.depth)

    # Convert earthquake data and center to UTM coordinates
    utmx, utmy = convert_to_utm(data.lon, data.lat, zone=zone, units='km', ellps='WGS84', datum='WGS84' )
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
        in_depth_range = (data['depth'] >= depth_range[0]) & (data['depth'] <= depth_range[1])
        on_section_coords =+ (utmy - center_coords[section][1]) * normal_ref[0] - (utmx - center_coords[section][0]) * normal_ref[1]
        
        close_and_in_depth = np.where((dist < event_distance_from_section) & in_depth_range & (np.abs(on_section_coords) < map_length))
        
        if plot:
            # Plot sections
            fig, ax = plt.subplots(figsize=(15, 7))
            
            ax.scatter(on_section_coords[close_and_in_depth], data.depth.iloc[close_and_in_depth], marker='.', color='black', s=0.25, alpha=0.75)
            ax.set_title(f'Section {section+1}', fontsize=14, fontweight='bold')
            
            # Format plot axis
            ax.xaxis.set_major_locator(MultipleLocator(map_length/5))
            ax.xaxis.set_major_formatter('{x:.0f}')
            ax.xaxis.set_minor_locator(MultipleLocator(map_length/10))
            
            ax.yaxis.set_major_locator(MultipleLocator(np.abs(depth_range).max()/5))
            ax.yaxis.set_major_formatter('{x:.0f}')
            ax.yaxis.set_minor_locator(MultipleLocator(np.abs(depth_range).max()/10))
            
            ax.set_xlabel('Distance along strike [km]', fontsize=12)
            ax.set_ylabel('Depth [km]', fontsize=12)
            ax.set_xlim(-map_length, map_length)
            ax.set_ylim(*depth_range)
            
            if save_figure:
                os.makedirs('./seismutils_figures', exist_ok=True)
                fig_name = os.path.join('./seismutils_figures', f'{save_name}_{section+1}.{save_extension}')
                plt.savefig(fig_name, dpi=300, bbox_inches='tight', facecolor=None)
            
            ax.invert_yaxis()
            ax.set_aspect('equal')
            ax.set_facecolor('#F0F0F0')
            ax.grid(True, alpha=0.25, linestyle=':')
            
            plt.show()
        
        # Add the events of this section to the list if return_dataframes is True
        if return_dataframes:
            # Add on section coordinates to the dataframe
            section_df = data.iloc[close_and_in_depth].copy()
            section_df['on_section_coords'] = on_section_coords[close_and_in_depth]
            
            # Append section dataframes to a list
            section_dataframes.append(section_df)     
    
    return section_dataframes

def exclude_close_timed_events(data: pd.DataFrame, window_length: float, min_interval: float):
    '''
    Filters out events in a DataFrame that occur too closely in time based on specified criteria.

    This function iterates through a sorted DataFrame of events, identifying and removing events that occur within a specified time window (``window_length``). 
    
    Additionally, if two consecutive events are separated by a time less than ``min_interval``, both events are excluded from the resulting DataFrame.

    .. note::
        The ``min_interval`` parameter must be less than ``window_length`` to avoid a ValueError. The function assumes the DataFrame contains a 'time' column with datetime or string representations of dates/times, which are converted to datetime objects for comparison.

    :param data: DataFrame containing the dataset with event timing information.
    :type data: pd.DataFrame
    :param window_length: The length of the time window in seconds within which consecutive events are considered too close and subject to exclusion.
    :type window_length: float
    :param min_interval: The minimum allowed interval in seconds between two events. If the interval between events is less than this, both events are removed.
    :type min_interval: float
    :return: A filtered DataFrame with closely timed events excluded.
    :rtype: pd.DataFrame

    **Usage example**

    .. code-block:: python

        import seismutils.geo as sug

        # Assume that data is a pd.DataFrame formatted in the following way:
        # index | lat | lon | depth | local_magnitude | momentum_magnitude | ID | time

        data_filtered = sug.exclude_close_timed_events(
            data=data,
            window_length=25,
            min_interval=5
        )
    
    This function serves an essential role in preparing datasets for waveform download, particularly when acquiring fixed-length windows around seismic events, such as those surrounding P-Wave arrivals.
    
    By specifying a window length and a minimum interval, it ensures that each downloaded trace contains data from a single event, thereby preventing the inclusion of multiple events within the same waveform trace. 
    
    This is crucial for analyses that require clear isolation of seismic signals associated with individual earthquake events, facilitating accurate study and interpretation of seismic wave characteristics and behavior.
    '''
    if min_interval >= window_length:
        raise ValueError("min_interval must be less than window_length")
    
    # Convert 'time' column to datetime
    data.time = pd.to_datetime(data.time)
    
    # Sort the DataFrame by the 'time' column
    data = data.sort_values(by='time').reset_index(drop=True)
    
    # Initialize a set to keep track of the indices of events to be removed
    indices_to_remove = set()
    
    for i in tqdm(range(len(data) - 1), desc="Analyzing events"):
        # Calculate the time difference in seconds between consecutive events
        time_diff = (data.at[i+1, 'time'] - data.at[i, 'time']).total_seconds()
        
        # If the time difference is less than the specified window, mark the event for removal
        if time_diff < window_length:
            if time_diff < min_interval:
                # If the time difference is also less than min_interval, remove both events
                indices_to_remove.add(i)
                indices_to_remove.add(i+1)
            else:
                # Otherwise, remove only the previous event
                indices_to_remove.add(i)
    
    # Remove close timed events
    data_filtered = data.drop(list(indices_to_remove)).reset_index(drop=True)
    
    return data_filtered

def select_on_map(data: pd.DataFrame, center: Tuple[float, float], size: Tuple[int, int], rotation: int, shape_type: str, zone: int, units: str, plot: bool=True, buffer_multiplier: int=10, plot_center: bool=True, save_figure: bool=False, save_name: str='selection_map', save_extension: str='jpg', return_indices: bool=False):
    '''
    Selects and optionally plots a subset of seismic events from a dataset that fall within a specified geometric shape on a map. The selection is based on the shape's center, size, and orientation, after converting geographic coordinates to UTM.

    :param data: DataFrame containing seismic event data, expected to include 'lon' and 'lat' columns among others.
    :type data: pd.DataFrame
    :param center: Geographic coordinates (longitude, latitude) of the geometric shape's center.
    :type center: Tuple[float, float]
    :param size: Dimensions of the geometric shape (radius for circles, width and height for squares).
    :type size: Tuple[int, int]
    :param rotation: The rotation angle of the shape in degrees, counter-clockwise from North.
    :type rotation: int
    :param shape_type: Specifies the geometric shape used for selection ('circle', 'square').
    :type shape_type: str
    :param zone: UTM zone for converting geographic coordinates to UTM coordinates.
    :type zone: int
    :param units: Measurement units for the UTM coordinates ('m' for meters, 'km' for kilometers).
    :type units: str
    :param plot: If True, plots the original dataset points and the selected subset on the map.
    :type plot: bool, optional
    :param buffer_multiplier: Factor to determine the plot's buffer area around the selected points.
    :type buffer_multiplier: int, optional
    :param plot_center: If True and `plot` is also True, marks the center of the selection shape on the plot.
    :type plot_center: bool, optional
    :param save_figure: If True, saves the generated plot to a file.
    :type save_figure: bool, optional
    :param save_name: Filename for the saved figure, without the extension.
    :type save_name: str, optional
    :param save_extension: File extension for the saved figure (e.g., 'jpg', 'png').
    :type save_extension: str, optional
    :param return_indices: If True, returns the indices of the selected points; otherwise, returns a subset DataFrame.
    :type return_indices: bool, optional
    :return: Depending on `return_indices`, either the indices of selected points or a DataFrame containing the selected subset.
    :rtype: List[int] or pd.DataFrame

    **Parameter details**
    
    - ``shape_type``: This parameter, combined with the ``size`` tuple, defines the selection shape. 
        - ``'circle'``: Use a tuple with equal values (e.g., ``(radius, radius)``) for a circular selection.
        - ``'square'``: A tuple with equal values specifies a square (functionally identical to a circle in terms of selection criteria due to equal dimensions).
        - For an oval or rectangle, provide a tuple with two different values (e.g., ``(width, height)``), which creates an elliptical or rectangular selection area when ``shape_type``, respectively.

    **Usage example**

    .. code-block:: python

        import seismutils.geo as sug

        # Assuming data is a pd.DataFrame containing 'lon' and 'lat' columns
        
        selection = select_on_map(
            data=data,
            center=(42.833550, 13.114270),
            shape_type='circle',
            size=(3, 3),
            rotation=0,
            zone=33,
            units='km',
            plot=True,
            buffer_multiplier=15
        )
    
    .. image:: https://imgur.com/xScpkfu.png
       :align: center
       :target: data_querying_and_selection.html#seismutils.geo.select
    
    The catalog used to demonstrate how the function works, specifically the data plotted in the image above, is derived from the `Tan et al. (2021) earthquake catalog <https://zenodo.org/records/4736089>`_.
    '''
    def rotate_point(point, center, angle):
        # Rotate a point around a given center by an angle
        angle_rad = np.deg2rad(angle)
        ox, oy = center
        px, py = point

        qx = ox + np.cos(angle_rad) * (px - ox) - np.sin(angle_rad) * (py - oy)
        qy = oy + np.sin(angle_rad) * (px - ox) + np.cos(angle_rad) * (py - oy)
        return qx, qy
    
    # Convert geographic coordinates to UTM (Placeholder logic)
    utm_x_coords, utm_y_coords = convert_to_utm(data.lon, data.lat, zone, units=units)
    utm_x_center, utm_y_center = convert_to_utm([center[0]], [center[1]], zone, units=units)
    center = (utm_x_center[0], utm_y_center[0])
    coords = (pd.Series(utm_x_coords, name='utm_x'), pd.Series(utm_y_coords, name='utm_y'))
    
    selected_indices = []
    x_coords, y_coords = coords
    for index in range(len(x_coords)):
        point = (x_coords.iloc[index], y_coords.iloc[index])
        rotated_point = rotate_point(point, center, -rotation)

        if shape_type == 'circle':
            rx, ry = size
            if ((rotated_point[0] - center[0])/rx)**2 + ((rotated_point[1] - center[1])/ry)**2 <= 1:
                selected_indices.append(index)
        
        elif shape_type == 'square':
            width, height = size
            if (center[0] - width/2 <= rotated_point[0] <= center[0] + width/2 and
                    center[1] - height/2 <= rotated_point[1] <= center[1] + height/2):
                selected_indices.append(index)

    # Plotting logic
    if plot:
        # Determina il valore maggiore tra larghezza e altezza della selezione
        max_dimension = max(size) * buffer_multiplier

        # Calcola i limiti intorno al centro utilizzando il valore maggiore
        xlim = (center[0] - max_dimension / 2, center[0] + max_dimension / 2)
        ylim = (center[1] - max_dimension / 2, center[1] + max_dimension / 2)

        fig, ax = plt.subplots(figsize=(10, 10))
        scatter_plot = ax.scatter(x_coords, y_coords, marker='.', c='grey', s=0.2, alpha=0.75)
        ax.scatter(x_coords.iloc[selected_indices], y_coords.iloc[selected_indices], marker='.', color='red', s=0.25, alpha=0.75)
        
        if plot_center:
            ax.scatter(center[0], center[1], marker='o', color='red', edgecolor='black', linewidth=0.5,  s=buffer_multiplier*10)

        # Applica i limiti calcolati
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        ax.set_aspect('equal', adjustable='box')
        plt.title(f'Selection', fontsize=14, fontweight='bold')
        plt.xlabel('UTM X [km]', fontsize=12)
        plt.ylabel('UTM Y [km]', fontsize=12)

        # Format plot axis
        ax.xaxis.set_major_locator(MultipleLocator((max(xlim)-min(xlim))/5))
        ax.xaxis.set_major_formatter('{x:.0f}')
        ax.xaxis.set_minor_locator(MultipleLocator(((max(xlim)-min(xlim))/10)))
        
        ax.yaxis.set_major_locator(MultipleLocator(((max(ylim)-min(ylim))/5)))
        ax.yaxis.set_major_formatter('{x:.0f}')
        ax.yaxis.set_minor_locator(MultipleLocator(((max(ylim)-min(ylim))/10)))
        
        ax.set_facecolor('#F0F0F0')
        plt.grid(True, alpha=0.25, linestyle=':')

        if save_figure:
            os.makedirs('./seismutils_figures', exist_ok=True)
            fig_name = os.path.join('./seismutils_figures', f'{save_name}.{save_extension}')
            plt.savefig(fig_name, dpi=300, bbox_inches='tight')
        
        plt.tight_layout()
        plt.show()

    if return_indices:
        return selected_indices
    else:
        return data.iloc[selected_indices]
    
def select_on_section(data: pd.DataFrame, center: Tuple[float, float], size: Tuple[int, int], rotation: int, shape_type: str, plot: bool=True, plot_center: bool=True, save_figure: bool=False, save_name: str='selection_section', save_extension: str='jpg', return_indices: bool=False):
    '''
    Selects and optionally plots a subset of seismic events from a dataset that fall within a specified geometric shape on a cross-section. The selection is based on the shape's center, size, and orientation, providing a focused analysis on specific areas of interest within the seismic data.

    :param data: DataFrame containing seismic event data, expected to include 'on_section_coords' and 'depth' columns among others.
    :type data: pd.DataFrame
    :param center: The (x, y) coordinates representing the center of the geometric shape on the section.
    :type center: Tuple[float, float]
    :param size: Dimensions of the geometric shape (radius for circles, width and height for squares).
    :type size: Tuple[int, int]
    :param rotation: The rotation angle of the shape in degrees, counter-clockwise from the horizontal axis of the section.
    :type rotation: int
    :param shape_type: Specifies the geometric shape used for selection ('circle', 'square').
    :type shape_type: str
    :param plot: If True, plots the original dataset points and the selected subset on the cross-section.
    :type plot: bool, optional
    :param plot_center: If True and `plot` is also True, marks the center of the selection shape on the plot.
    :type plot_center: bool, optional
    :param save_figure: If True, saves the generated plot to a file.
    :type save_figure: bool, optional
    :param save_name: Filename for the saved figure, without the extension.
    :type save_name: str, optional
    :param save_extension: File extension for the saved figure (e.g., 'jpg', 'png').
    :type save_extension: str, optional
    :param return_indices: If True, returns the indices of the selected points; otherwise, returns a subset DataFrame.
    :type return_indices: bool, optional
    :return: Depending on `return_indices`, either the indices of selected points or a DataFrame containing the selected subset.
    :rtype: List[int] or pd.DataFrame

    **Parameter details**
    
    - ``shape_type``: This parameter, combined with the ``size`` tuple, defines the selection shape. 
        - ``'circle'``: Use a tuple with equal values (e.g., ``(radius, radius)``) for a circular selection.
        - ``'square'``: A tuple with equal values specifies a square (functionally identical to a circle in terms of selection criteria due to equal dimensions).
        - For an oval or rectangle, provide a tuple with two different values (e.g., ``(width, height)``), which creates an elliptical or rectangular selection area when ``shape_type``, respectively.
    
    **Usage example**

    .. code-block:: python

        import seismutils.geo as sug

        # Assuming data is a pd.DataFrame containing 'on_section_coords' and 'depth' columns
        # This dataframes can be obtained in output from the cross_sections() function
        
        selection = sug.select_on_section(
            data=subset,
            center=(10.2, 7.2),
            size=(0.6, 1.1),
            rotation=160,
            shape_type='circle',
            plot=True,
            plot_center=True
        )
        
    .. image:: https://imgur.com/Q1OL1o0.png
       :align: center
       :target: data_querying_and_selection.html#seismutils.geo.select
    
    The catalog used to demonstrate how the function works, specifically the data plotted in the image above, is derived from the `Tan et al. (2021) earthquake catalog <https://zenodo.org/records/4736089>`_.
    '''
    def rotate_point(point, center, angle):
        # Rotate a point around a given center by an angle
        angle_rad = np.deg2rad(angle)
        ox, oy = center
        px, py = point

        qx = ox + np.cos(angle_rad) * (px - ox) - np.sin(angle_rad) * (py - oy)
        qy = oy + np.sin(angle_rad) * (px - ox) + np.cos(angle_rad) * (py - oy)
        return qx, qy
    
    coords = (data.on_section_coords, np.abs(data.depth))
    
    selected_indices = []
    x_coords, y_coords = coords
    for index in range(len(x_coords)):
        point = (x_coords.iloc[index], y_coords.iloc[index])
        rotated_point = rotate_point(point, center, -rotation)
        
        if shape_type == 'circle':
            rx, ry = size
            if ((rotated_point[0] - center[0])/rx)**2 + ((rotated_point[1] - center[1])/ry)**2 <= 1:
                selected_indices.append(index)
        
        elif shape_type == 'square':
            width, height = size
            if (center[0] - width/2 <= rotated_point[0] <= center[0] + width/2 and
                    center[1] - height/2 <= rotated_point[1] <= center[1] + height/2):
                selected_indices.append(index)

    # Plotting logic
    if plot:
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.scatter(x_coords, y_coords, marker='.', color='grey', s=0.25, alpha=0.75)
        ax.scatter(x_coords.iloc[selected_indices], y_coords.iloc[selected_indices], marker='.', color='red', s=0.25, alpha=0.75)
        
        if plot_center:
            ax.scatter(center[0], center[1], marker='o', color='red', edgecolor='black', linewidth=0.5,  s=150)
        
        plt.title(f'Selection', fontsize=14, fontweight='bold')
        plt.xlabel('Distance from center [km]', fontsize=12)
        plt.ylabel('Depth [km]', fontsize=12)
        plt.xlim(round(data['on_section_coords'].min()), round(data['on_section_coords'].max()))
        plt.ylim(round(data['depth'].min()), round(data['depth'].max()))

        # Format plot axis
        ax.xaxis.set_major_locator(MultipleLocator(round(x_coords.max())/5))
        ax.xaxis.set_major_formatter('{x:.0f}')
        ax.xaxis.set_minor_locator(MultipleLocator(round(x_coords.max())/10))
        
        ax.yaxis.set_major_locator(MultipleLocator(round(y_coords.max())/5))
        ax.yaxis.set_major_formatter('{x:.0f}')
        ax.yaxis.set_minor_locator(MultipleLocator(round(y_coords.max())/10))
        
        ax.invert_yaxis()
        ax.set_facecolor('#F0F0F0')
        ax.set_aspect('equal', adjustable='box')
        plt.grid(True, alpha=0.25, linestyle=':')

        if save_figure:
            os.makedirs('./seismutils_figures', exist_ok=True)
            fig_name = os.path.join('./seismutils_figures', f'{save_name}.{save_extension}')
            plt.savefig(fig_name, dpi=300, bbox_inches='tight')
        
        plt.show()

    if return_indices:
        return selected_indices
    else:
        return data.iloc[selected_indices]