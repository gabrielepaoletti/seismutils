import os
import pyproj

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from matplotlib.ticker import MultipleLocator
from typing import List, Tuple

def convert_to_geographical(
    utmx: float | List[float] | np.ndarray | pd.Series, 
    utmy: float | List[float] | np.ndarray | pd.Series, 
    zone: int, 
    northern: bool,
    units: str,
    ellps: str='WGS84',
    datum: str='WGS84'
):
    '''
    Converts UTM coordinates to geographical (longitude and latitude) coordinates.

    .. note::
        This function is capable of handling both individual floating-point numbers and bulk data in the form of lists, arrays, or pandas Series for the UTM coordinates.

    Parameters
    ----------
    utmx : float or list or np.ndarray or pd.Series
        Represents the UTM x coordinate(s) (easting), indicating the eastward-measured distance from the meridian of the UTM zone.

    utmy : float or list or np.ndarray or pd.Series
        Represents the UTM y coordinate(s) (northing), indicating the northward-measured distance from the equator.

    zone : int
        The UTM zone number that the coordinates fall into, which ranges from 1 to 60. This number helps identify the specific longitudinal band used for the UTM projection.

    northern : bool
        A boolean flag that specifies the hemisphere of the coordinates. Set to True if the coordinates are in the Northern Hemisphere, and False if they are in the Southern Hemisphere.

    units : str
        Specifies the units of the input UTM coordinates. Accepted values are 'm' for meters and 'km' for kilometers. This ensures that the conversion process accurately interprets the scale of the input coordinates.

    ellps : str, optional
        The ellipsoid model used for the Earth's shape in the conversion process, with 'WGS84' as the default. The choice of ellipsoid affects the accuracy of the conversion, as it defines the geometric properties of the Earth model used.

    datum : str, optional
        The geodetic datum that specifies the coordinate system and ellipsoidal model of the Earth, for calculating geographical coordinates. The default is 'WGS84', which is a widely used global datum that provides a good balance of accuracy for global applications.

    Returns
    -------
    lon : float or list or np.ndarray or pd.Series
        The longitude value(s) obtained from the conversion, presented in the same format as the input coordinates.

    lat : float or list or np.ndarray or pd.Series
        The latitude value(s) obtained from the conversion, presented in the same format as the input coordinates.

    See Also
    --------
    convert_to_utm : Converts geographical (longitude and latitude) coordinates to UTM coordinates.

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
        >>> 'Latitude: 13.271772, Longitude: 38.836032'
    '''
    # Define the geographic and UTM CRS based on the zone and hemisphere
    utm_crs = pyproj.CRS(f'+proj=utm +zone={zone} +{"+north" if northern else "+south"} +ellps={ellps} +datum={datum} +units={units}')
    geodetic_crs = pyproj.CRS('epsg:4326')
    
    # Create a Transformer object to convert between CRSs
    transformer = pyproj.Transformer.from_crs(utm_crs, geodetic_crs, always_xy=True)
    
    # Transform the coordinates
    lon, lat = transformer.transform(utmx, utmy)
    return lon, lat

def convert_to_utm(
    lon: float | List[float] | np.ndarray | pd.Series,
    lat: float | List[float] | np.ndarray | pd.Series,
    zone: int,
    units: str,
    ellps: str='WGS84',
    datum: str='WGS84'
):
    '''
    Converts geographical (longitude and latitude) coordinates to UTM coordinates.

    .. note::
        This function is capable of handling both individual floating-point numbers and bulk data in the form of lists, arrays or pandas Series for the UTM coordinates.

    Parameters
    ----------
    lon : float or list or np.ndarray or pd.Series
        The longitude value(s).
    
    lat : float or list or np.ndarray or pd.Series
        The latitude value(s)
        
    zone : int
        The UTM zone number that the coordinates fall into, which ranges from 1 to 60. This number helps identify the specific longitudinal band used for the UTM projection.


    units : str
        Specifies the units of the input UTM coordinates. Accepted values are 'm' for meters and 'km' for kilometers. This ensures that the conversion process accurately interprets the scale of the input coordinates.

    ellps : str, optional
        The ellipsoid model used for the Earth's shape in the conversion process, with 'WGS84' as the default. The choice of ellipsoid affects the accuracy of the conversion, as it defines the geometric properties of the Earth model used.

    datum : str, optional
        The geodetic datum that specifies the coordinate system and ellipsoidal model of the Earth, for calculating geographical coordinates. The default is 'WGS84', which is a widely used global datum that provides a good balance of accuracy for global applications.

    Returns
    -------
    utmx : float or list or np.ndarray or pd.Series
        The resulting UTM x coordinates (easting), indicating the distance eastward from the central meridian of the UTM zone, presented in the same format as the input coordinates.
        
    utmy : float or list or np.ndarray or pd.Series
        The resulting UTM y coordinates (northing), indicating the distance northward from the equator, presented in the same format as the input coordinates.

    See Also
    --------
    convert_to_geographical : Converts UTM coordinates to geographical (longitude and latitude) coordinates.

    Examples
    --------
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
        >>> 'UTM X: 350, UTM Y: 4300'
    '''
    # Create a pyproj Proj object for UTM conversion using the given zone and ellipsoid.
    utm_converter = pyproj.Proj(proj='utm', zone=zone, units=units, ellps=ellps, datum=datum)

    # Transform the coordinates
    utmx, utmy = utm_converter(np.array(lon), np.array(lat))
    return utmx, utmy

def cross_sections(
    data: pd.DataFrame,
    center: Tuple[float, float],
    num_sections: Tuple[int, int],
    event_distance_from_section: int,
    strike: int,
    map_length: int,
    depth_range: Tuple[float, float],
    zone: int,section_distance: int=1,
    plot: bool=True,
    save_figure: bool=False,
    save_name: str='section',
    save_extension: str='jpg',
    return_dataframes: bool=False
):
    '''
    Analyze earthquake data to generate parallel cross sections perpendicular to a given strike, allowing for a comprehensive analysis of seismic activity around a central point of interest.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame including data on earthquake events, specifically containing the columns: 'lon' (longitude), 'lat' (latitude), and 'depth' (below the surface in kilometers).

    center : tuple(float, float)
        Represents the geographical coordinates of the central point for the main cross section, provided as a tuple of (longitude, latitude).

    num_sections : tuple(float, float)
        A tuple indicating the number of cross sections to be generated to the left and right of the central (main) cross section.

    event_distance_from_section : int
        Defines the maximum allowable distance in kilometers from a cross section within which an earthquake event is considered relevant and included in the analysis.

    strike : int
        The geological strike direction, specified in degrees from North. Cross sections are generated perpendicular to this direction.

    map_length : int
        Specifies the half-length of the cross section lines in kilometers. The total length of the cross section is double this value, extending equally from the center point in both directions.

    depth_range : tuple(float, float)
        A range specifying the minimum and maximum depths (in kilometers) for earthquake events to be included. This parameter filters the events by depth to focus the analysis.

    zone : int
        The UTM zone number that the coordinates fall into, which ranges from 1 to 60. This number helps identify the specific longitudinal band used for the UTM projection. This is crucial for converting geographic coordinates (longitude, latitude) to UTM coordinates, facilitating distance measurements in kilometers.

    section_distance : int, optional
        The spacing between adjacent cross sections in kilometers, with a default value of 1 km. This determines how closely the sections are laid out across the area of interest.

    plot : bool, optional
        Determines whether to generate visual plots for each cross section, showing the distribution of earthquake events. The default setting is True, enabling the visualization of results.

    save_figure : bool, optional
        If set to True, the function saves the generated plots using the provided base name and file extension. The default is False.

    save_name : str, optional
        The base name used for saving figures when `save_figure` is True. It serves as the prefix for file names. The default base name is 'section'.

    save_extension : str, optional
        The file extension to use when saving figures, such as 'jpg', 'png', etc... The default extension is 'jpg'.

    return_dataframes : bool, optional
        If True, returns a list of DataFrames, each representing a different cross section with included earthquake events, otherwise, it returns None. The default is False.

    Returns
    -------
    List[pd.DataFrame] or None
        If ``return_dataframes`` is True, returns a list of DataFrames, with each DataFrame corresponding to a specific cross section containing the included earthquake events. Otherwise, returns None.
        
    See Also
    --------
    select_on_section : Enables the selection of seismic events within a specified geometric shape on a cross-section derived from earthquake catalog data.

    Examples
    --------
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
            depth_range=(0, 10),
            zone=33,
            plot=True
        )

    .. image:: https://i.imgur.com/wzffhZj.png
       :align: center
       :target: seismic_visualization.html#seismutils.geo.cross_section

    The catalog used to demonstrate how the function works, specifically the data plotted in the image above, is derived from the `Tan et al. (2021) earthquake catalog <https://zenodo.org/records/4736089>`_.

    .. warning::
        The use of this function could be very complex since there are many parameters to manage at the same time. Make sure you read the full tutorial before using it.
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
            fig, ax = plt.subplots(figsize=(12, 6))
            
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
            
            ax.invert_yaxis()
            ax.set_aspect('equal')
            ax.set_facecolor('#F0F0F0')
            ax.grid(True, alpha=0.25, linestyle=':')
            
            if save_figure:
                os.makedirs('./seismutils_figures', exist_ok=True)
                fig_name = os.path.join('./seismutils_figures', f'{save_name}_{section+1}.{save_extension}')
                plt.savefig(fig_name, dpi=300, bbox_inches='tight', facecolor=None)
            
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

def select_on_map(
    data: pd.DataFrame,
    center: Tuple[float, float],
    size: Tuple[int, int],
    rotation: int,
    shape_type: str,
    zone: int,
    units: str,
    plot: bool=True,
    buffer_multiplier: int=10,
    plot_center: bool=True,
    save_figure: bool=False,
    save_name: str='selection_map',
    save_extension: str='jpg',
    return_indices: bool=False
):
    '''
    Given an earthquake catalog containing latitude and longitude data, this function facilitates the selection of a subset of events falling within a specified geometric shape centered at a given point.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame containing seismic event data, with mandatory ``'lon'`` (longitude) and ``'lat'`` (latitude) columns.

    center : tuple(float, float)
        The geographic coordinates (longitude, latitude) defining the center of the selection shape.

    size : tuple(int, int)
        The dimensions of the selection shape, specified as a tuple. For a circle, both elements of the tuple should be equal, representing the radius (otherwise oval). For a square, the elements define the length of the sides (if different, rectangle). The units are determined by the ``units`` parameter.

    rotation : int
        The rotation angle of the selection shape, in degrees, measured counter-clockwise from North.

    shape_type : str
        The type of geometric shape used for the selection. Valid options are 'circle' or 'square', which define the form of the area within which seismic events will be selected.

    zone : int
        The UTM zone number that the coordinates fall into, which ranges from 1 to 60. This number helps identify the specific longitudinal band used for the UTM projection.

    units : str
        Specifies the units of the input UTM coordinates. Accepted values are 'm' for meters and 'km' for kilometers.

    plot : bool, optional
        If True, generates a plot illustrating the original dataset points alongside the subset of points that fall within the selected shape. Defaults to True, enabling visual verification of the selection.

    buffer_multiplier : int, optional
        A factor that enlarges the buffer area around the selection shape in the plot, enhancing visibility and context. The default value is 10.

    plot_center : bool, optional
        When True (and ``plot`` is also True), marks the geometric center of the selection shape on the plot, aiding in the visualization of the selection's focal point. Defaults to True.

    save_figure : bool, optional
        If set to True, the function saves the generated plots using the provided base name and file extension. The default is False.

    save_name : str, optional
        The base name used for saving figures when `save_figure` is True. It serves as the prefix for file names. The default base name is 'section'.

    save_extension : str, optional
        The file extension to use when saving figures, such as 'jpg', 'png', etc... The default extension is 'jpg'.

    return_indices : bool, optional
        If True, the function returns the indices of the points selected within the specified shape. If False, it returns a DataFrame containing the subset of selected data points. Defaults to False.

    Returns
    -------
    List[int] or pd.DataFrame
        Depending on the ``return_indices`` parameter, this function returns either a list of indices corresponding to the selected points or a DataFrame containing the subset of selected data points.

    See Also
    --------
    select_on_section : Enables the selection of seismic events within a specified geometric shape on a cross-section derived from earthquake catalog data.

    Example
    -------
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
        :target: seismic_visualization.html#seismutils.geo.cross_section

    The catalog used to demonstrate how the function works, specifically the data plotted in the image above, is derived from the `Tan et al. (2021) earthquake catalog <https://zenodo.org/records/4736089>`_.

    .. warning::
        The use of this function could be very complex since there are many parameters to manage at the same time. Make sure you read the full tutorial before using it.
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
    Enables the selection of seismic events within a specified geometric shape on a cross-section derived from earthquake catalog data. It supports selections within circular or square shapes, allowing for targeted analysis of events in specific areas, considering both horizontal distance and depth.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame containing seismic event data, specifically including ``'on_section_coords'`` (horizontal distance along the section) and ``'depth'`` (vertical distance, typically below the surface). This dataset is obtained from :func: `cross_sections` function.

    center : tuple(float, float)
        The coordinates (x, y) representing the center of the geometric shape on the section, where 'x' is the horizontal distance from the starting point of the section and 'y' is the depth.

    size : tuple(int, int)
        The dimensions of the selection shape (in kilometers), specified as a tuple. For a circle, both elements of the tuple should be equal, representing the radius (otherwise oval). For a square, the elements define the length of the sides (if different, rectangle).

    rotation : int
        The rotation angle of the selection shape, in degrees, measured counter-clockwise from North.

    shape_type : str
        The type of geometric shape used for the selection. Valid options are 'circle' or 'square', which define the form of the area within which seismic events will be selected.

    plot : bool, optional
        If True, generates a plot illustrating the original dataset points alongside the subset of points that fall within the selected shape. Defaults to True, enabling visual verification of the selection.

    plot_center : bool, optional
        When True (and ``plot`` is also True), marks the geometric center of the selection shape on the plot, aiding in the visualization of the selection's focal point. Defaults to True.

    save_figure : bool, optional
    If set to True, the function saves the generated plots using the provided base name and file extension. The default is False.

    save_name : str, optional
        The base name used for saving figures when `save_figure` is True. It serves as the prefix for file names. The default base name is 'section'.

    save_extension : str, optional
        The file extension to use when saving figures, such as 'jpg', 'png', etc... The default extension is 'jpg'.

    return_indices : bool, optional
        If True, the function returns the indices of the points selected within the specified shape. If False, it returns a DataFrame containing the subset of selected data points. Defaults to False.

    Returns
    -------
    List[int] or pd.DataFrame
        Depending on the ``return_indices`` parameter, this function returns either a list of indices corresponding to the selected points or a DataFrame containing the subset of selected data points.
    
    See Also
    --------
    cross_sections : Analyze earthquake data to generate parallel cross sections perpendicular to a given strike, allowing for a comprehensive analysis of seismic activity around a central point of interest.
    
    select_on_map : Given an earthquake catalog containing latitude and longitude data, this function facilitates the selection of a subset of events falling within a specified geometric shape centered at a given point.
    
    Example
    -------
    
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
        
    .. image:: https://i.imgur.com/5dEwYsl.png
       :align: center
       :target: data_querying_and_selection.html#seismutils.geo.select
    
    The catalog used to demonstrate how the function works, specifically the data plotted in the image above, is derived from the `Tan et al. (2021) earthquake catalog <https://zenodo.org/records/4736089>`_.
    
    .. warning::
        The use of this function could be very complex since there are many parameters to manage at the same time. Make sure you read the full tutorial before using it.
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
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(x_coords, y_coords, marker='.', color='grey', s=0.25, alpha=0.75)
        ax.scatter(x_coords.iloc[selected_indices], y_coords.iloc[selected_indices], marker='.', color='red', s=0.25, alpha=0.75)
        
        if plot_center:
            ax.scatter(center[0], center[1], marker='o', color='red', edgecolor='black', linewidth=0.5,  s=150)
        
        plt.title(f'Selection', fontsize=14, fontweight='bold')
        plt.xlabel('Distance along strike [km]', fontsize=12)
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