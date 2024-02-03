import pyproj

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator

def convert_to_geographical(utmx, utmy, zone, northern, units, ellps='WGS84', datum='WGS84'):
    '''
    Converts UTM (Universal Transverse Mercator) coordinates to geographical coordinates (latitude and longitude).

    :param float utmx: UTM x coordinate (Easting).
    :param float utmy: UTM y coordinate (Northing).
    :param int zone: UTM zone number from which the coordinates are mapped.
    :param bool northern: Boolean indicating whether the UTM coordinates are in the northern hemisphere (True) or southern (False).
    :param str units: The units of the UTM coordinates (defaults to 'm' for meters).
    :param str ellps: The ellipsoid used in the conversion (default is 'WGS84').
    :param str datum: The datum used in the conversion (default is 'WGS84').
    :return: A tuple containing the latitude and longitude.
    :rtype: (float, float)

    This function uses the ``pyproj`` library to convert UTM coordinates into latitude and longitude based on the specified UTM zone,
    hemisphere, units, and ellipsoid. This reverse conversion is useful for mapping applications that require global positioning data.
    '''
    # Define the geographic and UTM CRS based on the zone and hemisphere
    utm_crs = pyproj.CRS(f'+proj=utm +zone={zone} +{"+north" if northern else "+south"} +ellps={ellps} +datum={datum} +units={units}')
    geodetic_crs = pyproj.CRS('epsg:4326')
    
    # Create a Transformer object to convert between CRSs
    transformer = pyproj.Transformer.from_crs(utm_crs, geodetic_crs, always_xy=True)
    
    # Transform the coordinates
    lon, lat = transformer.transform(utmx, utmy)
    return lon, lat

def convert_to_utm(lon, lat, zone, units, ellps='WGS84', datum='WGS84'):
    '''
    Converts geographical coordinates (latitude and longitude) to UTM (Universal Transverse Mercator) coordinates.

    :param float lon: Longitude of the point(s) to convert.
    :param float lat: Latitude of the point(s) to convert.
    :param int zone: UTM zone number that the coordinates should be mapped into.
    :param str units: The units of the UTM coordinates (e.g., 'm' for meters).
    :param str ellps: The ellipsoid used in the conversion (default is 'WGS84').
    :param str datum: The datum used in the conversion (default is 'WGS84').
    :return: A tuple containing the UTM coordinates (utmx, utmy).
    :rtype: (float, float)

    The function uses the ``pyproj`` library to convert lat/lon coordinates into UTM coordinates based on the specified UTM zone,
    units, ellipsoid, and datum. This is particularly useful for transforming geospatial data into a planar coordinate system
    suitable for linear distance measurements and mapping applications.
    '''
    # Create a pyproj Proj object for UTM conversion using the given zone and ellipsoid.
    utm_converter = pyproj.Proj(proj='utm', zone=zone, units=units, ellps=ellps, datum=datum)

    # Transform the coordinates
    utmx, utmy = utm_converter(np.array(lon), np.array(lat))
    return utmx, utmy

def cross_sections(data, center, num_sections, event_distance_from_section, strike, map_length, depth_range, section_distance, zone, plot=False, return_dataframes=True):
    '''
    Analyzes and optionally plots earthquake data in cross sections around a specified central point, based on their orientation and position relative to a geological strike.

   :param pd.DataFrame data: DataFrame containing earthquake event data, with columns 'lon', 'lat', 'depth', etc.
   :param (float, float) center: Coordinates (longitude, latitude) of the central point for the primary cross section.
   :param (int, int) num_sections: Number of additional sections to be analyzed and plotted around the primary section, specified as (num_left, num_right).
   :param int event_distance_from_section: Maximum distance (in kilometers) from a section within which an earthquake event is considered for inclusion.
   :param int strike: Strike angle in degrees from North, indicating the geological structure's orientation. Sections are plotted perpendicular to this strike direction.
   :param int map_length: Length of the section's trace in kilometers, extending equally in both directions from the center point.
   :param (int, int) depth_range: Depth range (min_depth, max_depth) for considering earthquake events.
   :param int section_distance: Distance between adjacent sections, in kilometers. Defaults to 1 kilometer.
   :param int zone: UTM zone number that the coordinates should be mapped into.
   :param bool plot: If True, plots the cross sections with the included earthquake events. Defaults to False.
   :param bool return_dataframes: If True, returns a list of DataFrames, each representing a section with included earthquake events. Defaults to True.
   :return: A list of DataFrames, each corresponding to a section with relevant earthquake data, if 'return_dataframes' is True. Returns None otherwise.
   :rtype: List[pd.DataFrame]

   This function facilitates the analysis of earthquake events in relation to a specified geological strike. It computes cross sections perpendicular to the strike, centered around a given point, and analyzes earthquake events within these sections based on their proximity to the section plane and depth. The function optionally plots these sections, showing the spatial distribution of events, and can return the data in a structured format for further analysis.
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
    
    # Convert earthquake data and center to UTM coordinates
    utmx, utmy = convert_to_utm(data['lon'], data['lat'], )
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
            plt.show()
        
        # Add the events of this section to the list if return_dataframes is True
        if return_dataframes:
            # Add on section coordinates to the dataframe
            section_df = data.iloc[close_and_in_depth].copy().reset_index(drop=True)
            section_df['on_section_coords'] = on_section_coords[close_and_in_depth]
            
            # Append section dataframes to a list
            section_dataframes.append(section_df)     
    
    return section_dataframes

def select(data, coords, center, size, rotation, shape_type, plot, return_indices):
    '''
    Selects and optionally plots indices of points from a dataset that fall within a specified geometric shape (circle, oval, rectangle) centered at a given point and rotated by a specified angle.

    :param pd.DataFrame data: DataFrame containing the dataset with points to select from.
    :param (pd.Series, pd.Series) coords: A pair of Series representing the x and y coordinates.
    :param (float, float) center: The (x, y) coordinates of the shape's center.
    :param (int, int) size: The dimensions of the shape (width, height) or radius if the shape is a circle.
    :param int rotation: The counter-clockwise rotation angle of the shape in degrees from the x-axis.
    :param str shape_type: The type of geometric shape to select within ('circle', 'oval', 'rectangle').
    :param bool plot: If True, plots the original points and the selected points. Defaults to False.
    :param bool return_indices: If True, returns a list of indices, otherwise returns a subset DataFrame. Defaults to False.
    :return: A list of selected indices or a subset DataFrame, depending on the return_indices parameter.
    :rtype: List[int] or pd.DataFrame

    This function determines if points lie within a specified rotated geometric shape centered at the provided coordinates.
    Supported shapes include circles, ovals, and rectangles. The function can also visualize the selection if plotting is enabled.
    '''
    def rotate_point(point, center, angle):
        '''
    Rotates a point around a given center by a specified angle in the 2D plane.

    :param (float, float) point: The (x, y) coordinates of the point to be rotated.
    :param (float, float) center: The (x, y) coordinates of the rotation center.
    :param int angle: Rotation angle in degrees, positive values indicate counter-clockwise rotation.
    :return: The (x, y) coordinates of the rotated point.
    :rtype: tuple

    This helper function calculates the new position of a point after being rotated about a specified center point by a given angle in degrees.
        '''
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
        plt.show()
    
    if return_indices:  
        return selected_indices
    else:
        return data.iloc[selected_indices].reset_index(drop=True)