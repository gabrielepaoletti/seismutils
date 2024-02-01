import pyproj

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import List, Tuple
from matplotlib.ticker import MultipleLocator

def cross_sections(data: pd.DataFrame,
                   center: Tuple,
                   num_sections: Tuple,
                   event_distance_from_section: int,
                   strike: int,
                   map_length: Tuple,
                   depth_range: Tuple,
                   section_distance: int = 1,
                   return_dataframes: bool=False) -> List[pd.DataFrame]:
    '''
    Plots earthquake data in cross sections around a specified central point based on their orientation and position.

    Args:
        data (pd.DataFrame): DataFrame with columns 'lon', 'lat', 'depth', etc., containing earthquake event data.
        center (tuple): Longitude and latitude of the central point (format: (longitude, latitude)) for the primary cross section.
        num_sections (tuple): Tuple specifying the number of additional sections to be plotted around the primary section.
                              Format: (num_left, num_right), where 'num_left' is the number of sections to the left (or before)
                              the primary section, and 'num_right' is the number of sections to the right (or after) the primary section.
        event_distance_from_section (int): Maximum distance (in kilometers) from a section within which an earthquake event is included.
        strike (int): The strike angle, in degrees from North, of the geological structure being studied. The sections are plotted 
                      perpendicular to this strike direction.
        map_length (tuple): Length of the section's trace in kilometers, specified as a tuple (length_left, length_right). 
                            'length_left' is the length to the left of the center point, and 'length_right' is the length to the right.
        depth_range (tuple): Depth range for considering earthquake events, specified as (min_depth, max_depth).
        section_distance (int): Distance between adjacent sections, measured in kilometers. Default is 1 kilometer.
        return_dataframes (bool, optional): If True, returns a list of DataFrames, one for each section. Default is False.

    Returns:
        list[pd.DataFrame] or None: If 'return_dataframes' is True, returns a list of DataFrames, each corresponding to a plotted section.
                                    Returns None if 'return_dataframes' is False.

    This function creates a visual representation of earthquake data by plotting cross sections. Each section is oriented
    perpendicular to a specified strike angle and centered around a given point. Earthquake events are included in a section
    if they fall within a specified distance from the section line and are within the defined depth range. The primary section
    is centered at the given 'center', with additional sections plotted to the left and right based on the 'num_sections' parameter.
    '''
    # Function to convert lat/lon to UTM
    def convert_to_utm(lon, lat):
        utm_converter = pyproj.Proj(proj='utm', zone=33, ellps='WGS84', datum='WGS84', units='m')
        utmx, utmy = utm_converter(np.array(lon), np.array(lat))
        return utmx / 1000, utmy / 1000  # Convert to km

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
    utmx, utmy = convert_to_utm(data['lon'], data['lat'])
    center_utmx, center_utmy = convert_to_utm(center[0], center[1])
    
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
        close_and_in_depth = np.where((dist < event_distance_from_section) & in_depth_range & (np.abs(utmx - center_coords[section][0]) < map_length))

        on_section_coords =+ (utmy[close_and_in_depth] - center_coords[section][1]) * normal_ref[0] - (utmx[close_and_in_depth] - center_coords[section][0])*normal_ref[1]
        
        # Plot sections
        fig = plt.figure(figsize=(15, 7))
        
        plt.scatter(on_section_coords, -data.depth.iloc[close_and_in_depth], marker='.', color='black', s=0.25, alpha=0.75)
        plt.title(f'Section {section+1}', fontsize=14, fontweight='bold')
        
        # Format plot axis
        plt.gca().xaxis.set_major_locator(MultipleLocator(5))
        plt.gca().xaxis.set_major_formatter('{x:.0f}')
        plt.gca().xaxis.set_minor_locator(MultipleLocator(1))
        
        plt.gca().yaxis.set_major_locator(MultipleLocator(5))
        plt.gca().yaxis.set_major_formatter('{x:.0f}')
        plt.gca().yaxis.set_minor_locator(MultipleLocator(1))
        
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
            section_df['on_section_coords'] = on_section_coords
            
            # Append section dataframes to a list
            section_dataframes.append(section_df)
            
    return section_dataframes

def select(coords: Tuple, 
           center: Tuple, 
           size: Tuple,
           rotation: int,
           shape_type: str) -> List:
    '''
    Selects and returns indices of points from a tuple of coordinates that fall within a specified geometric 
    shape (circle, oval, rectangle) around a central point, with an option to rotate the shape.

    Args:
        coords (tuple of arrays): A tuple containing two arrays (x_coords, y_coords), representing the x and y 
                                  coordinates of each point.
        center (tuple): Coordinates of the center of the shape (format: (x, y)).
        size (tuple or int): Size of the shape. For circles, it's a single integer representing the radius. For ovals 
                             and rectangles, it's a tuple representing (width, height).
        rotation (int): Rotation angle of the shape in degrees, counter-clockwise from the x-axis.
        shape_type (str): Type of the geometric shape. Options are 'circle', 'oval', or 'rectangle'.

    Returns:
        list: A list of indices of the points that fall within the specified shape.

    This function iterates over each point in the input tuple of coordinates, transforming the coordinates of 
    each point according to the specified rotation and shape center. It then checks if the transformed point falls 
    within the given shape (circle, oval, or rectangle).
    '''
    def rotate_point(point: Tuple,
                     center: Tuple, 
                     angle: int) -> tuple:
        '''
        Rotates a point around a given center by a specified angle.

        Args:
            point (tuple): The coordinates of the point to be rotated (format: (x, y)).
            center (tuple): The coordinates of the rotation center (format: (x, y)).
            angle (int): The angle of rotation in degrees. Positive values rotate the point counter-clockwise, 
                         while negative values rotate it clockwise.

        Returns:
            tuple: The coordinates of the rotated point (format: (x, y)).

        This function calculates the new coordinates of a point after rotating it around a specified center. 
        The rotation is performed in the 2D plane, and the angle is specified in degrees.
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

    return selected_indices