import pyproj

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import List, Tuple
from matplotlib.ticker import MultipleLocator

def cross_sections(data: pd.DataFrame,
                   center: Tuple[float, float],
                   num_sections: Tuple[int, int],
                   event_distance_from_section: int,
                   strike: int,
                   map_length: int,
                   depth_range: Tuple[int, int],
                   section_distance: int = 1,
                   plot: bool=False,
                   return_dataframes: bool=True) -> List[pd.DataFrame]:
    '''
    Analyzes and optionally plots earthquake data in cross sections around a specified central point, based on their orientation and position relative to a geological strike. 

    Args:
        data (pd.DataFrame): DataFrame containing earthquake event data, with columns 'lon', 'lat', 'depth', etc.
        center (Tuple[float, float]): Coordinates (longitude, latitude) of the central point for the primary cross section.
        num_sections (Tuple[int, int]): Number of additional sections to be analyzed and plotted around the primary section, specified as (num_left, num_right).
        event_distance_from_section (int): Maximum distance (in kilometers) from a section within which an earthquake event is considered for inclusion.
        strike (int): Strike angle in degrees from North, indicating the geological structure's orientation. Sections are plotted perpendicular to this strike direction.
        map_length (int): Length of the section's trace in kilometers, extending equally in both directions from the center point.
        depth_range (Tuple[int, int]): Depth range (min_depth, max_depth) for considering earthquake events.
        section_distance (int, optional): Distance between adjacent sections, in kilometers. Defaults to 1 kilometer.
        plot (bool, optional): If True, plots the cross sections with the included earthquake events. Defaults to False.
        return_dataframes (bool, optional): If True, returns a list of DataFrames, each representing a section with included earthquake events. Defaults to True.

    Returns:
        List[pd.DataFrame] or None: A list of DataFrames, each corresponding to a section with relevant earthquake data, if 'return_dataframes' is True. Returns None otherwise.

    This function facilitates the analysis of earthquake events in relation to a specified geological strike. It computes cross sections perpendicular to the strike, centered 
    around a given point, and analyzes earthquake events within these sections based on their proximity to the section plane and depth. The function optionally plots these sections, 
    showing the spatial distribution of events, and can return the data in a structured format for further analysis.
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

def select(data: pd.DataFrame,
           coords: Tuple[pd.Series, pd.Series], 
           center: Tuple[float, float], 
           size: Tuple[int, int],
           rotation: int,
           shape_type: str,
           plot: bool = False,
           return_indices: bool=False) -> List:
    '''
    Selects indices of points from a set of coordinates that fall within a specified geometric shape (circle, oval, rectangle) centered at a given point. The shape can be rotated by 
    a specified angle.

    Args:
        coords (Tuple[pd.Series, pd.Series]): A pair of Series representing x and y coordinates of points.
        center (Tuple[float, float]): The x and y coordinates of the shape's center.
        size (Tuple[int, int] or int): The dimensions of the shape. For circles, it's a single integer (radius); for ovals and rectangles, it's (width, height).
        rotation (int): The counter-clockwise rotation angle of the shape in degrees from the x-axis.
        shape_type (str): The type of geometric shape ('circle', 'oval', or 'rectangle').
        plot (bool, optional): If True, plots the original points and the selected points. Defaults to False.

    Returns:
        List[int]: Indices of points within the specified shape.

    This function assesses each point to determine if it lies within a rotated geometric shape centered at a specified point. It supports selection within circles, ovals, and rectangles.
    An optional plot can be generated to visualize the selection.
    '''
    def rotate_point(point: Tuple[float, float],
                     center: Tuple[float, float], 
                     angle: int) -> tuple:
        '''
        Rotates a point around a given center by a specified angle in the 2D plane.

        Args:
            point (Tuple[float, float]): Coordinates of the point (x, y) to be rotated.
            center (Tuple[float, float]): Coordinates of the rotation center (x, y).
            angle (int): Rotation angle in degrees, where positive values indicate counter-clockwise rotation.

        Returns:
            Tuple[float, float]: The coordinates of the rotated point (x, y).

        This function computes the new coordinates of a point after rotating it around a specified center point by a given angle in degrees.
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