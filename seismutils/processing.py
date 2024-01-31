import numpy as np

def make_selection(coords, center, size, rotation, shape_type):
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
    def rotate_point(point, center, angle):
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