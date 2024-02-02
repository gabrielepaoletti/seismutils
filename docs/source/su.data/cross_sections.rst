Cross sections
======================================

.. py:function:: cross_sections(data: pd.DataFrame, center: Tuple[float, float], num_sections: Tuple[int, int], event_distance_from_section: int, strike: int, map_length: int, depth_range: Tuple[int, int], section_distance: int = 1, plot: bool = False, return_dataframes: bool = True) -> List[pd.DataFrame]

   Analyzes and optionally plots earthquake data in cross sections around a specified central point, based on their orientation and position relative to a geological strike.

   :param pd.DataFrame data: DataFrame containing earthquake event data, with columns 'lon', 'lat', 'depth', etc.
   :param Tuple[float, float] center: Coordinates (longitude, latitude) of the central point for the primary cross section.
   :param Tuple[int, int] num_sections: Number of additional sections to be analyzed and plotted around the primary section, specified as (num_left, num_right).
   :param int event_distance_from_section: Maximum distance (in kilometers) from a section within which an earthquake event is considered for inclusion.
   :param int strike: Strike angle in degrees from North, indicating the geological structure's orientation. Sections are plotted perpendicular to this strike direction.
   :param int map_length: Length of the section's trace in kilometers, extending equally in both directions from the center point.
   :param Tuple[int, int] depth_range: Depth range (min_depth, max_depth) for considering earthquake events.
   :param int section_distance: Distance between adjacent sections, in kilometers. Defaults to 1 kilometer.
   :param bool plot: If True, plots the cross sections with the included earthquake events. Defaults to False.
   :param bool return_dataframes: If True, returns a list of DataFrames, each representing a section with included earthquake events. Defaults to True.
   :return: A list of DataFrames, each corresponding to a section with relevant earthquake data, if 'return_dataframes' is True. Returns None otherwise.
   :rtype: List[pd.DataFrame]

   This function facilitates the analysis of earthquake events in relation to a specified geological strike. It computes cross sections perpendicular to the strike, centered around a given point, and analyzes earthquake events within these sections based on their proximity to the section plane and depth. The function optionally plots these sections, showing the spatial distribution of events, and can return the data in a structured format for further analysis.