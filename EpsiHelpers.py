# EpsiHelpers.py

import numpy as np


def read_epsi_plot(self):
    """
    Set the EPSI data based on configuration and current value.

    @author Benjamin (Ben) Yoon
    @date Fri, Aug 11, 2023
    @version 1.0
    """
    # Constants
    proton_quarter = self.info_epsi['proton'] / 4

    # Read EPSI data based on the EPSI slider value
    self.path_epsi = f"{self.path_13c}{self.current_value_epsi:02d}"
    spectral_data = self.class_spectral_data_instance.read_write_spectral_data(self.info_epsi, self.path_epsi,
                                                                               proton_quarter)
    # Preprocessing
    spectral_data = np.flip(np.flip(spectral_data, 0), 1)
    # if self.picture_information:
    #    spectral_data = self.class_spectral_data_instance.correct_epsi_plot(self, spectral_data)
    if self.scale:
        maximum_spectral_data_value = np.max(spectral_data)
        spectral_data = spectral_data / maximum_spectral_data_value
    else:
        maximum_spectral_data_value = np.max(spectral_data, axis=2)
        spectral_data = spectral_data / maximum_spectral_data_value
    epsi = []
    for i in range(0, self.rows):
        row_information = []
        for j in range(0, self.columns):
            if np.max(spectral_data[i, j, :]) < 0.20:
                spectral_data[i, j, :] = np.nan
            row_information = np.concatenate(
                (np.squeeze(row_information), np.squeeze(np.roll(spectral_data[i, j, :], 0))))
        epsi = np.concatenate((np.squeeze(epsi), np.squeeze(row_information + self.rows - i)))
    x_epsi = np.tile(np.arange(0, spectral_data.shape[2] * self.columns), self.rows)
    for nan_rows in range(0, self.rows - 1):
        epsi[nan_rows * spectral_data.shape[2] * self.columns] = np.nan
    epsi = np.convolve(epsi, np.ones(self.moving_average_window), mode='same') / self.moving_average_window
    epsi[~np.isnan(epsi)] -= 1

    # Adjust subplot position
    self.lro_fid = self.class_spectral_data_instance.read_write_procpar('lro', self.path_fid)[0] * 10
    self.lpe_fid = self.class_spectral_data_instance.read_write_procpar('lpe 1', self.path_fid)[0] * 10
    self.lro_epsi = self.class_spectral_data_instance.read_write_procpar('lro', self.path_epsi)[0] * 10
    self.lpe_epsi = self.class_spectral_data_instance.read_write_procpar('lpe 1', self.path_epsi)[0] * 10

    self.x_epsi = x_epsi
    self.epsi = epsi
    self.spectral_data = spectral_data


def convert_to_inset_coordinates(self, coords_total, axis_c):
    """
    Converts a list of data coordinates to inset axis coordinates.

    :param self: self of EpsiGui class to update global flags
    :param coords_total: (list) List of data coordinates.
    :param axis_c: (Axes) Inset axis for the color map.

    :return: (list) List of inset axis coordinates.

    @author Benjamin (Ben) Yoon
    @date Fri, Aug 11, 2023
    @version 1.0
    """
    coords_inset = []

    for x, y in coords_total:
        coordinates_data_transformed = self.axis.transData + axis_c.transData.inverted()
        coords_axis_c = coordinates_data_transformed.transform([(x, y)])[0]
        coords_inset.append(coords_axis_c)

    return coords_inset


def get_interior_points(coordinates_perimeter):
    """
    Finds interior points within the given perimeter using a grid-based approach.

    :param coordinates_perimeter: (list) List of perimeter coordinates.
    :return: (list) List of interior coordinates.

    @author Benjamin (Ben) Yoon
    @date Fri, Aug 11, 2023
    @version 1.0
    """
    # Import required libraries
    from shapely.geometry import Polygon, Point

    # Create a shapely polygon from the perimeter coordinates
    polygon = Polygon(coordinates_perimeter)

    # Determine whether the shape is closed by checking if the first and last points are the same
    is_roi_closed = coordinates_perimeter[0] == coordinates_perimeter[-1]

    # Get bounding box coordinates
    min_x, min_y = min(coordinates_perimeter, key=lambda item: item[0])[0], \
        min(coordinates_perimeter, key=lambda item: item[1])[1]
    max_x, max_y = max(coordinates_perimeter, key=lambda item: item[0])[0], \
        max(coordinates_perimeter, key=lambda item: item[1])[1]

    # Define grid spacing
    grid_spacing = 0.5

    # Generate interior points using a grid-based approach
    coords_interior = []

    for x in np.arange(min_x, max_x + grid_spacing, grid_spacing):
        for y in np.arange(min_y, max_y + grid_spacing, grid_spacing):
            point = Point(x, y)

            # Check if the point falls within the polygon
            if point.within(polygon):
                coords_interior.append((x, y))

    # If the shape is not closed, connect the first and last points
    if not is_roi_closed:
        coords_interior.append(coords_interior[0])

    return coords_interior


def load_array_subplots(self, coords_axis_plot, rs, cs):
    """
    Calculates information for each subplot based on the given plot axis coordinates.

    :param self: self of the EpsiGui class to update global fields
    :param coords_axis_plot: (tuple) The coordinates of the main plot axis (x_min, x_max, y_min, y_max).
    :param rs: (int) Number of rows in the subplot grid.
    :param cs: (int) Number of columns in the subplot grid.

    :return: (list) List of lists containing subplot information: [x_min, x_max, y_min, y_max, max_spectral_value].

    @author Benjamin (Ben) Yoon
    @date Fri, Aug 11, 2023
    @version 1.0
    """
    min_main_x, max_main_x, min_main_y, max_main_y = coords_axis_plot
    domain = max_main_x - min_main_x
    range_y = max_main_y - min_main_y
    step_x = domain / cs
    step_y = range_y / rs
    array = []

    for r in range(rs):
        for c in range(cs):
            min_x_subplot = min_main_x + c * step_x
            max_x_subplot = min_main_x + (c + 1) * step_x
            min_y_subplot = min_main_y + r * step_y
            max_y_subplot = min_main_y + (r + 1) * step_y

            with np.errstate(all='ignore'):
                # Extract max spectral value within subplot boundaries
                max_spectral_value = np.nanmax(self.spectral_data[r, c, :])

                if np.isnan(max_spectral_value):
                    max_spectral_value = 0.0  # Default value for nan max values

            array.append([min_x_subplot, max_x_subplot, min_y_subplot, max_y_subplot, max_spectral_value])

    return array
