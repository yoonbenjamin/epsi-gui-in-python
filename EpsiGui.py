# @package EpsiGui
#  A graphical user interface (GUI) for visualizing and interacting with EPSI data.
#
#  Version: 1.0
#  This GUI allows users to display proton pictures, adjust contrast, plot EPSI data, and create color maps based on
#  spectral values. It provides interactive tools for selecting regions of interest (ROIs) and generating color-coded
#  maps to enhance the visualization of the data.
#
#  Version: 1.1 General bugs fixed, global contrast implemented, variables the user may want to change moved to the
#  top of the code, sliders replaced with buttons, color map speed improved--only finds color map coordinates once (
#  fun split into multiple functions), general speed increase (remove previous plots), & code restructured to move
#  certain functions into EpsiHelpers.
#
#  Author: Benjamin (Ben) Yoon
#  Date: Fri, Sep 29, 2023
#  Version: 1.1

import os
import matplotlib.pyplot as plt
import matplotlib.style as style
from matplotlib import colors as m_colors
from matplotlib.widgets import LassoSelector, Button
import numpy as np
import pydicom
import cv2

from SpectralData import SpectralData
from EpsiHelpers import read_epsi_plot, get_interior_points, load_array_subplots, convert_to_inset_coordinates

# USER SHOULD CHANGE VARIABLES TO MATCH DESIRED EXPERIMENT
# Path to the base folder containing EPSI data and DICOM files.
folder_path_base = "/Users/benjaminyoon/Desktop/PIGI folder/Projects/Project2 EPSI GUI in Python/EPSI " \
                   "GUI/data_mouse_kidney/s_2023041103/"

# Path to the folder containing DICOM files for proton pictures.
folder_path_dcm = folder_path_base + "fsems_rat_liver_03.dmc/"

# Path to the folder containing EPSI data files.
folder_path_13c = folder_path_base + "epsi_16x12_13c_"

# Path to the folder containing fid data files.
folder_path_fid = folder_path_base + "fsems_rat_liver_03"

# Shift values for the EPSI plot.
epsi_plot_shift = [-0.3, -0.4]

# Information about EPSI data.
epsi_information = {
    'pictures_to_read_write': 1,
    'proton': 60,
    'centric': 1
}

# Number of columns in the EPSI data grid.
columns = 16

# Number of rows in the EPSI data grid.
rows = 12

# Plot axis coordinates
coordinates_axis_plot = (14, 233, 51, 215)

# The number of image slices
image_slices = 20

# The number of EPSI data sets
epsi_sets = 17

# Set the style to a dark theme
style.use('dark_background')


# @class EpsiGui
#  A graphical user interface (GUI) for visualizing and interacting with EPSI (Echo Planar Spectroscopic Imaging) data.
#
#  This GUI allows users to display proton pictures, adjust contrast, plot EPSI data, and create color maps based on
#  spectral values. It provides interactive tools for selecting regions of interest (ROIs) and generating color-coded
#  maps to enhance the visualization of the data.
#
#  @author Benjamin (Ben) Yoon
#  @date Fri, Sep 29, 2023
#  @version 1.1
class EpsiGui:

    # Initializes the EpsiGui class.
    #
    #  @author Benjamin (Ben) Yoon
    #  @date Fri, Sep 29, 2023
    #  @version 1.1
    #
    #  @param f_path_dcm Path to the directory containing DICOM files for proton image slices (str).
    #  @param f_path_13c Path to the directory containing EPSI data files (str).
    #  @param f_path_fid Path to the directory containing fid data files (str).
    #  @param epsi_shift A list of two shift values for the EPSI plot (list of float).
    #  @param epsi_info A dictionary containing information about EPSI data (dict).
    #                   Example:
    #                   {
    #                       'pictures_to_read_write': 1,  # Number of pictures to read and write (int)
    #                       'proton': 60,  # Proton picture number (int)
    #                       'centric': 1  # Centric flag (int)
    #                   }
    #
    #  @param cols Number of columns (int).
    #  @param rs Number of rows (int).
    #  @param coords_axis_plot Plot axis coordinates (Tuple)
    #  @param img_slices Number of image slices (int)
    #  @param epsi_set_count Number of EPSI data sets (int)
    def __init__(self, f_path_dcm, f_path_13c, f_path_fid, epsi_shift, epsi_info, cols, rs, coords_axis_plot,
                 img_slices, epsi_set_count):
        # VERSION 1.0:
        self.button_left_image_slice = None  # Button for decreasing image slice slider
        self.button_right_image_slice = None  # Button for increasing image slice slider
        self.button_left_contrast = None  # Button for decreasing contrast slider
        self.button_right_contrast = None  # Button for increasing contrast slider
        self.button_left_epsi = None  # Button for decreasing epsi slider
        self.button_right_epsi = None  # Button for increasing epsi slider
        self.button_off_write_epsi = None  # Button for EPSI viewing toggle on
        self.button_on_write_epsi = None  # Button for EPSI viewing toggle off
        self.is_write_epsi_on = False
        self.window = None
        self.axis = None
        self.axis_plot = None
        self.axis_epsi_data = None
        self.class_spectral_data_instance = SpectralData()
        self.path_dmc = f_path_dcm
        self.path_fid = f_path_fid
        self.path_13c = f_path_13c
        self.path_epsi = ""
        # self.picture_information = 1
        self.scale = True
        self.moving_average_window = 1
        self.slice_proton_picture = None
        self.files_proton_pictures = []
        for file_name in os.listdir(self.path_dmc):
            if file_name.endswith(".dcm"):
                self.files_proton_pictures.append(os.path.join(self.path_dmc, file_name))
        self.files_proton_pictures.sort()
        self.plot_shift = epsi_shift
        self.info_epsi = epsi_info
        self.lro_fid = None
        self.lpe_fid = None
        self.lro_epsi = None
        self.lpe_epsi = None
        self.x_epsi = None
        self.epsi = None
        self.spectral_data = None
        self.button_color_map = None  # Button for colormap
        self.axis_color_map = None
        self.button_reset = None  # Button for resetting plots on axes
        self.is_color_map_on = False

        # VERSION 1.1
        self.proton_pictures = img_slices
        self.current_value_image_slice = int((image_slices - 1) / 2)
        self.current_value_contrast = 0.2
        self.current_value_epsi = 1
        self.is_contrast_adjusted = False
        self.columns = cols
        self.rows = rs
        self.coordinates_axis_plot = coords_axis_plot
        self.epsi_data_sets = epsi_set_count
        self.coordinates_total = None
        self.text_element_image_slice = None
        self.text_element_contrast = None
        self.text_element_epsi = None
        self.data_color_map = None

        self.initialize_gui()

    def process_epsi(self, event):
        """
        Initiates the display of the EPSI plot when the "on" button is clicked and removes the previous EPSI plot when
        the EPSI number changes.

        :param event: The event generated by clicking the "EPSI View ON/OFF" button.

        @author Benjamin (Ben) Yoon
        @date Mon, Aug 14, 2023
        @version 1.0
        """
        try:
            self.axis_epsi_data.remove()
        except:
            pass
        try:
            self.axis_plot.remove()
        except:
            pass

        if self.is_write_epsi_on:
            self.write_epsi_plot(event)
        else:
            if self.is_color_map_on is True:
                self.write_epsi_plot(event)

        self.window.canvas.draw()

    def create_inset_axis(self, axis_number):
        """
        Creates an inset axis within the axis.

        @author Benjamin (Ben) Yoon
        @date Fri, Sep 29, 2023
        @version 1.1
        """
        if axis_number == 0:
            self.axis_plot = self.axis.inset_axes([((self.lro_fid - self.lro_epsi) / 2 + self.plot_shift[0] * self.lro_epsi /
                                      self.columns) / self.lro_fid, ((self.lpe_fid - self.lpe_epsi) / 2 +
                                                                     self.plot_shift[1] * self.lpe_epsi / self.rows) /
                                     self.lpe_fid, self.lro_epsi / self.lro_fid, self.lpe_epsi / self.lpe_fid])
        if axis_number == 1:
            self.axis_epsi_data = self.axis.inset_axes([((self.lro_fid - self.lro_epsi) / 2 + self.plot_shift[0] * self.lro_epsi /
                                      self.columns) / self.lro_fid, ((self.lpe_fid - self.lpe_epsi) / 2 +
                                                                     self.plot_shift[1] * self.lpe_epsi / self.rows) /
                                     self.lpe_fid, self.lro_epsi / self.lro_fid, self.lpe_epsi / self.lpe_fid])
        if axis_number == 2:
            self.axis_color_map = self.axis.inset_axes([((self.lro_fid - self.lro_epsi) / 2 + self.plot_shift[0] * self.lro_epsi /
                                      self.columns) / self.lro_fid, ((self.lpe_fid - self.lpe_epsi) / 2 +
                                                                     self.plot_shift[1] * self.lpe_epsi / self.rows) /
                                     self.lpe_fid, self.lro_epsi / self.lro_fid, self.lpe_epsi / self.lpe_fid])

    def show_proton_picture(self):
        """
        Displays a proton picture corresponding to the given index.

        @author Benjamin (Ben) Yoon
        @date Fri, Jul 21, 2023
        @version 1.0
        """
        file_path_proton_picture = self.files_proton_pictures[self.current_value_image_slice]
        dcm_read = pydicom.dcmread(file_path_proton_picture)
        self.slice_proton_picture = dcm_read.pixel_array
        self.axis.clear()
        self.axis.axis("off")
        self.axis.imshow(self.slice_proton_picture, cmap='gray')

        if self.is_contrast_adjusted:
            self.adjust_proton_picture_contrast()
        if self.is_write_epsi_on:
            self.process_epsi(None)
        else:
            if self.is_color_map_on:
                try:
                    self.axis_color_map.remove()
                except:
                    pass

                self.create_inset_axis(2)
                self.plot_on_color_map(self.data_color_map)

        self.window.canvas.draw()

    def adjust_proton_picture_contrast(self):
        """
        Adjusts the contrast of the displayed proton picture using Contrast Limited Adaptive Histogram Equalization
        (CLAHE).

        @author Benjamin (Ben) Yoon
        @date Fri, Sep 29, 2023
        @version 1.1
        """
        self.is_contrast_adjusted = True

        # Normalize pixel values to [0, 1]
        proton_picture_normalized = (self.slice_proton_picture - np.min(self.slice_proton_picture)) / (
                np.max(self.slice_proton_picture) - np.min(self.slice_proton_picture))

        # Apply CLAHE for contrast adjustment
        clahe = cv2.createCLAHE(clipLimit=self.current_value_contrast, tileGridSize=(8, 8))
        proton_picture_clahe = clahe.apply(np.uint8(proton_picture_normalized * 255))

        # Rescale pixel values back to [0, 1]
        proton_picture_rescaled = proton_picture_clahe / 255.0

        self.axis.clear()
        self.axis.axis("off")
        self.axis.imshow(proton_picture_rescaled, cmap='gray')
        if self.is_write_epsi_on is True:
            self.process_epsi(None)
        else:
            if self.is_color_map_on:
                try:
                    self.axis_color_map.remove()
                except:
                    pass

                self.create_inset_axis(2)
                self.plot_on_color_map(self.data_color_map)

        self.window.canvas.draw()

    def plot_on_color_map(self, data_map):
        """
        Plots the color map with data points and corresponding colors.

        :param data_map: (list) A list of tuples where each tuple contains a data coordinate and its associated
        RGB color.

        @author Benjamin (Ben) Yoon
        @date Fri, Aug 11, 2023
        @version 1.0
        """
        # Save current axis limits
        x_limit = self.axis_color_map.get_xlim()
        y_limit = self.axis_color_map.get_ylim()

        # Extract x and y coordinates and corresponding colors from data_color_map
        x_coords = [coord[0] for coord, _ in data_map]
        y_coords = [coord[1] for coord, _ in data_map]
        colors = [rgb_color for _, rgb_color in data_map]

        # Scatter plot of data points with specified colors and transparency
        self.axis_color_map.scatter(x_coords, y_coords, c=colors, marker='o', alpha=0.01)

        # Restore original axis limits
        self.axis_color_map.set_xlim(x_limit)
        self.axis_color_map.set_ylim(y_limit)

        # Turn off axis ticks and labels
        self.axis_color_map.axis('off')

        # Refresh the plot to display the color map on the axis
        plt.draw()

    def write_epsi_plot(self, event):
        """
        Generates and displays the EPSI plot while updating the colormap based on the EPSI dataset, if plotted.

        :param event: The event generated for generating the EPSI plot.

        @author Benjamin Yoon
        @date Mon, Aug 14, 2023
        @version 1.0
        """
        read_epsi_plot(self)

        if self.is_color_map_on:
            try:
                self.axis_color_map.remove()
            except:
                pass

            self.process_color_map_and_plot(self.coordinates_total)

        if self.is_write_epsi_on:
            self.create_inset_axis(1)
            self.axis_epsi_data.plot(self.x_epsi, np.squeeze(self.epsi), color='#FF00FF', linewidth=1)
            self.axis_epsi_data.set_ylim([0, self.rows])
            self.axis_epsi_data.set_xlim([0, self.columns * self.spectral_data.shape[2]])
            self.axis_epsi_data.axis('off')
            self.create_inset_axis(0)

            for i in range(0, self.columns + 1):
                self.axis_plot.axvline(x=i, linestyle='--', color='w', linewidth=1, alpha=0.5)
            for j in range(0, self.rows + 1):
                self.axis_plot.axhline(y=j, linestyle='--', color='w', linewidth=1, alpha=0.5)

            self.axis_plot.set_ylim([0, self.rows + 0.05])
            self.axis_plot.set_xlim([0, self.columns + 0.05])
            self.axis_plot.axis('off')
            self.window.canvas.draw()

    def process_color_map_and_plot(self, coordinates_total):
        """
        Creates a color map inset axis and processes the color map for the ROI and
        plots the coordinates.

        :param coordinates_total: (list) Combined coordinates representing the ROI.

        @author Benjamin (Ben) Yoon
        @date Fri, Sep 29, 2023
        @version 1.1
        """
        try:
            self.axis_color_map.remove()
        except:
            pass
        # Creat a color map inset axis
        self.create_inset_axis(2)

        # Load subplot information: [x_min, x_max, y_min, y_max, max_spectral_value]
        array_subplots = load_array_subplots(self, self.coordinates_axis_plot, self.rows, self.columns)

        # Create a color map gradient
        gradient_color_map = []
        for subplot in array_subplots:
            value_maximum = subplot[4]
            value_normalized = value_maximum / np.nanmax(self.epsi)
            color_rgba = (0.0, 0.0, value_normalized, value_normalized)
            gradient_color_map.append(color_rgba)

        # Initialize the array for data and color mapping
        color_map_data = []

        # Define RGB values for the color points
        points_color_map_gradient = [
            (0.0, (0.11, 0.435, 0.973)),  # Blue
            (0.2, (0.153, 0.733, 0.878)),  # Battery Charged Blue
            (0.4, (0.192, 0.859, 0.573)),  # Eucalyptus
            (0.6, (0.106, 0.945, 0.094)),  # Neon Green
            (0.8, (0.608, 0.980, 0.141)),  # Green Lizard
            (1.0, (0.996, 0.969, 0.125))  # Yellow
        ]

        # Define the custom colormap with RGB color points
        color_map = m_colors.LinearSegmentedColormap.from_list('custom_colormap', points_color_map_gradient)

        # Assign colors based on max spectral values
        for coord in coordinates_total:
            coords, y = coord
            matching_subplot = None

            for subplot in array_subplots:
                x_min, x_max, y_min, y_max, _ = subplot
                if x_min <= coords <= x_max and y_min <= y <= y_max:
                    matching_subplot = subplot
                    break

            if matching_subplot:
                color = color_map(matching_subplot[4])

                # Convert data coordinates to inset axis coordinates
                coordinates_inset = convert_to_inset_coordinates(self, [(coords, y)], self.axis_color_map)[0]

                color_map_data.append((coordinates_inset, color))

        self.data_color_map = color_map_data

        # Set the color map axis
        self.axis_color_map = self.axis_color_map

        # Plot the coordinates on the color map axis
        self.plot_on_color_map(self.data_color_map)

        self.axis_color_map.axis('off')  # Turn off axis ticks and labels

        # Refresh the plot to display the ROI on the color map axis
        plt.draw()

    def initialize_gui(self):
        """
        Initializes the GUI components for the EpsiGui.

        @author Benjamin (Ben) Yoon
        @date Fri, Sep 29, 2023
        @version 1.1
        """
        self.window, self.axis = plt.subplots(figsize=(15, 7.95))

        # Create a panel to contain sliders and buttons
        visualizer_items = plt.axes([0.7575, 0.11, 0.21, 0.77], frameon=True)
        visualizer_items.set_title("Visualizer Items")
        visualizer_items.xaxis.set_visible(False)
        visualizer_items.yaxis.set_visible(False)

        # Add sliders and buttons to the panel
        visualizer_items_y = 0.85
        visualizer_items_spacing = 0.2

        # Add text above the sliders and buttons
        text_above_buttons = "Image Slice"
        plt.text(0.8625, visualizer_items_y, text_above_buttons, horizontalalignment='center', fontsize=10,
                 transform=self.window.transFigure)

        # Create buttons
        self.button_left_image_slice = Button(plt.axes([0.784, visualizer_items_y - 0.065, 0.05, 0.05]), '←',
                                              color='black',
                                              hovercolor='0.6')
        self.button_right_image_slice = Button(plt.axes([0.892, visualizer_items_y - 0.065, 0.05, 0.05]), '→',
                                               color='black',
                                               hovercolor='0.6')
        self.text_element_image_slice = plt.text(-0.725, visualizer_items_y - 0.5, self.current_value_image_slice + 1,
                                                 fontsize=12, color='white')

        # Connect button callbacks
        self.button_left_image_slice.on_clicked(self.decrement_image_slice)
        self.button_right_image_slice.on_clicked(self.increment_image_slice)

        # Add text above the sliders and buttons
        text_above_buttons = "Contrast Level"
        plt.text(0.8625, visualizer_items_y - visualizer_items_spacing + 0.085, text_above_buttons,
                 horizontalalignment='center',
                 fontsize=10, transform=self.window.transFigure)

        # Create buttons
        self.button_left_contrast = Button(
            plt.axes([0.784, visualizer_items_y - visualizer_items_spacing + 0.025, 0.05, 0.05]), '←',
            color='black', hovercolor='0.6')
        self.button_right_contrast = Button(
            plt.axes([0.892, visualizer_items_y - visualizer_items_spacing + 0.025, 0.05, 0.05]), '→',
            color='black', hovercolor='0.6')
        self.text_element_contrast = plt.text(-0.675, visualizer_items_y - 0.5, self.current_value_contrast,
                                              fontsize=12,
                                              color='white')

        # Connect button callbacks
        self.button_left_contrast.on_clicked(self.decrement_contrast)
        self.button_right_contrast.on_clicked(self.increment_contrast)

        # Add text above the sliders and buttons
        text_above_buttons = "EPSI"
        plt.text(0.863, visualizer_items_y - visualizer_items_spacing - 0.03, text_above_buttons,
                 horizontalalignment='center',
                 fontsize=10, transform=self.window.transFigure)

        # Create buttons
        self.button_left_epsi = Button(
            plt.axes([0.784, visualizer_items_y - visualizer_items_spacing - 0.094, 0.05, 0.05]), '←',
            color='black', hovercolor='0.6')
        self.button_right_epsi = Button(
            plt.axes([0.892, visualizer_items_y - visualizer_items_spacing - 0.094, 0.05, 0.05]), '→',
            color='black', hovercolor='0.6')
        self.text_element_epsi = plt.text(-0.675, visualizer_items_y - 0.5, self.current_value_epsi, fontsize=12,
                                          color='white')

        # Connect button callbacks
        self.button_left_epsi.on_clicked(self.decrement_epsi)
        self.button_right_epsi.on_clicked(self.increment_epsi)

        # Add text above the buttons
        text_above_buttons = "View EPSI"
        plt.text(0.865, visualizer_items_y - visualizer_items_spacing - 0.15, text_above_buttons,
                 horizontalalignment='center',
                 fontsize=10, transform=self.window.transFigure)

        self.button_on_write_epsi = plt.Button(
            plt.axes([0.7665, visualizer_items_y - visualizer_items_spacing - 0.22, 0.09, 0.05]),
            'ON', color='black', hovercolor='0.6')
        self.button_off_write_epsi = plt.Button(
            plt.axes([0.8685, visualizer_items_y - visualizer_items_spacing - 0.22, 0.09, 0.05]),
            'OFF', color='black', hovercolor='0.6')

        # Connect button callback
        self.button_on_write_epsi.on_clicked(self.callback_epsi_button_on)
        self.button_off_write_epsi.on_clicked(self.callback_button_epsi_off)

        self.show_proton_picture()

        # Add the color map button
        self.button_color_map = plt.Button(
            plt.axes([0.77425, visualizer_items_y - visualizer_items_spacing + - 0.3175, 0.177,
                      0.05]), 'Colormap', color='black', hovercolor='0.6')
        self.button_color_map.on_clicked(self.on_clicked_button_color_map)

        # Add the refresh button
        self.button_reset = plt.Button(
            plt.axes([0.77425, visualizer_items_y - (visualizer_items_spacing * 3) + 0.01, 0.177, 0.05]),
            'RESET', color='black', hovercolor='0.6')
        self.button_reset.on_clicked(self.remove_axes)

        # Remove x and y axis labels
        self.axis.set_xticks([])
        self.axis.set_yticks([])

        # Create panels to contain zoomed EPSI subplots
        axis_a = plt.axes([0.032, 0.52, 0.24, 0.36], frameon=True)
        axis_a.set_title("EPSI Subplots")
        axis_a.xaxis.set_visible(False)
        axis_a.yaxis.set_visible(False)
        axis_b = plt.axes([0.032, 0.11, 0.24, 0.36], frameon=True)
        axis_b.xaxis.set_visible(False)
        axis_b.yaxis.set_visible(False)

        # Add tmp text in subplot panels
        text_a = "A"
        plt.text(0.065, 0.55, text_a, alpha=0.25, horizontalalignment='center', fontsize=75,
                 transform=self.window.transFigure)
        text_b = "B"
        plt.text(0.065, 0.15, text_b, alpha=0.25, horizontalalignment='center', fontsize=75,
                 transform=self.window.transFigure)

        # Show the plot
        plt.show()

    def decrement_image_slice(self, event):
        """
        Decreases the value of the image slice slider by one step.

        :param event: The event object generated by the slider interaction.

        @author Benjamin (Ben) Yoon
        @date Fri, Sep 29, 2023
        @version 1.1
        """
        if self.current_value_image_slice > 0:
            self.current_value_image_slice -= 1

            # Update the text element in the GUI
            self.text_element_image_slice.set_text(self.current_value_image_slice + 1)
            self.show_proton_picture()

    def increment_image_slice(self, event):
        """
        Increases the value of the image slice slider by one step.

        :param event: The event object generated by the slider interaction.

        @author Benjamin (Ben) Yoon
        @date Fri, Sep 29, 2023
        @version 1.1
        """
        if self.current_value_image_slice < (self.proton_pictures - 1):
            self.current_value_image_slice += 1

            # Update the text element in the GUI
            self.text_element_image_slice.set_text(self.current_value_image_slice + 1)
            self.show_proton_picture()

    def decrement_contrast(self, event):
        """
        Decreases the value of the contrast slider by one step.

        :param event: The event object generated by the slider interaction.

        @author Benjamin (Ben) Yoon
        @date Fri, Sep 29, 2023
        @version 1.1
        """
        if self.current_value_contrast > 0.2:
            self.current_value_contrast -= 0.2
        else:
            return

        rounded_value_contrast = "{:.1f}".format(self.current_value_contrast)

        # Update the text element in the GUI
        self.text_element_contrast.set_text(rounded_value_contrast)
        self.adjust_proton_picture_contrast()

    def increment_contrast(self, event):
        """
        Increases the value of the contrast slider by one step.

        :param event: The event object generated by the slider interaction.

        @author Benjamin (Ben) Yoon
        @date Fri, Sep 29, 2023
        @version 1.1
        """
        if self.current_value_contrast < 3:
            self.current_value_contrast += 0.2

        rounded_value_contrast = "{:.1f}".format(self.current_value_contrast)

        # Update the text element in the GUI
        self.text_element_contrast.set_text(rounded_value_contrast)
        self.adjust_proton_picture_contrast()

    def decrement_epsi(self, event):
        """
        Decreases the value of the EPSI slider by one step.

        :param event: The event object generated by the slider interaction.

        @author Benjamin (Ben) Yoon
        @date Fri, Sep 29, 2023
        @version 1.1
        """
        if self.current_value_epsi > 1:
            self.current_value_epsi -= 1

            # Update the text element in the GUI
            self.text_element_epsi.set_text(self.current_value_epsi)
            self.window.canvas.draw()
            self.process_epsi(None)

    def increment_epsi(self, event):
        """
        Increases the value of the EPSI slider by one step.

        :param event: The event object generated by the slider interaction.

        @author Benjamin (Ben) Yoon
        @date Fri, Sep 29, 2023
        @version 1.1
        """
        if self.current_value_epsi < self.epsi_data_sets:
            self.current_value_epsi += 1

            # Update the text element in the GUI
            self.text_element_epsi.set_text(self.current_value_epsi)
            self.window.canvas.draw()
            self.process_epsi(None)

    def on_clicked_button_color_map(self, event):
        """
        Callback function for the color map button click event.
        Initiates lasso selection for selecting a region of interest (ROI).

        @author Benjamin (Ben) Yoon
        @date Fri, Aug 11, 2023
        @version 1.0
        """
        if self.is_color_map_on is False:
            self.is_color_map_on = True
            read_epsi_plot(self)

            # Create a lasso selector for the axis
            roi = LassoSelector(self.axis, onselect=self.combine_coords)

            # Show the plot with the lasso selector active
            plt.show()

    def combine_coords(self, coords):
        """
        Finds interior points within the perimeter, and combines the perimeter and 
        interior coordinates.

        :param coords: (list) List of coordinates representing the perimeter of the ROI.

        @author Benjamin (Ben) Yoon
        @date Fri, Sep 29, 2023
        @version 1.1
        """
        # Find interior points within the perimeter
        coordinates_interior = get_interior_points(coords)

        # Combine perimeter and interior coordinates
        self.coordinates_total = coords + coordinates_interior

        self.process_color_map_and_plot(self.coordinates_total)

    def callback_epsi_button_on(self, event):
        """
        Callback function for the "Plot EPSI On" button.

        :param event: The event object generated by the button click.

        @author Benjamin (Ben) Yoon
        @date Mon, Aug 14, 2023
        @version 1.0
        """
        if self.is_write_epsi_on is False:
            self.is_write_epsi_on = True
            self.process_epsi(event)  # Call the epsi_show function

    def callback_button_epsi_off(self, event):
        """
        Callback function for the "Plot EPSI Off" button.

        :param event: The event object generated by the button click.

        @author Benjamin (Ben) Yoon
        @date Mon, Aug 14, 2023
        @version 1.0
        """
        if self.is_write_epsi_on is True:
            self.is_write_epsi_on = False
            self.process_epsi(event)  # Call the epsi_show function

    def remove_axes(self, event):
        """
        Callback function for the refresh button click event.
        Removes everything plotted on the axes.

        @author Benjamin (Ben) Yoon
        @date Fri, Sep 29, 2023
        @version 1.1
        """
        # Clear epsi_axis, plot_axis, and color_map_axis
        if self.is_write_epsi_on is True:
            try:
                self.axis_epsi_data.remove()
            except:
                pass
        if self.is_write_epsi_on is True:
            try:
                self.axis_plot.remove()
            except:
                pass
        if self.is_color_map_on is True:
            try:
                self.axis_color_map.remove()
            except:
                pass

        self.is_write_epsi_on = False
        self.is_color_map_on = False
        plt.draw()


if __name__ == "__main__":
    """
    @author: Benjamin (Ben) Yoon
    @date: Fri, Sep 29, 2023
    @version: 1.1
    Main execution block when this script is run directly.
    """
    viewer = EpsiGui(folder_path_dcm, folder_path_13c, folder_path_fid, epsi_plot_shift, epsi_information, columns,
                     rows, coordinates_axis_plot, image_slices, epsi_sets)
