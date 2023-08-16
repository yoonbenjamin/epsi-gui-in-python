import os
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
import pydicom
from matplotlib import colors as m_colors
from matplotlib.widgets import LassoSelector, Slider, Button
import cv2

from SpectralData import SpectralData

# Set the style to a dark theme
style.use('dark_background')


class EPSIGUI:
    """
    A graphical user interface (GUI) for visualizing and interacting with EPSI (Echo Planar Spectroscopic Imaging) data.

    This GUI allows users to display proton pictures, adjust contrast, plot EPSI data, and create color maps based on
    spectral values. It provides interactive tools for selecting regions of interest (ROIs) and generating color-coded
    maps to enhance the visualization of the data.

    Author: Benjamin (Ben) Yoon
    Date: Fri, Jul 21, 2023
    Version: 1.0

    Parameters:
        :param f_path_dcm: Path to the directory containing DICOM files for proton pictures (str).
        :param f_path_13c: Path to the directory containing EPSI data files (str).
        :param f_path_fid: Path to the directory containing fid data files (str).
        :param epsi_shift: A list of two shift values for the EPSI plot (list of float).
        :param epsi_info: A dictionary containing information about EPSI data (dict).
                          Example:
                          {
                              'pictures_to_read_write': 1,  # Number of pictures to read and write (int)
                              'proton': 60,  # Proton picture number (int)
                              'centric': 1  # Centric flag (int)
                          }
    """

    def __init__(self, f_path_dcm, f_path_13c, f_path_fid, epsi_shift, epsi_info):
        """
        Initializes the EPSIGUI class.

        Author: Benjamin (Ben) Yoon
        Date: Mon, Aug 14, 2023
        Version: 1.0

        Parameters:
            :param f_path_dcm: Path to the directory containing DICOM files for proton image slices (str).
            :param f_path_13c: Path to the directory containing EPSI data files (str).
            :param f_path_fid: Path to the directory containing fid data files (str).
            :param epsi_shift: A list of two shift values for the EPSI plot (list of float).
            :param epsi_info: A dictionary containing information about EPSI data (dict).
                              Example:
                              {
                                  'pictures_to_read_write': 1,  # Number of pictures to read and write (int)
                                  'proton': 60,  # Proton picture number (int)
                                  'centric': 1  # Centric flag (int)
                              }
        """
        self.image_slice_slider = None  # Slider for image slice iteration
        self.image_button_left = None  # Button for decreasing image slice slider
        self.image_slice_right = None  # Button for increasing image slice slider
        self.contrast_slider = None  # Slider for contrast
        self.contrast_button_left = None  # Button for decreasing contrast slider
        self.contrast_button_right = None  # Button for increasing contrast slider
        self.epsi_slider = None  # Slider for EPSI
        self.epsi_button_left = None  # Button for decreasing epsi slider
        self.epsi_button_right = None  # Button for increasing epsi slider
        self.epsi_button_off = None  # Button for EPSI viewing toggle on
        self.epsi_button_on = None  # Button for EPSI viewing toggle off
        self.show_epsi = False
        self.window = None
        self.axis = None
        self.plot_axis = None
        self.epsi_axis = None
        self.class_SpectralData_instance = SpectralData()
        self.path_dmc = f_path_dcm
        self.path_fid = f_path_fid
        self.path_13c = f_path_13c
        self.path_epsi = ""
        # self.picture_information = 1
        self.scale = True
        self.moving_average_window = 1
        self.proton_picture = None
        self.proton_picture_files = []
        for file_name in os.listdir(self.path_dmc):
            if file_name.endswith(".dcm"):
                self.proton_picture_files.append(os.path.join(self.path_dmc, file_name))
        self.proton_picture_files.sort()
        self.plot_shift = epsi_shift
        self.info_epsi = epsi_info
        self.lro_fid = None
        self.lpe_fid = None
        self.lro_epsi = None
        self.lpe_epsi = None
        self.x_epsi = None
        self.epsi = None
        self.spectral_data = None
        self.color_map_button = None  # Button for colormap
        self.color_map_axis = None
        self.reset_button = None  # Button for resetting plots on axes
        self.show_color_map = False
        self.x = None
        self.initialize_gui()

    def initialize_gui(self):
        """
        Initializes the GUI components for the EPSIGUI.

        Author: Benjamin (Ben) Yoon
        Date: Mon, Aug 16, 2023
        Version: 1.0
        """
        self.window, self.axis = plt.subplots(figsize=(15, 7.95))

        # Create a panel to contain sliders and buttons
        items_axis = plt.axes([0.7575, 0.11, 0.21, 0.77], frameon=True)
        items_axis.set_title("Visualizer Items")
        items_axis.xaxis.set_visible(False)
        items_axis.yaxis.set_visible(False)

        # Add sliders and buttons to the panel
        y_items_axis = 0.85
        slider_height = 0.1
        slider_spacing = 0.2

        # Add text above the sliders and buttons
        text_above_sliders = "Image Slice"
        plt.text(0.8625, y_items_axis, text_above_sliders, horizontalalignment='center', fontsize=10,
                 transform=self.window.transFigure)

        # Create slider
        self.image_slice_slider = Slider(
            plt.axes([0.795, y_items_axis - 0.088, 0.125, slider_height]), '', 0, len(self.proton_picture_files) - 1,
            valstep=1)

        # Create buttons
        self.image_button_left = Button(plt.axes([0.7625, y_items_axis - 0.065, 0.025, 0.05]), '←', color='black',
                                        hovercolor='0.975')
        self.image_slice_right = Button(plt.axes([0.939, y_items_axis - 0.065, 0.025, 0.05]), '→', color='black',
                                        hovercolor='0.975')

        # Connect button callbacks
        self.image_button_left.on_clicked(self.decrease_picture_slider)
        self.image_slice_right.on_clicked(self.increase_picture_slider)

        # Connect slider callback
        self.image_slice_slider.on_changed(self.show_proton_picture)

        # Add text above the sliders and buttons
        text_above_sliders = "Contrast Level"
        plt.text(0.8625, y_items_axis - slider_spacing + 0.085, text_above_sliders, horizontalalignment='center',
                 fontsize=10, transform=self.window.transFigure)

        # Create slider
        self.contrast_slider = plt.Slider(
            plt.axes([0.795, y_items_axis - slider_spacing, 0.123, slider_height]), '', 0.0, 2.0, valstep=0.1,
            valinit=1.0)

        # Create buttons
        self.contrast_button_left = Button(plt.axes([0.7625, y_items_axis - slider_spacing + 0.025, 0.025, 0.05]), '←',
                                           color='black', hovercolor='0.975')
        self.contrast_button_right = Button(plt.axes([0.939, y_items_axis - slider_spacing + 0.025, 0.025, 0.05]), '→',
                                            color='black', hovercolor='0.975')

        # Connect button callbacks
        self.contrast_button_left.on_clicked(self.decrease_contrast_slider)
        self.contrast_button_right.on_clicked(self.increase_contrast_slider)

        # Connect slider callback
        self.contrast_slider.on_changed(self.adjust_proton_picture_contrast)

        # Add text above the sliders and buttons
        text_above_sliders = "EPSI"
        plt.text(0.863, y_items_axis - slider_spacing - 0.03, text_above_sliders, horizontalalignment='center',
                 fontsize=10, transform=self.window.transFigure)

        # Create slider
        self.epsi_slider = Slider(plt.axes([0.795, y_items_axis - slider_spacing - 0.1175, 0.125, slider_height]), '',
                                  1, 17, valstep=1)

        # Create buttons
        self.epsi_button_left = Button(plt.axes([0.7625, y_items_axis - slider_spacing - 0.094, 0.025, 0.05]), '←',
                                       color='black', hovercolor='0.975')
        self.epsi_button_right = Button(plt.axes([0.939, y_items_axis - slider_spacing - 0.094, 0.025, 0.05]), '→',
                                        color='black', hovercolor='0.975')

        # Connect button callbacks
        self.epsi_button_left.on_clicked(self.decrease_epsi_slider)
        self.epsi_button_right.on_clicked(self.increase_epsi_slider)

        # Connect slider callback
        self.epsi_slider.on_changed(self.remove_previous_epsi_plot)

        # Add text above the buttons
        text_above_sliders = "View EPSI"
        plt.text(0.865, y_items_axis - slider_spacing - 0.15, text_above_sliders, horizontalalignment='center',
                 fontsize=10, transform=self.window.transFigure)

        self.epsi_button_on = plt.Button(plt.axes([0.7665, y_items_axis - slider_spacing - 0.22, 0.09, 0.05]),
                                         'ON', color='black')
        self.epsi_button_off = plt.Button(plt.axes([0.8685, y_items_axis - slider_spacing - 0.22, 0.09, 0.05]),
                                          'OFF', color='black')

        # Connect button callback
        self.epsi_button_on.on_clicked(self.on_button_callback)
        self.epsi_button_off.on_clicked(self.off_button_callback)

        self.show_proton_picture(0)

        # Add the color map button
        self.color_map_button = plt.Button(plt.axes([0.77425, y_items_axis - slider_spacing + - 0.3175, 0.177,
                                                     0.05]), 'Colormap', color='black')
        self.color_map_button.on_clicked(self.on_clicked_color_map)

        # Add the refresh button
        self.reset_button = plt.Button(plt.axes([0.77425, y_items_axis - (slider_spacing * 3) + 0.01, 0.177, 0.05]),
                                       'RESET', color='black')
        self.reset_button.on_clicked(self.remove_axes)

        # Remove x and y axis labels
        self.axis.set_xticks([])
        self.axis.set_yticks([])

        # Create panels to contain zoomed EPSI subplots
        a_axis = plt.axes([0.032, 0.52, 0.24, 0.36], frameon=True)
        a_axis.set_title("EPSI Subplots")
        a_axis.xaxis.set_visible(False)
        a_axis.yaxis.set_visible(False)
        b_axis = plt.axes([0.032, 0.11, 0.24, 0.36], frameon=True)
        b_axis.xaxis.set_visible(False)
        b_axis.yaxis.set_visible(False)

        # Add tmp text in subplot panels
        text_a = "A"
        plt.text(0.065, 0.55, text_a, alpha=0.25, horizontalalignment='center', fontsize=75,
                 transform=self.window.transFigure)
        text_b = "B"
        plt.text(0.065, 0.15, text_b, alpha=0.25, horizontalalignment='center', fontsize=75,
                 transform=self.window.transFigure)

        # Show the plot
        plt.show()

    def decrease_picture_slider(self, event):
        """
        Decreases the value of the image slice slider by one step.

        :param event: The event object generated by the slider interaction.

        Author: Benjamin (Ben) Yoon
        Date: Mon, Aug 14, 2023
        Version: 1.0
        """
        current_value = self.image_slice_slider.val
        step = self.image_slice_slider.valstep
        self.image_slice_slider.set_val(max(current_value - step, self.image_slice_slider.valmin))

    def increase_picture_slider(self, event):
        """
        Increases the value of the image slice slider by one step.

        :param event: The event object generated by the slider interaction.

        Author: Benjamin (Ben) Yoon
        Date: Mon, Aug 14, 2023
        Version: 1.0
        """
        current_value = self.image_slice_slider.val
        step = self.image_slice_slider.valstep
        self.image_slice_slider.set_val(min(current_value + step, self.image_slice_slider.valmax))

    def show_proton_picture(self, proton_picture_index):
        """
        Displays a proton picture corresponding to the given index.

        :param proton_picture_index: Index of the proton picture to display (int).

        Author: Benjamin (Ben) Yoon
        Date: Fri, Jul 21, 2023
        Version: 1.0
        """
        proton_picture_index = int(proton_picture_index)
        proton_picture_file_path = self.proton_picture_files[proton_picture_index]
        dcm_read = pydicom.dcmread(proton_picture_file_path)
        self.proton_picture = dcm_read.pixel_array
        self.axis.imshow(self.proton_picture, cmap='gray')
        self.window.canvas.draw()

    def decrease_contrast_slider(self, event):
        """
        Decreases the value of the contrast slider by one step.

        :param event: The event object generated by the slider interaction.

        Author: Benjamin (Ben) Yoon
        Date: Mon, Aug 14, 2023
        Version: 1.0
        """
        current_value = self.contrast_slider.val
        step = self.contrast_slider.valstep
        self.contrast_slider.set_val(max(current_value - step, self.contrast_slider.valmin))

    def increase_contrast_slider(self, event):
        """
        Increases the value of the contrast slider by one step.

        :param event: The event object generated by the slider interaction.

        Author: Benjamin (Ben) Yoon
        Date: Mon, Aug 14, 2023
        Version: 1.0
        """
        current_value = self.contrast_slider.val
        step = self.contrast_slider.valstep
        self.contrast_slider.set_val(min(current_value + step, self.contrast_slider.valmax))

    def adjust_proton_picture_contrast(self, value):
        """
        Adjusts the contrast of the displayed proton picture using Contrast Limited Adaptive Histogram Equalization
        (CLAHE).

        :param value: The clip limit used for contrast adjustment (float).

        Author: Benjamin (Ben) Yoon
        Date: Mon, Aug 14, 2023
        Version: 1.0
        """
        # Normalize pixel values to [0, 1]
        normalized_img = (self.proton_picture - np.min(self.proton_picture)) / (
                np.max(self.proton_picture) - np.min(self.proton_picture))

        # Define contrast value (received as input parameter)
        contrast_value = value

        # Apply CLAHE for contrast adjustment
        clahe = cv2.createCLAHE(clipLimit=contrast_value, tileGridSize=(8, 8))
        clahe_img = clahe.apply(np.uint8(normalized_img * 255))

        # Rescale pixel values back to [0, 1]
        rescaled_img = clahe_img / 255.0

        self.axis.imshow(rescaled_img, cmap='gray')
        self.window.canvas.draw()

    def decrease_epsi_slider(self, event):
        """
        Decreases the value of the EPSI slider by one step.

        :param event: The event object generated by the slider interaction.

        Author: Benjamin (Ben) Yoon
        Date: Mon, Aug 14, 2023
        Version: 1.0
        """
        current_value = self.epsi_slider.val
        step = self.epsi_slider.valstep
        self.epsi_slider.set_val(max(current_value - step, self.epsi_slider.valmin))

    def increase_epsi_slider(self, event):
        """
        Increases the value of the EPSI slider by one step.

        :param event: The event object generated by the slider interaction.

        Author: Benjamin (Ben) Yoon
        Date: Mon, Aug 14, 2023
        Version: 1.0
        """
        current_value = self.epsi_slider.val
        step = self.epsi_slider.valstep
        self.epsi_slider.set_val(min(current_value + step, self.epsi_slider.valmax))

    def remove_previous_epsi_plot(self, value):
        """
        Removes the previous EPSI plot when the EPSI slider changes.

        :param value: The new value of the EPSI slider (float).

        Author: Benjamin (Ben) Yoon
        Date: Fri, Jul 21, 2023
        Version: 1.0
        """
        # Clear and replot epsi_axis and grid_axis whenever the EPSI slider changes
        if self.show_epsi:
            if self.epsi_axis:
                self.epsi_axis.remove()
            if self.plot_axis:
                self.plot_axis.remove()
        self.write_epsi_plot(None)

    # Define the callback function for the plot epsi on button (Yoon, 2023, v1.0)
    def on_button_callback(self, event):
        self.epsi_show(event, show=True)  # Call the epsi_show function with the show parameter set to True

    # Define the callback function for the plot epsi off button (Yoon, 2023, v1.0)
    def off_button_callback(self, event):
        self.epsi_show(event, show=False)  # Call the epsi_show function with the show parameter set to False

    def epsi_show(self, event, show):
        """
        Initiates the display of the EPSI plot when the "Plot EPSI" button is clicked.

        :param event: The event generated by clicking the "Plot EPSI" button.
        :param show: A boolean value indicating whether to show or hide the EPSI plot.

        Author: Benjamin (Ben) Yoon
        Date: Mon, Aug 14, 2023
        Version: 1.0
        """
        self.show_epsi = show
        if show:
            self.write_epsi_plot(event)
        else:
            self.epsi_axis.remove()
            self.plot_axis.remove()
            self.window.canvas.draw()

    def set_epsi(self):
        """
        Set the EPSI data based on configuration and slider value.

        Author: Benjamin (Ben) Yoon
        Date: Fri, Aug 11, 2023
        Version: 1.0
        """
        # Constants
        proton_quarter = self.info_epsi['proton'] / 4

        # Read EPSI data based on the EPSI slider value
        self.path_epsi = f"{self.path_13c}{self.epsi_slider.val:02d}"
        spectral_data = self.class_SpectralData_instance.read_write_spectral_data(self.info_epsi, self.path_epsi,
                                                                                  proton_quarter)
        # Preprocessing
        spectral_data = np.flip(np.flip(spectral_data, 0), 1)
        # if self.picture_information:
        #    spectral_data = SpectralData.correct_epsi_plot(self, spectral_data)
        if self.scale:
            maximum_spectral_data_value = np.max(spectral_data)
            spectral_data = spectral_data / maximum_spectral_data_value
        else:
            maximum_spectral_data_value = np.max(spectral_data, axis=2)
            spectral_data = spectral_data / maximum_spectral_data_value
        columns = 16
        rows = 12
        epsi = []
        for i in range(0, rows):
            row_information = []
            for j in range(0, columns):
                if np.max(spectral_data[i, j, :]) < 0.20:
                    spectral_data[i, j, :] = np.nan
                row_information = np.concatenate(
                    (np.squeeze(row_information), np.squeeze(np.roll(spectral_data[i, j, :], 0))))
            epsi = np.concatenate((np.squeeze(epsi), np.squeeze(row_information + rows - i)))
        x_epsi = np.tile(np.arange(0, spectral_data.shape[2] * columns), rows)
        for nan_rows in range(0, rows - 1):
            epsi[nan_rows * spectral_data.shape[2] * columns] = np.nan
        epsi = np.convolve(epsi, np.ones(self.moving_average_window), mode='same') / self.moving_average_window
        epsi[~np.isnan(epsi)] -= 1

        # Adjust subplot position
        self.lro_fid = self.class_SpectralData_instance.read_write_procpar('lro', self.path_fid)[0] * 10
        self.lpe_fid = self.class_SpectralData_instance.read_write_procpar('lpe 1', self.path_fid)[0] * 10
        self.lro_epsi = self.class_SpectralData_instance.read_write_procpar('lro', self.path_epsi)[0] * 10
        self.lpe_epsi = self.class_SpectralData_instance.read_write_procpar('lpe 1', self.path_epsi)[0] * 10

        self.x_epsi = x_epsi
        self.epsi = epsi
        self.spectral_data = spectral_data

    def write_epsi_plot(self, event):
        """
        Generates and displays the EPSI plot while updating the colormap based on the EPSI dataset, if plotted.

        :param event: The event generated for generating the EPSI plot.

        Author: Benjamin Yoon
        Date: Mon, Aug 14, 2023
        Version: 1.0
        """
        self.set_epsi()
        if self.show_color_map:
            self.color_map_axis.remove()
            self.onselect_roi(self.x)
        columns = 16
        rows = 12
        if self.show_epsi:
            epsi_axis = self.axis.inset_axes(
                [((self.lro_fid - self.lro_epsi) / 2 + self.plot_shift[0] * self.lro_epsi / columns) /
                 self.lro_fid, 1 - ((self.lpe_fid - self.lpe_epsi) / 2 + rows * self.lpe_epsi / rows -
                                    self.plot_shift[1] * self.lpe_epsi / rows) / self.lpe_fid,
                 self.lro_epsi / self.lro_fid, self.lpe_epsi / self.lpe_fid])
            plot_axis = self.axis.inset_axes(
                [((self.lro_fid - self.lro_epsi) / 2 + self.plot_shift[0] * self.lro_epsi / columns) /
                 self.lro_fid, ((self.lpe_fid - self.lpe_epsi) / 2 + self.plot_shift[1] * self.lpe_epsi / rows) /
                 self.lpe_fid, self.lro_epsi / self.lro_fid, self.lpe_epsi / self.lpe_fid])
            self.epsi_axis = epsi_axis
            self.plot_axis = plot_axis
            epsi_axis.plot(self.x_epsi, np.squeeze(self.epsi), color='#FF00FF', linewidth=1)
            epsi_axis.set_ylim([0, rows])
            epsi_axis.set_xlim([0, columns * self.spectral_data.shape[2]])
            epsi_axis.axis('off')
            for i in range(0, columns + 1):
                plot_axis.axvline(x=i, linestyle='--', color='w', linewidth=1, alpha=0.5)
            for j in range(0, rows + 1):
                plot_axis.axhline(y=j, linestyle='--', color='w', linewidth=1, alpha=0.5)
            plot_axis.set_ylim([0, rows + 0.05])
            plot_axis.set_xlim([0, columns + 0.05])
            plot_axis.axis('off')
            self.window.canvas.draw()

    def on_clicked_color_map(self, event):
        """
        Callback function for the color map button click event.
        Initiates lasso selection for selecting a region of interest (ROI).

        Author: Benjamin (Ben) Yoon
        Date: Fri, Aug 11, 2023
        Version: 1.0
        """
        self.show_color_map = True
        self.set_epsi()
        # Create a lasso selector for the axis
        roi = LassoSelector(self.axis, onselect=self.onselect_roi)
        # Show the plot with the lasso selector active
        plt.show()

    def onselect_roi(self, x):
        """
        Callback function for the ROI (Region of Interest) selection event using the lasso selector.
        Gets all coordinates to plot, translates data coordinates into inset axis coordinates, make a color map for the
        ROI.
        for subsequently drawing the color map.

        :param x: (list) List of coordinates representing the perimeter of the ROI.

        Author: Benjamin (Ben) Yoon
        Date: Fri, Aug 11, 2023
        Version: 1.0
        """
        self.x = x
        columns = 16
        rows = 12

        # Create a color map inset axis within the axis
        color_map_axis = self.axis.inset_axes(
            [((self.lro_fid - self.lro_epsi) / 2 + self.plot_shift[0] * self.lro_epsi / columns) /
             self.lro_fid, ((self.lpe_fid - self.lpe_epsi) / 2 + self.plot_shift[1] * self.lpe_epsi / rows) /
             self.lpe_fid, self.lro_epsi / self.lro_fid, self.lpe_epsi / self.lpe_fid])

        # Find interior points within the perimeter
        interior_coords = self.find_interior_points(x)

        # Combine perimeter and interior coordinates
        total_coords = x + interior_coords

        # Load subplot information: [x_min, x_max, y_min, y_max, max_spectral_value]
        plot_axis_coords = (14, 233, 51, 215)  # plot axis coords
        num_rows = 12
        num_cols = 16
        subplot_info = self.calculate_subplot_info(plot_axis_coords, num_rows, num_cols)

        # Create a color map gradient
        color_map_gradient = []
        for info in subplot_info:
            max_value = info[4]
            normalized_value = max_value / np.nanmax(self.epsi)
            rgba_color = (0.0, 0.0, normalized_value, normalized_value)
            color_map_gradient.append(rgba_color)

        # Initialize the array for data and color mapping
        data_color_map = []

        # Define RGB values for the color points
        color_points = [
            (0.0, (0.11, 0.435, 0.973)),  # Blue
            (0.2, (0.153, 0.733, 0.878)),  # Battery Charged Blue
            (0.4, (0.192, 0.859, 0.573)),  # Eucalyptus
            (0.6, (0.106, 0.945, 0.094)),  # Neon Green
            (0.8, (0.608, 0.980, 0.141)),  # Green Lizard
            (1.0, (0.996, 0.969, 0.125))  # Yellow
        ]

        # Define the custom colormap with RGB color points
        cmap = m_colors.LinearSegmentedColormap.from_list('custom_colormap', color_points)

        # Assign colors based on max spectral values
        for coord in total_coords:
            x, y = coord
            matching_subplot = None

            for info in subplot_info:
                x_min, x_max, y_min, y_max, _ = info
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    matching_subplot = info
                    break

            if matching_subplot:
                color = cmap(matching_subplot[4])

                # Convert data coordinates to inset axis coordinates
                inset_coord = self.convert_to_inset_coords([(x, y)], color_map_axis)[0]

                data_color_map.append((inset_coord, color))

        # Set the color map axis
        self.color_map_axis = color_map_axis

        # Plot the coordinates on the color map axis
        self.plot_on_color_map(data_color_map)

        color_map_axis.axis('off')  # Turn off axis ticks and labels

        # Refresh the plot to display the ROI on the color map axis
        plt.draw()

    @staticmethod
    def find_interior_points(perimeter_coords):
        """
        Finds interior points within the given perimeter using a grid-based approach.

        :param perimeter_coords: (list) List of perimeter coordinates.
        :return: (list) List of interior coordinates.

        Author: Benjamin (Ben) Yoon
        Date: Fri, Aug 11, 2023
        Version: 1.0
        """
        # Import required libraries
        from shapely.geometry import Polygon, Point

        # Create a shapely polygon from the perimeter coordinates
        polygon = Polygon(perimeter_coords)

        # Determine whether the shape is closed by checking if the first and last points are the same
        is_closed_shape = perimeter_coords[0] == perimeter_coords[-1]

        # Get bounding box coordinates
        min_x, min_y = min(perimeter_coords, key=lambda item: item[0])[0], \
            min(perimeter_coords, key=lambda item: item[1])[1]
        max_x, max_y = max(perimeter_coords, key=lambda item: item[0])[0], \
            max(perimeter_coords, key=lambda item: item[1])[1]

        # Define grid spacing
        grid_spacing = 0.5

        # Generate interior points using a grid-based approach
        interior_coords = []
        for x in np.arange(min_x, max_x + grid_spacing, grid_spacing):
            for y in np.arange(min_y, max_y + grid_spacing, grid_spacing):
                point = Point(x, y)

                # Check if the point falls within the polygon
                if point.within(polygon):
                    interior_coords.append((x, y))

        # If the shape is not closed, connect the first and last points
        if not is_closed_shape:
            interior_coords.append(interior_coords[0])

        return interior_coords

    def calculate_subplot_info(self, plot_axis_coords, num_rows, num_cols):
        """
        Calculates information for each subplot based on the given plot axis coordinates.

        :param plot_axis_coords: (tuple) The coordinates of the main plot axis (x_min, x_max, y_min, y_max).
        :param num_rows: (int) Number of rows in the subplot grid.
        :param num_cols: (int) Number of columns in the subplot grid.

        :return: (list) List of lists containing subplot information: [x_min, x_max, y_min, y_max, max_spectral_value].

        Author: Benjamin (Ben) Yoon
        Date: Fri, Aug 11, 2023
        Version: 1.0
        """
        x_min_main, x_max_main, y_min_main, y_max_main = plot_axis_coords
        x_range = x_max_main - x_min_main
        y_range = y_max_main - y_min_main
        x_step = x_range / num_cols
        y_step = y_range / num_rows

        subplot_info = []

        for row in range(num_rows):
            for col in range(num_cols):
                x_min_subplot = x_min_main + col * x_step
                x_max_subplot = x_min_main + (col + 1) * x_step
                y_min_subplot = y_min_main + row * y_step
                y_max_subplot = y_min_main + (row + 1) * y_step

                with np.errstate(all='ignore'):
                    # Extract max spectral value within subplot boundaries
                    max_spectral_value = np.nanmax(self.spectral_data[row, col, :])

                    if np.isnan(max_spectral_value):
                        max_spectral_value = 0.0  # Default value for nan max values

                subplot_info.append([x_min_subplot, x_max_subplot, y_min_subplot, y_max_subplot, max_spectral_value])

        return subplot_info

    def convert_to_inset_coords(self, total_coords, axis):
        """
        Converts a list of data coordinates to inset axis coordinates.

        :param total_coords: (list) List of data coordinates.
        :param axis: (Axes) Inset axis for the color map.

        :return: (list) List of inset axis coordinates.

        Author: Benjamin (Ben) Yoon
        Date: Fri, Aug 11, 2023
        Version: 1.0
        """
        inset_coords = []
        for x, y in total_coords:
            data_to_color_map_transform = self.axis.transData + axis.transData.inverted()
            color_map_coords = data_to_color_map_transform.transform([(x, y)])[0]
            inset_coords.append(color_map_coords)

        return inset_coords

    def plot_on_color_map(self, data_color_map):
        """
        Plots the color map with data points and corresponding colors.

        :param data_color_map: (list) A list of tuples where each tuple contains a data coordinate and its associated
        RGB color.

        Author: Benjamin (Ben) Yoon
        Date: Fri, Aug 11, 2023
        Version: 1.0
        """
        # Save current axis limits
        x_limit = self.color_map_axis.get_xlim()
        y_limit = self.color_map_axis.get_ylim()

        # Extract x and y coordinates and corresponding colors from data_color_map
        x_coords = [coord[0] for coord, _ in data_color_map]
        y_coords = [coord[1] for coord, _ in data_color_map]
        colors = [rgb_color for _, rgb_color in data_color_map]

        # Scatter plot of data points with specified colors and transparency
        self.color_map_axis.scatter(x_coords, y_coords, c=colors, marker='o', alpha=0.01)

        # Restore original axis limits
        self.color_map_axis.set_xlim(x_limit)
        self.color_map_axis.set_ylim(y_limit)

        # Turn off axis ticks and labels
        self.color_map_axis.axis('off')

        # Refresh the plot to display the color map on the axis
        plt.draw()

    def remove_axes(self, event):
        """
        Callback function for the colormap button click event.
        Removes everything plotted on the axes.

        Author: Benjamin (Ben) Yoon
        Date: Mon, Aug 14, 2023
        Version: 1.0
        """
        print("button works")
        # Clear epsi_axis, plot_axis, and color_map_axis
        self.epsi_axis.remove()
        self.plot_axis.remove()
        self.color_map_axis.remove()
        plt.draw()


if __name__ == "__main__":
    """
    Author: Benjamin Yoon
    Date: Fri, Jul 21, 2023
    Version: 1.0
        Main execution block when this script is run directly.
    """
    folder_path_dcm = "/Users/benjaminyoon/Desktop/PIGI folder/Projects/Project2 EPSI GUI in Python/EPSI " \
                      "GUI/data_mouse_kidney/s_2023041103/fsems_rat_liver_03.dmc/"
    folder_path_13c = "/Users/benjaminyoon/Desktop/PIGI folder/Projects/Project2 EPSI GUI in Python/EPSI " \
                      "GUI/data_mouse_kidney/s_2023041103/epsi_16x12_13c_"
    folder_path_fid = "/Users/benjaminyoon/Desktop/PIGI folder/Projects/Project2 EPSI GUI in Python/EPSI " \
                      "GUI/data_mouse_kidney/s_2023041103/fsems_rat_liver_03"
    epsi_plot_shift = [-0.3, -0.4]
    epsi_information = {
        'pictures_to_read_write': 1,
        'proton': 60,
        'centric': 1
    }
    viewer = EPSIGUI(folder_path_dcm, folder_path_13c, folder_path_fid, epsi_plot_shift, epsi_information)
