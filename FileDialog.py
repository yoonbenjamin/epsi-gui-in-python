# @file FileDialog.py
# @package EpsiGui
# A graphical user interface (GUI) for selecting the base folder path for the EPSI (Echo Planar Spectroscopic Imaging) application.
#
# Version: 1.2
# This GUI allows users to interactively choose the base folder path where the EPSI application will operate. The selected
# folder will be used as the base directory for EPSI data and DICOM files.
#
# Author: Benjamin (Ben) Yoon
# Date: Fri, Oct 20, 2023
# Version: 1.2

import tkinter as tk
from tkinter import filedialog
from EpsiGui import EpsiGui

# Create a root window (Tkinter)
root = tk.Tk()
root.withdraw()  # Hide the root window

# Open a folder dialog using the system's file dialog
folder_path = filedialog.askdirectory()

# Ensure the selected path is not empty
if folder_path:
    print("Selected Folder:", folder_path)

    # Create an instance of your EpsiGui class and pass the base folder path
    my_epsi_gui = EpsiGui(folder_path)
else:
    print("No folder selected.")

# Ensure the Tkinter main loop is properly closed
root.destroy()
