# EPSI-GUI-in-Python
Developed a user-friendly Echo Planar Spectroscopic Imaging (EPSI) GUI program in Python for visualizing and interacting with medical imaging data (this is a more refined version of the EPSI GUI in MATLAB).

A graphical user interface (GUI) for visualizing and interacting with EPSI (Echo Planar Spectroscopic Imaging) data. This GUI allows users to display proton pictures, adjust contrast, plot EPSI data, and 
create color maps based on spectral values. It provides interactive tools for selecting regions of interest (ROIs) and generating color-coded maps to enhance the visualization of the data.

Author: Benjamin (Ben) Yoon
Date: Mon, Aug 14, 2023
Version: 1.0

## Features

- Display and navigate through proton pictures.
- Adjust contrast of displayed proton pictures.
- Plot EPSI data with interactive sliders.
- Create color maps based on spectral values.
- Select regions of interest (ROIs) using a lasso selector.

Author: Benjamin (Ben) Yoon
Date: Fri, Sep 29, 2023
Version: 1.1

## Features

- General debugging.
- Contrast adjustments are maintained during image slice iteration.
- All variables that users may want to change moved to the top of the file, base path added.
- Sliders removed, replaced with arrow buttons and updating text denoting the current value. 
- Robust system to ensure the removal of previous plots improves the speed of the program significantly.
- Increased color map generation speed by only calculating ROI coordinates once.
- Code restructured

Author: Benjamin (Ben) Yoon
Date: Fri, Oct 27, 2023
Version: 1.2

## Features

- Interactive file directory selection feature implemented

## Getting Started

### Prerequisites

- Python 3.10
- Required Python packages (install using `pip` or `conda`):
  - `matplotlib`
  - `numpy`
  - `pydicom`
  - `cv2`

- NOTE: INSTALLING PACKAGES CAN BE A HASTLE! I RECOMMEND USING THE VISUAL CODE STUDIO IDE FROM MICROSOFT AS IT HAS AN AUTOMATIC PACKAGE INSTALLATION FEATURE.

### Installation

1. 

### Instructions

- Navigate to the beginning of the EpsiGui.py file
  - Change the comment denoted fields accordingly
    
### Usage
  - Run the program (FileDialog.py)
  - Use buttons to adjust the proton picture, contrast, and EPSI dataset
  - Use on button to plot the EPSI data, and off button to remove
  - Use color map button to select a region of interest for the color map to be applied
  - Use reset button to remove all data visualization items

Acknowledgements
  - Author: Benjamin (Ben) Yoon, Penn Engineering Undergraduate
  - Contributor: Alexander (Shurik) Zavriyev, Penn Graduate Student
  - P.I. at PIGI Lab: Dr. Terence Gade
