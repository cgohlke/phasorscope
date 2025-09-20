#!/usr/bin/env python
# coding: utf-8

#
# # PhasorAnalysis.ipynb
#
# Based on version 2.1 (2025.05.14).
#
# Original Copyright (c) 2023-2025 Lorenzo Scipioni.
#
# This file has been modified by Christoph Gohlke.
#
# All modifications are licensed under the MIT License:
#
# ```
# MIT License
#
# Copyright (c) 2025 Christoph Gohlke
# Copyright (c) 2023-2025 Lorenzo Scipioni
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ```

# ## Introduction
#
# This is a code for processing sine-cosine spectral data developed for bioluminescence phasors on the Phasor Scope.
#
# ---
#
# By *Lorenzo Scipioni*
#
# ---
#
# Files needed (tiff format):
# 1.   3-channel images (INT - COS - SIN)
# 2.   Bright Calibration
# 3.   Dark Calibration
# 4.   Registration file

# ### Flags
#
# **FLAG_SAVE_IMAGES:**
#
# Set it to **True** if you want the data to be saved (e.g., overtime exp, BRET efficiency and other experiments that need postprocessing), otherwise set it to **False** (saves time and storage, processed files are large)
#
# **FLAG_REGISTRATION_PRECALCULATE:**
#
# Set it to **True** if you want the registration matrix to be calculated on the first image in the folder and applied to the rest (more stable between files, better for dim samples but register on the brightest). Set it to **False** if you want the registration to be recalculated for each file.
#
# ### Calibration Paths
#
# Copy-paste here the path to the calibration **FOLDER** (files in .tif format from your Google Drive)
#
# To do so, right-click on the file or folder from the menu on the left and select "Copy path", then paste it between apostrophes (' ') below.
#
# ### Experiment Analysis
#
# Copy-paste here the path to the experiment **FOLDER** (containing the .tif files arranged in **FOLDERS** of the experiments you want to analyze.)
#
# To do so, right-click on the file or folder from the menu on the left and select "Copy path", then paste it between apostrophes (' ') below.

# In[ ]:


# @title **Automatic - Install and import libraries**

try:
    print("Mounting Google Drive...")
    from google.colab import drive

    drive.mount("/content/drive", force_remount=True)
    print("Installing libraries...")
    get_ipython().run_line_magic("pip", "install cellpose")
    get_ipython().run_line_magic("pip", "install phasorpy==0.7")
except ImportError:
    pass

print("Import libraries...")
import math
import os
import time
import traceback

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import phasorpy
import scipy
import skimage
import tifffile
from cellpose import models
from phasorpy.color import CATEGORICAL
from phasorpy.cursor import mask_from_circular_cursor, pseudo_color
from phasorpy.plot import PhasorPlot
from scipy.ndimage import gaussian_filter
from skimage.filters import median
from skimage.morphology import disk

time_started = time.perf_counter()

# output additional information throughout the notebook
DEBUG = os.environ.get("PA_DEBUG", "1") in {"1", "TRUE", "True"}

if DEBUG:
    print(f"numpy {np.__version__}")
    print(f"scipy {scipy.__version__}")
    print(f"pandas {pd.__version__}")
    print(f"skimage {skimage.__version__}")
    print(f"matplotlib {matplotlib.__version__}")
    print(f"phasorpy {phasorpy.__version__}")
    print(f"tifffile {tifffile.__version__}")


# In[ ]:


# @title **Automatic - Functions definition**

print("Defining functions...")

# Resolution of figures
DPI = float(os.environ.get("PA_DPI", 300))

# Image interpolation
INTERPOLATION = os.environ.get("PA_INTERPOLATION", "nearest")

# Figure aspect ratio
ASPECT = 1.0

# colormap for debug images
CMAP = plt.cm.viridis
CMAP.set_under("red")  # Color for values below vmin
CMAP.set_over("blue")  # Color for values above vmax


def get_labeled_ROIs(img_bright, thr, n_regions=3):
    # this function gets the region of the two largest ROIs above a defined
    # threshold, sorted by x-position (left to right)
    labeled_image, num_labels = scipy.ndimage.label(img_bright > thr)
    regions = scipy.ndimage.find_objects(labeled_image)
    region_sizes = [np.sum(labeled_image[region]) for region in regions]
    sorted_regions = sorted(zip(region_sizes, regions), reverse=True)
    largest_regions = sorted_regions[:n_regions]
    region_ROIs_start = [region[1][1].start for region in largest_regions]
    largest_regions = [largest_regions[i] for i in np.argsort(region_ROIs_start)]
    new_labeled_image = np.zeros_like(labeled_image)
    for i, (_, region) in enumerate(largest_regions, start=1):
        new_labeled_image[region] = i
    return new_labeled_image, largest_regions


def Apply_Calibration(img, Calibration, plot=DEBUG, title="DEBUG"):
    dark = Calibration["Value Calibration_Dark"]

    if plot:
        # Plot mean of images used for calibration
        fig, axs = plt.subplots(2, 3, figsize=(10.24 * ASPECT, 7.68), dpi=DPI)
        fig.suptitle(f"{title} - Calibration ({dark=:.1f})")

        ch0 = img[Calibration["Slices"][0]]
        if ch0.ndim == 3:
            ch0 = ch0.mean(axis=2)
        ch1 = img[Calibration["Slices"][1]]
        if ch1.ndim == 3:
            ch1 = ch1.mean(axis=2)
        ch2 = img[Calibration["Slices"][2]]
        if ch2.ndim == 3:
            ch2 = ch2.mean(axis=2)
        vmax = None  # max(ch0.max(), ch1.max(), ch2.max()) - 1e-3

        ax = axs[0, 0]
        ax.set_title("CH 0")
        # ax.set_axis_off()
        im = ax.imshow(ch0, vmin=dark, vmax=vmax, cmap=CMAP, interpolation=INTERPOLATION)
        if vmax is None:
            plt.colorbar(im)

        ax = axs[0, 1]
        ax.set_title("CH 1")
        ax.set_axis_off()
        im = ax.imshow(ch1, vmin=dark, vmax=vmax, cmap=CMAP, interpolation=INTERPOLATION)
        if vmax is None:
            plt.colorbar(im)

        ax = axs[0, 2]
        ax.set_title("CH 2")
        ax.set_axis_off()
        im = ax.imshow(ch2, vmin=dark, vmax=vmax, cmap=CMAP, interpolation=INTERPOLATION)
        plt.colorbar(im)

        ax = axs[1, 0]
        ax.set_axis_off()

        ax = axs[1, 1]
        ax.set_title("Bright 1")
        ax.set_axis_off()
        im = ax.imshow(Calibration["Bright"][0], interpolation=INTERPOLATION)
        plt.colorbar(im)

        ax = axs[1, 2]
        ax.set_title("Bright 2")
        ax.set_axis_off()
        im = ax.imshow(Calibration["Bright"][1], interpolation=INTERPOLATION)
        plt.colorbar(im)

        plt.tight_layout()
        plt.show()

    CH = [img[Slice] - dark for Slice in Calibration["Slices"]]
    if img.ndim == 3:
        CH[1] = CH[1] / Calibration["Bright"][0][:, :, None]
        CH[2] = CH[2] / Calibration["Bright"][1][:, :, None]
    else:
        CH[1] = CH[1] / Calibration["Bright"][0]
        CH[2] = CH[2] / Calibration["Bright"][1]
    return CH


def Process_Img(img, Processing, ComputeMask=True, plot=DEBUG, title="DEBUG"):
    if Processing["Bkg_subtraction"] != 0:
        background = skimage.filters.gaussian(img, Processing["Bkg_subtraction"])
        img_processed = img - background
    else:
        background = None
        img_processed = img

    if Processing["Median_filter"] > 0:
        img_processed = median(img_processed, disk(Processing["Median_filter"]))

    img_thresholded = img_processed.copy()
    img_thresholded[img_thresholded < 0] = 0.0

    if ComputeMask:
        masks = Processing["Cellpose_model"].eval(img_thresholded, diameter=Processing["Cellpose_diameter"])[0]
    else:
        masks = []

    if plot:
        # Plot images used for segmentation
        fig, axs = plt.subplots(1, 4, figsize=(12.8 * ASPECT, 4.8), dpi=DPI)
        fig.suptitle(f"{title} - Segmentation")

        ax = axs[0]
        ax.set_title("Intensity")
        ax.set_axis_off()
        im = ax.imshow(img, vmin=0, cmap=CMAP, interpolation=INTERPOLATION)
        plt.colorbar(im)

        ax = axs[1]
        ax.set_title("Background")
        ax.set_axis_off()
        if background is not None:
            im = ax.imshow(background, cmap=CMAP, interpolation=INTERPOLATION)
            plt.colorbar(im)

        ax = axs[2]
        ax.set_title("Processed")
        ax.set_axis_off()
        im = ax.imshow(img_processed, vmin=0, cmap=CMAP, interpolation=INTERPOLATION)
        plt.colorbar(im)

        ax = axs[3]
        ax.set_title("Masks")
        ax.set_axis_off()
        im = ax.imshow(masks, vmin=0, cmap="nipy_spectral", interpolation=INTERPOLATION)
        plt.colorbar(im)

        plt.tight_layout()
        plt.show()

    return img_processed, masks


def plot_grid(ax, radii=[0, 0.25, 0.5, 0.75, 1], angles=np.arange(0, 360, 45), color="white"):
    for radius in radii:
        circle = plt.Circle((0, 0), radius, fill=False, linestyle="-", color=color)
        ax.add_artist(circle)
    for angle in angles:
        x = [0, np.cos(np.deg2rad(angle))]
        y = [0, np.sin(np.deg2rad(angle))]
        ax.plot(x, y, linestyle="--", color=color)


def preprocess(folder_path, radius=None, sigma=None):
    files = [f for f in os.listdir(folder_path) if os.path.splitext(f)[-1].lower() == ".tif"]
    if not files:
        print(f"Error: No '.tif' files found in {folder_path!r}")
        return None  # Explicitly return None to signal an error
    files = tifffile.natural_sorted(files)
    img = [tifffile.imread(os.path.join(folder_path, f)).astype(np.float32) for f in files]
    if radius is not None and radius > 0:
        print(f"Removing hot pixels with median filter {radius=} ...")
        disk_element = disk(radius)
        for i, im in enumerate(img):
            im = median(im, disk_element, mode="reflect")
            img[i] = im
    if sigma is not None and sigma > 0:
        print(f"Applying gaussian smoothing {sigma=} ...")
        for i, im in enumerate(img):
            im = gaussian_filter(im, sigma=sigma)
            img[i] = im
    stack = np.dstack(img)
    if DEBUG:
        print(f"{stack.shape=}, {stack.dtype=}")
    return stack


def norm_slicing(tld, img, dark, region=3, plot=False):
    print("Automatically selecting ROIs...")

    fig, axs = plt.subplots(nrows=1, ncols=3, dpi=DPI, figsize=(12.8, 4.8))
    axs[0].imshow(img, cmap="gray", interpolation=INTERPOLATION)
    axs[0].set_title("Original")
    binary_mask = img > tld
    axs[1].imshow(binary_mask, cmap="gray", interpolation=INTERPOLATION)
    axs[1].set_title("Manual threshold")
    axs[1].set_axis_off()
    labeled_rois_img, rois_info = get_labeled_ROIs(img, tld, region)
    axs[2].imshow(labeled_rois_img, cmap="viridis", interpolation=INTERPOLATION)
    axs[2].set_title("Detected ROIs")
    axs[2].set_axis_off()
    plt.tight_layout()
    plt.show()

    print("Channel Ratios - Define common ROIs...")
    roi_slices = [roi[1] for roi in rois_info]
    row_slices = [sl[0] for sl in roi_slices]
    col_slices = [sl[1] for sl in roi_slices]

    dL_R = min(sl.stop - sl.start for sl in row_slices)
    dL_C = min(sl.stop - sl.start for sl in col_slices)
    CH1_slice, CH2_slice, CH3_slice = (
        (slice(row.start, row.start + dL_R), slice(col.start, col.start + dL_C))
        for row, col in zip(row_slices, col_slices)
    )

    CH1_ROI = img[CH1_slice] - dark
    CH2_ROI = img[CH2_slice] - dark
    CH3_ROI = img[CH3_slice] - dark

    if plot:
        fig, axs = plt.subplots(nrows=1, ncols=3, dpi=DPI, figsize=(6.4, 4.8))
        axs[0].imshow(CH1_ROI, interpolation=INTERPOLATION)
        axs[0].set_title("CH1")
        axs[1].imshow(CH2_ROI, interpolation=INTERPOLATION)
        axs[1].set_title("CH2")
        axs[2].imshow(CH3_ROI, interpolation=INTERPOLATION)
        axs[2].set_title("CH3")
        for ax in axs:
            ax.set_axis_off()
        plt.tight_layout()
        plt.show()

    return CH1_ROI, CH2_ROI, CH3_ROI, CH1_slice, CH2_slice, CH3_slice


def ratio_sc(ch1, ch2, ch3, plot=False):
    print("Channel Ratios - Define and plot...")
    R_cos_int = ch2 / ch1
    R_sin_int = ch3 / ch1
    if plot:
        fig, axs = plt.subplots(ncols=2, dpi=DPI)
        axs = np.ravel(axs)
        axs[0].imshow(R_cos_int, vmin=0.5, vmax=1.5, cmap="bwr", interpolation=INTERPOLATION)
        axs[0].set_axis_off()
        axs[1].imshow(R_sin_int, vmin=0.5, vmax=1.5, cmap="bwr", interpolation=INTERPOLATION)
        axs[1].set_axis_off()
        plt.show()
    return R_cos_int, R_sin_int


def Calculate_Phasors(CH_list, calibration, Processing, plot=DEBUG, title="DEBUG"):
    if plot:
        vmax = None  # max(ch.max() for ch in CH_list)
        fig, axs = plt.subplots(1, 3, figsize=(12.8 * ASPECT, 4.8), dpi=DPI)
        fig.suptitle(f"{title} - Phasor calculation")
        for i in range(3):
            ax = axs[i]
            ax.set_title(f"CH {i}")
            im = ax.imshow(CH_list[i], vmin=0, vmax=vmax, cmap=CMAP, interpolation=INTERPOLATION)
            if vmax is None or i == 2:
                plt.colorbar(im)
            if i != 0:
                ax.set_axis_off()
        plt.tight_layout()
        plt.show()

    SinCos_Fcos = np.asarray(calibration["SinCos_Fcos"])
    SinCos_Fsin = np.asarray(calibration["SinCos_Fsin"])
    img_g = 2 * (CH_list[1] / CH_list[0] - SinCos_Fcos[0]) / (SinCos_Fcos[1] - SinCos_Fcos[0]) - 1
    img_s = 2 * (CH_list[2] / CH_list[0] - SinCos_Fsin[0]) / (SinCos_Fsin[1] - SinCos_Fsin[0]) - 1
    img_g = median(img_g, disk(Processing["Median_filter_GS"]))
    img_s = median(img_s, disk(Processing["Median_filter_GS"]))
    img_ph = np.arctan2(img_s, img_g) % (2 * math.pi)
    img_mod = np.hypot(img_g, img_s)

    if plot:
        # vmax = np.percentile(CH_list[0], 99.9)
        vmax = CH_list[0].max()
        fig, axs = plt.subplots(1, 2, figsize=(10.24, 4.8), dpi=DPI)
        fig.suptitle(f"{title} - Phasor vs intensity")
        ax = axs[0]
        ax.set_xlabel("Phase")
        ax.set_ylabel("Intensity")
        ax.hist2d(img_ph.flat, CH_list[0].flat, bins=50, range=[[0, 2 * math.pi], [-10, vmax]], norm="log")
        ax = axs[1]
        ax.set_xlabel("Modulation")
        ax.set_yticklabels([])
        ax.hist2d(img_mod.flat, CH_list[0].flat, bins=50, range=[[0, 3], [-10, vmax]], norm="log")
        plt.tight_layout()
        plt.show()

    return img_g, img_s, img_ph, img_mod


def CCF2D(img1, img2, L_CCF):
    Size = np.shape(img1)
    F1 = np.fft.fft2(img1)
    F2 = np.fft.fft2(img2)
    CCF = F1 * np.conjugate(F2)
    G = np.sum(img1) * np.sum(img2) / Size[0] / Size[1]
    CCF = np.real(np.fft.fftshift(np.fft.ifft2(CCF), axes=(0, 1))) / G - 1
    CCF = CCF[Size[0] // 2 - L_CCF : Size[0] // 2 + L_CCF, Size[1] // 2 - L_CCF : Size[1] // 2 + L_CCF]
    return CCF


def normalize_percentile(data):
    data = np.array(data)
    min_val = np.percentile(data, 1)
    max_val = np.percentile(data, 99)

    if min_val == max_val:
        return np.zeros_like(data)

    normalized_data = (data - min_val) / (max_val - min_val)
    normalized_data[normalized_data < 0] = 0
    normalized_data[normalized_data > 1] = 1
    return normalized_data


def autocorr(CH_list):
    shift_list = []
    for i_img in range(1, 3):
        L = np.min(CH_list[0].shape) // 2
        CCF = CCF2D(CH_list[0], CH_list[i_img], L)
        shift = np.asarray([s[0] - L for s in np.where(CCF == CCF.max())]).astype(int)
        shift_list.append(shift)
    return shift_list


# In[ ]:


# @title **User input - Define Sine and Cosine filters parameters...**

SinCos_Fsin = np.asarray([11.2, 92.8]) / 100
SinCos_Fcos = np.asarray([10.9, 94.3]) / 100
SinCos_Lambda = np.asarray([400, 700])


# In[ ]:


# @title **User input - Declaring file paths** (copy-paste path to each file, see below)
# %% CALIBRATION PATHS

# Path to the data root directory
try:
    # running on Google Colab
    from google.colab import drive

    data_path = "/content/drive/Shareddrives/Prescher Lab/Phasor Data Analysis/Data for tutorial/"
except ImportError:
    # running locally
    data_path = "Data for tutorial/"

# Path to the dark calibration file (same parameters as experiments)
Dark_Path = os.path.join(data_path, "Calibration/Dark - 2min")

# Path to the dark calibration file for bright calibration (same parameters as bright)
Dark_Bright_Path = os.path.join(data_path, "Calibration/Dark - 200ms")

# Path to the bright calibration file
Bright_Path = os.path.join(data_path, "Calibration/Bright - Lamp")

# %% EXPERIMENT FOR REGISTRATION
# Path to the registration experiment (one experiment from the dataset)
Registration_Path = os.path.join(data_path, "Experiments/MB-YenL - 1 - Bio")

# %% EXPERIMENT ANALYSIS
# Path to the experiment folder (inside this, each experiment is a folder with .tiff files for each frame)
Experiment_Folder_Path = os.path.join(data_path, "Experiments")

# Whether to save analyzed images as PNG (slower)
FLAG_SAVE_IMAGES = "True"  # @param ["True", "False"]

Dark_Path = os.path.normpath(os.environ.get("PA_DARK_PATH", Dark_Path))
Dark_Bright_Path = os.path.normpath(os.environ.get("PA_DARK_BRIGHT_PATH", Dark_Bright_Path))
Bright_Path = os.path.normpath(os.environ.get("PA_BRIGHT_PATH", Bright_Path))
Registration_Path = os.path.normpath(os.environ.get("PA_REGISTRATION_PATH", Registration_Path))
Experiment_Folder_Path = os.path.normpath(os.environ.get("PA_EXPERIMENT_PATH", Experiment_Folder_Path))

SAVE_IMAGES = os.environ.get("PA_SAVE_IMAGES", FLAG_SAVE_IMAGES) in {"1", "TRUE", "True"}

if DEBUG:
    print(f"{Dark_Path=!r}")
    print(f"{Dark_Bright_Path=!r}")
    print(f"{Bright_Path=!r}")
    print(f"{Registration_Path=!r}")
    print(f"{Experiment_Folder_Path=!r}")


# In[ ]:


# @title **Automatic - Define Calibration paths and load files**

radius = 2.0
sigma = 3.0
dark_correction = 0.0  # additional dark counts to subtract

radius = float(os.environ.get("PA_RADIUS", radius))
sigma = float(os.environ.get("PA_SIGMA", sigma))
dark_correction = float(os.environ.get("PA_DARK_CORRECTION", dark_correction))

print("Loading Dark files... ")
if DEBUG:
    print(f"{radius=}")
    print(f"{sigma=}")
    print(f"{dark_correction=}")

# Load dark path (for experiments) and calcute offset
images_dark = preprocess(Dark_Path, radius=radius)
dark = np.median(images_dark).item() + dark_correction

if DEBUG:
    print(f"{dark=}")

print()
print("Loading Bright-Dark files... ")
# Load bright dark path (for bright) and calculate offset
images_bright_dark = preprocess(Dark_Bright_Path, radius=radius)
bright_dark = np.median(images_bright_dark)

if DEBUG:
    print(f"{bright_dark=}")

print()
print("Loading Calibration files... ")
# Load bright IMAGE and remove offset
images_bright = preprocess(Bright_Path, radius=radius, sigma=sigma)
img_bright = np.median(images_bright, 2) - bright_dark

if DEBUG:
    if images_dark.ndim == 2:
        images_dark = np.expand_dims(images_dark, axis=2)
    if images_bright.ndim == 2:
        images_bright = np.expand_dims(images_bright, axis=2)
    if images_bright_dark.ndim == 2:
        images_bright_dark = np.expand_dims(images_bright_dark, axis=2)

    # fig, axs = plt.subplots(4, 3, figsize=(12.8, 10.24), dpi=DPI)
    fig = plt.figure(figsize=(12.8, 12.8), dpi=DPI)
    gs = gridspec.GridSpec(nrows=4, ncols=3, figure=fig, height_ratios=[2.5, 1, 1, 1])
    # fig.suptitle("...")
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title(f"Dark ({dark:.1f})")
    im = ax.imshow(np.median(images_dark, axis=2), cmap=CMAP, interpolation=INTERPOLATION)
    fig.colorbar(im, orientation="horizontal")
    ax = fig.add_subplot(gs[1, 0])
    ax.set_title("Dark")
    ax.set_ylabel("Intensity")
    ax.set_xlabel("Frame index")
    ax.plot(np.median(images_dark, axis=(0, 1)), label="Median")
    ax.plot(np.mean(images_dark, axis=(0, 1)), label="Mean")
    ax.legend()
    ax = fig.add_subplot(gs[2, 0])
    ax.set_title("Dark")
    ax.set_ylabel("Intensity")
    ax.set_xlabel("Horizontal pixel")
    ax.plot(np.median(images_dark, axis=(0, 2)), label="Median")
    ax.plot(np.mean(images_dark, axis=(0, 2)), label="Mean")
    ax.legend()
    ax = fig.add_subplot(gs[3, 0])
    ax.set_title("Dark")
    ax.set_ylabel("Intensity")
    ax.set_xlabel("Vertical pixel")
    ax.plot(np.median(images_dark, axis=(1, 2)), label="Median")
    ax.plot(np.mean(images_dark, axis=(1, 2)), label="Mean")
    ax.legend()

    ax = fig.add_subplot(gs[0, 1])
    ax.set_title(f"Bright dark ({bright_dark:.1f})")
    im = ax.imshow(np.median(images_bright_dark, axis=2), cmap=CMAP, interpolation=INTERPOLATION)
    fig.colorbar(im, orientation="horizontal")
    ax = fig.add_subplot(gs[1, 1])
    ax.set_title("Bright dark")
    ax.set_ylabel("Intensity")
    ax.set_xlabel("Frame index")
    ax.plot(np.median(images_bright_dark, axis=(0, 1)), label="Median")
    ax.plot(np.mean(images_bright_dark, axis=(0, 1)), label="Mean")
    ax.legend()
    ax = fig.add_subplot(gs[2, 1])
    ax.set_title("Bright dark")
    ax.set_ylabel("Intensity")
    ax.set_xlabel("Horizontal pixel")
    ax.plot(np.median(images_bright_dark, axis=(0, 2)), label="Median")
    ax.plot(np.mean(images_bright_dark, axis=(0, 2)), label="Mean")
    ax.legend()
    ax = fig.add_subplot(gs[3, 1])
    ax.set_title("Bright dark")
    ax.set_ylabel("Intensity")
    ax.set_xlabel("Vertical pixel")
    ax.plot(np.median(images_bright_dark, axis=(1, 2)), label="Median")
    ax.plot(np.mean(images_bright_dark, axis=(1, 2)), label="Mean")
    ax.legend()

    ax = fig.add_subplot(gs[0, 2])
    ax.set_title("Bright")
    im = ax.imshow(np.median(images_bright, axis=2), cmap=CMAP, interpolation=INTERPOLATION)
    fig.colorbar(im, orientation="horizontal")
    ax = fig.add_subplot(gs[1, 2])
    ax.set_title("Bright")
    ax.set_ylabel("Intensity")
    ax.set_xlabel("Frame index")
    ax.plot(np.median(images_bright, axis=(0, 1)), label="Median")
    # ax.plot(np.mean(images_bright, axis=(0, 1)), label="Mean")
    ax.legend()
    ax = fig.add_subplot(gs[2, 2])
    ax.set_title("Bright")
    ax.set_ylabel("Intensity")
    ax.set_xlabel("Horizontal pixel")
    ax.plot(np.median(images_bright, axis=(0, 2)), label="Median")
    # ax.plot(np.mean(images_bright, axis=(0, 2)), label="Mean")
    ax.legend()
    ax = fig.add_subplot(gs[3, 2])
    ax.set_title("Bright")
    ax.set_ylabel("Intensity")
    ax.set_xlabel("Vertical pixel")
    ax.plot(np.median(images_bright, axis=(1, 2)), label="Median")
    # ax.plot(np.mean(images_bright, axis=(1, 2)), label="Mean")
    ax.legend()
    plt.tight_layout()
    plt.show()


# In[ ]:


# @title **User input - Define threshold for identifying channels** (the image "Detected ROIs" should show three rectangles corresponding to the three images){run: "auto"}

# Threshold value to allow algorithm to automatically find the three channels (ROI: Region Of Interest)
threshold_value = 3850  # @param {type: "slider", min: 200, max: 10000}

threshold_value = int(os.environ.get("PA_THRESHOLD_VALUE", threshold_value))
if DEBUG:
    print(f"{threshold_value=}")

# Extracts ROIs based on threshold, saves ROIs as images and stored pixel coordinates (slice)
CH1_ROI, CH2_ROI, CH3_ROI, CH1_slice, CH2_slice, CH3_slice = norm_slicing(threshold_value, img_bright, dark)
Ch_slices = [CH1_slice, CH2_slice, CH3_slice]


# In[ ]:


# @title **Automatic - Initializing selected file for registration**

print("Loading registration experiment...")
# Load and process registration image
img_exp = np.median(preprocess(Registration_Path), 2)

print("Correcting channel shifts...")
# Apply slicing (computed above) to registration image
CH_list = [normalize_percentile(img_exp[Ch_slices[n]]) for n in range(3)]

shift = autocorr(CH_list)
# Select the maximum shift found through the autocorrelation function
crop = np.abs(shift).max()

# Apply the crop to all the slices (this step avoids potential negative coordinates)
Ch_slices = [
    (
        slice(Ch_slices[n][0].start + crop, Ch_slices[n][0].stop - crop),
        slice(Ch_slices[n][1].start + crop, Ch_slices[n][1].stop - crop),
    )
    for n in range(3)
]

# Apply the shifts calculated above
Ch_slices = [
    Ch_slices[0],
    (
        slice(Ch_slices[1][0].start - shift[0][0], Ch_slices[1][0].stop - shift[0][0]),
        slice(Ch_slices[1][1].start - shift[0][1], Ch_slices[1][1].stop - shift[0][1]),
    ),
    (
        slice(Ch_slices[2][0].start - shift[1][0], Ch_slices[2][0].stop - shift[1][0]),
        slice(Ch_slices[2][1].start - shift[1][1], Ch_slices[2][1].stop - shift[1][1]),
    ),
]


# In[ ]:


# @title **User input - Manual crop** (Removes a number of pixels from the sides if a border error is apparent){run: "auto"}

# Number of pixels to remove from each side
Left = 5  # @param {type: "slider", min: 0, max: 200}
Right = 30  # @param {type: "slider", min: 0, max: 200}
Top = 10  # @param {type: "slider", min: 0, max: 500}
Bottom = 10  # @param {type: "slider", min: 0, max: 500}

Left = int(os.environ.get("PA_CROP_LEFT", Left))
Right = int(os.environ.get("PA_CROP_RIGHT", Right))
Top = int(os.environ.get("PA_CROP_TOP", Top))
Bottom = int(os.environ.get("PA_CROP_BOTTOM", Bottom))

# Define manual cropping parameters
Crop = [Top, Bottom, Left, Right]
if DEBUG:
    print(f"{Crop=}")

# Return cropped ROIs
Ch_slices = [
    (
        slice(ch_slice[0].start + Top, ch_slice[0].stop - Bottom),
        slice(ch_slice[1].start + Left, ch_slice[1].stop - Right),
    )
    for ch_slice in Ch_slices
]

img = img_exp[Ch_slices[0]]
if DEBUG:
    print(f"Image shape={img.shape}")
# ASPECT = img.shape[1] / img.shape[0] * 2

fig, ax = plt.subplots(ncols=1, figsize=(6.4 * ASPECT, 4.8), dpi=100)
ax.set_title("Sliced and cropped image")
vmin = np.percentile(img, 1)
vmax = np.percentile(img, 99)
im = ax.imshow(img, vmin=vmin, vmax=vmax, interpolation=INTERPOLATION)
fig.colorbar(im)
# ax.set_axis_off()
plt.show()


# In[ ]:


# @title **Automatic - Create experiment dictionary**

# Calculate intensity ratios
print("Channel Ratios - Define and plot...")
CH1_ROI, CH2_ROI, CH3_ROI = (img_bright[Slice] for Slice in Ch_slices)
R_cos_int, R_sin_int = ratio_sc(CH1_ROI, CH2_ROI, CH3_ROI)

# Compute the arithmetic mean along the specified axis, ignoring NaNs.
R_cos_int[np.isnan(R_cos_int)] = np.nanmean(R_cos_int)
R_sin_int[np.isnan(R_sin_int)] = np.nanmean(R_sin_int)

# Creates folder for storing analysis parameters
# name = Experiment_Folder_Path + "Experiments_PhasorScope"
# directory = Path(Experiment_Folder_Path) / name
# directory.mkdir(exist_ok=True)

# Creates dictionary for storing analysis parameters
calibration = {
    # Paths
    "Path Calibration_Bright": Bright_Path,
    "Path Calibration_Bright_Dark": Dark_Path,
    "Path Calibration_Dark": Dark_Bright_Path,
    "Path Registration": Registration_Path,
    "Path Experiments Folder": Experiment_Folder_Path,
    # Sine and cosine parameters
    "SinCos_Fsin": [SinCos_Fsin[0], SinCos_Fsin[-1]],
    "SinCos_Fcos": [SinCos_Fcos[0], SinCos_Fcos[-1]],
    # Calibration processing parameters
    "Median Filter for hot pixels removal": radius,
    "Gaussian Filter for bright image smoothing": sigma,
    # User inputs
    "Manual_Threshold": threshold_value,
    # Calibration and registration outputs
    "Value Calibration_Dark": dark,
    "Slices": Ch_slices,
    "Bright": [R_cos_int, R_sin_int],
}


# In[ ]:


# @title **Automatic - Define GPU-accelerated default Cellpose model**

print("Defining Cellpose model...")
model = models.CellposeModel(gpu=True)

print("Load registration experiment...")
# Load and process registration image
img_exp = preprocess(Registration_Path, radius=calibration["Median Filter for hot pixels removal"])

print("Applying calibration...")
# Load and process registration image
CH1, CH2, CH3 = Apply_Calibration(img_exp, calibration, plot=False)


# In[ ]:


# @title **User Input - Processing values** (WARNING: Median_filter_GS smooths the phasor images but <u>**Phasors are calculated on UNPROCESSED images**<u>, tune parameters ONLY for optimizing segmentation){run: "auto"}

# How many frames to average for segmentation
Time_binning = 200  # @param {type: "slider", min: 1, max: 200}
# Median filter to image
Median_filter = 3  # @param {type: "slider", min: 1, max: 21, step: 2}
# Background subtration (sigma of gaussian smoothing)
Bkg_subtraction = 400  # @param {type: "slider", min: 0, max: 1001, step: 1}
# Size of cells in pixels
Cellpose_diameter = 356  # @param {type: "slider", min: 5, max: 600}
# Radius of median filter to (g,s) coordinates
Median_filter_GS = 1  # @param {type: "slider", min: 0, max: 21, step: 1}

Time_binning = int(os.environ.get("PA_TIME_BINNING", Time_binning))
Median_filter = int(os.environ.get("PA_MEDIAN_FILTER", Median_filter))
Bkg_subtraction = int(os.environ.get("PA_BKG_SUBTRACTION", Bkg_subtraction))
Cellpose_diameter = int(os.environ.get("PA_CELLPOSE_DIAMETER", Cellpose_diameter))
Median_filter_GS = int(os.environ.get("PA_MEDIAN_FILTER_GS", Median_filter_GS))

if Cellpose_diameter <= 0:
    Cellpose_diameter = None  # let Cellpose choose the diameter

# Store processing parameters
Processing = {
    "Time_binning": Time_binning,
    "Median_filter": Median_filter,
    "Median_filter_GS": Median_filter_GS,
    "Bkg_subtraction": Bkg_subtraction,
    "Cellpose_diameter": Cellpose_diameter,
    "Cellpose_model": model,
}

if DEBUG:
    print(f"{Time_binning=}")
    print(f"{Median_filter=}")
    print(f"{Median_filter_GS=}")
    print(f"{Bkg_subtraction=}")
    print(f"{Cellpose_diameter=}")

# Create a list of slices
CH_list = [np.mean(ch[:, :, :Time_binning], 2) for ch in [CH1, CH2, CH3]]
# Compute image segmentation and generate masks
tmp_img, masks = Process_Img(CH_list[0], Processing, plot=False)
# Calculate the phasor using the calibration and processing parameters defined above
img_g, img_s, img_ph, img_mod = Calculate_Phasors(CH_list, calibration, Processing, plot=False)

fig, axs = plt.subplots(ncols=4, dpi=DPI, figsize=(12 * ASPECT, 4))
axs[1].imshow(tmp_img, vmax=np.percentile(tmp_img, 99.9), cmap="hot", interpolation=INTERPOLATION)
axs[1].set_title("Processed image")
axs[1].set_axis_off()
axs[1].set_aspect(1)
axs[2].imshow(masks, cmap="nipy_spectral", interpolation=INTERPOLATION)
axs[2].set_title("Segmentation")
axs[2].set_axis_off()
axs[2].set_aspect(1)
# axs[0].imshow(CH_list[0], vmax=np.percentile(tmp_img, 99.9), cmap="hot", interpolation=INTERPOLATION)
# axs[0].set_title("Original image")
# axs[0].set_axis_off()
# axs[0].set_aspect(1)
axs[0].imshow(CH_list[0], vmax=np.percentile(CH_list[0], 99.9), cmap="hot", interpolation=INTERPOLATION)
axs[0].set_title("Original image")
axs[0].set_axis_off()
axs[0].set_aspect(1)

gs_lim = 1
axs[3].hist2d(
    img_g[masks > 0],
    img_s[masks > 0],
    bins=128,
    range=np.asarray([[-gs_lim, gs_lim], [-gs_lim, gs_lim]]),
    cmap="nipy_spectral",
)
axs[3].set_axis_off()
axs[3].set_title("Phasor - Pixels")
axs[3].set_aspect(1)
plot_grid(axs[3], radii=[0, 0.25, 0.5, 0.75, 1], angles=np.arange(0, 360, 45), color="white")
plt.tight_layout()


# In[ ]:


# @title **Automatic - Save calibration and processing parameters...**

# Save the calibration parameters
fname = os.path.join(Experiment_Folder_Path, "Calibration.npy")
print(f"Saving calibration to {fname!r}")
np.save(fname, calibration)  # type: ignore[call-overload]

with open(os.path.join(Experiment_Folder_Path, "Calibration.txt"), "w") as f:
    f.write(str(calibration))

# Save the processing parameters
fname = os.path.join(Experiment_Folder_Path, "Processing.npy")
print(f"Saving parameters to {fname!r}")
np.save(fname, Processing)  # type: ignore[call-overload]


# In[ ]:


# @title **Automatic - Process all experiment files**

# Load experiments from folder
Experiments_Path = []
for fname in os.listdir(Experiment_Folder_Path):
    fname = os.path.join(Experiment_Folder_Path, fname)
    if os.path.isdir(fname) and "Experiments_PhasorScope" not in fname:
        Experiments_Path.append(fname)

print(f"Found: {len(Experiments_Path)} files")

# Define dataframe
print("Starting experiment(s) analysis...")
Columns = ("g", "s", "intensity (cam1 sum INT)", "intensity (cam2 sum INT)", "x", "y", "color", "fname")
df_all = pd.DataFrame(columns=Columns, index=[])
dy_text = 0

# Custom colormap for phase images
hsv = plt.colormaps.get_cmap("hsv")
n_steps = 256  # Number of color steps in the final colormap
colors = [hsv(i / (n_steps - 1))[:3] for i in range(n_steps)]
colors[0] = (0.0, 0.0, 0.0)  # black
cmap_custom = mcolors.LinearSegmentedColormap.from_list("hsv_black_start", colors)

# Load processing and calibration parameters
Processing = np.load(os.path.join(Experiment_Folder_Path, "Processing.npy"), allow_pickle=True).item()
Calibration = np.load(os.path.join(Experiment_Folder_Path, "Calibration.npy"), allow_pickle=True).item()

# Compute experiment analysis.
# The loop calibrates and processes the experiment data one by one.
# Final plots include: Experiment title
# - Cell masks
# - Image intensity
# - Phase
# - Modulation
# - Phasor - Single cells
# - Phasor - Single pixel

for i_exp, exp_path in enumerate(Experiments_Path[:]):
    try:
        print(f"Experiment #{i_exp + 1}/{len(Experiments_Path)}")
        fname = os.path.basename(exp_path)
        print(f"Experiment name: {fname}")

        print("Loading...")
        img_exp = preprocess(exp_path, radius=Calibration["Median Filter for hot pixels removal"])
        print("Applying calibration...")
        CH1, CH2, CH3 = Apply_Calibration(img_exp, Calibration, title=fname)
        print("Applying time average...")
        CH1, CH2, CH3 = (np.median(ch[:, :, : Processing["Time_binning"]], 2) for ch in [CH1, CH2, CH3])
        print("Applying image processing...")
        tmp_img, masks = Process_Img(CH1, Processing, title=fname)
        print("Calculating phasors...")
        img_g, img_s, img_ph, img_mod = Calculate_Phasors([CH1, CH2, CH3], Calibration, Processing, title=fname)

        print("Plotting phasors...")
        cmap = plt.get_cmap("nipy_spectral")
        CMap = cmap(np.linspace(0, 1, np.max(masks) + 1))
        df = pd.DataFrame(columns=Columns, index=[])

        # Images - Phasors as median of g,s
        fig, axs = plt.subplots(ncols=4, dpi=DPI, figsize=(12.8 * ASPECT, 4.8))
        fig.suptitle(fname + " - Images")
        img1 = axs[0].imshow(masks, cmap="nipy_spectral", interpolation=INTERPOLATION)
        cbar = plt.colorbar(img1)
        axs[0].set_title("Cell index")
        axs[0].set_axis_off()
        for i_cells in range(0, np.max(masks)):
            cell_idx = i_cells + 1
            logic = masks == (cell_idx)
            idx = np.where(logic)
            g_cell = np.median(img_g[logic])
            s_cell = np.median(img_s[logic])
            x_cell = np.mean(idx[0])
            y_cell = np.mean(idx[1])
            axs[0].arrow(y_cell + dy_text, x_cell - dy_text, -dy_text, dy_text, color="gray", linewidth=0.3)
            axs[0].text(
                y_cell + dy_text,
                x_cell - dy_text,
                str(cell_idx),
                color="white",
                horizontalalignment="center",
                verticalalignment="center",
            )
        img2 = axs[1].imshow(CH1, cmap="hot", vmin=0, vmax=np.percentile(CH1, 99.9), interpolation=INTERPOLATION)
        cbar = plt.colorbar(img2)
        axs[1].set_title("Intensity")
        axs[1].set_axis_off()
        img3 = axs[2].imshow(
            img_ph * (masks > 0), cmap=cmap_custom, vmin=0, vmax=math.pi * 2, interpolation=INTERPOLATION
        )
        cbar = plt.colorbar(img3)
        axs[2].set_title("Phase (rad)")
        axs[2].set_axis_off()
        img4 = axs[3].imshow(img_mod * (masks > 0), cmap="nipy_spectral", vmin=0, vmax=1, interpolation=INTERPOLATION)
        cbar = plt.colorbar(img4)
        axs[3].set_title("Modulation")
        axs[3].set_axis_off()
        plt.tight_layout()
        fig.savefig(exp_path + "_Images.png")
        plt.tight_layout()
        plt.show()

        # Phasor plot - Single cells and pixels
        fig, axs = plt.subplots(ncols=2, dpi=DPI, figsize=(10.24 * ASPECT, 4.8))
        fig.suptitle(fname + " - Phasors")
        gs_lim = 1
        axs[1].hist2d(
            img_g[masks > 0],
            img_s[masks > 0],
            bins=128,
            range=np.asarray([[-gs_lim, gs_lim], [-gs_lim, gs_lim]]),
            cmap="nipy_spectral",
        )
        axs[1].set_axis_off()
        axs[1].set_title("Phasor - Pixels")
        axs[1].set_aspect(1)
        plot_grid(axs[1], radii=[0, 0.25, 0.5, 0.75, 1], angles=np.arange(0, 360, 45), color="white")
        plot_grid(axs[0], radii=[0, 0.25, 0.5, 0.75, 1], angles=np.arange(0, 360, 45), color="black")
        for i_cells in range(0, np.max(masks)):
            cell_idx = i_cells + 1
            logic = masks == (cell_idx)
            idx = np.where(logic)
            CH_list1 = [np.sum(CH_list[n][logic]) for n in range(len(CH_list))]
            g_sum_int_cell = 2 * (CH_list1[1] / CH_list1[0] - SinCos_Fcos[0]) / (SinCos_Fcos[1] - SinCos_Fcos[0]) - 1
            s_sum_int_cell = 2 * (CH_list1[2] / CH_list1[0] - SinCos_Fsin[0]) / (SinCos_Fsin[1] - SinCos_Fsin[0]) - 1
            g_cell = np.median(img_g[logic])
            s_cell = np.median(img_s[logic])
            x_cell = np.mean(idx[0])
            y_cell = np.mean(idx[1])
            df_tmp = pd.DataFrame(
                {"g": g_cell, "s": s_cell, "x": x_cell, "y": y_cell, "color": [CMap[cell_idx]], "fname": fname},
                index=[cell_idx],
            )
            df = pd.concat((df if not df.empty else None, df_tmp))
            axs[0].plot(g_cell, s_cell, "o", markerfacecolor=CMap[cell_idx], markeredgecolor="k", markersize=16)
            axs[0].text(
                g_cell, s_cell, str(cell_idx), color="black", horizontalalignment="center", verticalalignment="center"
            )
        axs[0].set_xlim([-1, 1])
        axs[0].set_ylim([-1, 1])
        axs[0].set_aspect(1)
        axs[0].set_xlabel("g")
        axs[0].set_ylabel("s")
        axs[0].set_title("Phasor - Single cells")
        plt.tight_layout()
        plt.savefig(exp_path + "_Phasors.png")
        plt.show()

        # Mask regions of interest in the phasor space using circular cursors:
        print("Plotting cursors...")

        cursor_real = [0.4, -0.2]  # G coordinate of cursors
        cursor_imag = [0.5, 0.2]  # S coordinate of cursors
        cursor_radius = [0.4, 0.3]

        i = 0
        while True:
            if f"PA_CURSOR_G{i}" in os.environ and f"PA_CURSOR_S{i}" in os.environ and f"PA_CURSOR_R{i}" in os.environ:
                if i == 0:
                    cursor_real = []
                    cursor_imag = []
                    cursor_radius = []
                cursor_real.append(float(os.environ[f"PA_CURSOR_G{i}"]))
                cursor_imag.append(float(os.environ[f"PA_CURSOR_S{i}"]))
                cursor_radius.append(float(os.environ[f"PA_CURSOR_R{i}"]))
            else:
                break
            i += 1

        if DEBUG:
            print(f"{cursor_real=}")
            print(f"{cursor_imag=}")
            print(f"{cursor_radius=}")

        # mean = CH1
        # mean /= np.percentile(mean, 99.9)
        # mean = np.clip(mean, 0, 1.0)
        # mean = skimage.exposure.adjust_log(mean, 1)

        real = img_g
        imag = img_s

        real[masks == 0] = np.nan
        imag[masks == 0] = np.nan

        cursors_masks = mask_from_circular_cursor(real, imag, cursor_real, cursor_imag, radius=cursor_radius)

        fig, axs = plt.subplots(1, 2, figsize=(10.24 * ASPECT, 4.8), dpi=DPI)
        fig.suptitle(f"{fname} - Cursors")

        phasorplot = PhasorPlot(ax=axs[0], allquadrants=True, title="Phasor plot")
        phasorplot.hist2d(real, imag, cmap="Greys")
        for i in range(len(cursor_real)):
            phasorplot.circle(
                cursor_real[i],
                cursor_imag[i],
                radius=cursor_radius[i],
                color=CATEGORICAL[i],
                linestyle="-",
                linewidth=2,
            )
        axs[0].set_xlabel("G")
        axs[0].set_ylabel("S")

        pseudo_color_image = pseudo_color(*cursors_masks)  #  intensity=mean

        axs[1].imshow(pseudo_color_image)
        axs[1].set_axis_off()
        axs[1].set_title("Cursor masks")
        plt.tight_layout()
        fig.savefig(exp_path + "_Cursors.png")
        plt.show()

        # Save results
        df.to_excel(exp_path + ".xlsx")
        df_all = pd.concat((df_all if not df_all.empty else None, df))

        if SAVE_IMAGES:
            print("Saving images...")
            Experiment_Images = {
                "CH_list": CH_list,
                "masks": masks,
                "img_g": img_g,
                "img_s": img_s,
                "img_ph": img_ph,
                "img_mod": img_mod,
                "df": df,
                "Calibration": Calibration,
                "Processing": Processing,
            }
            np.savez_compressed(exp_path + "_Images.npz", Experiment_Images)  # type: ignore[arg-type]

    except Exception as exc:
        print(f"SKIPPED: Experiment #{i_exp + 1}/{len(Experiments_Path)}")
        if DEBUG:
            traceback.print_exc()
        else:
            print(exc)

df_all.to_excel(os.path.join(os.path.dirname(exp_path), "All experiments.xlsx"))

if DEBUG:
    print(f"Runtime {time.perf_counter() - time_started:.0f} s")


# In[ ]:
