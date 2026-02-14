"""
DICOM file reading utilities for ultrasound images.

Provides functions to read DICOM files and convert them into PIL Images
suitable for deep learning preprocessing pipelines.
"""

import numpy as np
from PIL import Image
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut


def read_dicom_file(path):
    """Read a DICOM file and convert it to a PIL RGB Image.

    Applies modality LUT and VOI LUT transformations when available,
    normalizes pixel values to 0-255, and converts grayscale to RGB.

    Args:
        path (str): Path to the DICOM file.

    Returns:
        PIL.Image or None: RGB image if successful, None on failure.
    """
    try:
        dicom = pydicom.dcmread(path)
        data = dicom.pixel_array

        # Apply modality LUT (rescale slope/intercept) if available
        if hasattr(dicom, "RescaleIntercept") and hasattr(dicom, "RescaleSlope"):
            data = apply_modality_lut(data, dicom)

        # Apply VOI LUT (window center/width) if available
        if hasattr(dicom, "WindowCenter") and hasattr(dicom, "WindowWidth"):
            try:
                data = apply_voi_lut(data, dicom)
            except Exception as e:
                print(f"Warning: Could not apply VOI LUT: {e}")

        # Normalize pixel values to 0-255 range
        data_min = data.min()
        data_max = data.max()

        if data_max != data_min:
            data = ((data - data_min) / (data_max - data_min)) * 255.0
        else:
            print(f"Warning: Uniform pixel values in DICOM file: {path}")

        data = data.astype(np.uint8)

        # Convert single-channel grayscale to 3-channel RGB
        if len(data.shape) == 2:
            data = np.stack([data, data, data], axis=2)

        img = Image.fromarray(data)
        return img

    except Exception as e:
        print(f"Error reading DICOM file {path}: {e}")
        return None
