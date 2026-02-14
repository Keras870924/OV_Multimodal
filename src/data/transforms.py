"""
Custom image transforms for ultrasound image preprocessing and augmentation.

Includes ROI auto-cropping, histogram normalization, and ultrasound-specific
noise augmentation strategies (speckle noise, shadow simulation, etc.).
"""

import random

import cv2
import numpy as np
import torch
from PIL import Image


class AddSpeckleNoise:
    """Add multiplicative speckle noise to a tensor image.

    Args:
        noise_variance (float): Standard deviation of the Gaussian noise.
    """

    def __init__(self, noise_variance=0.05):
        self.noise_variance = noise_variance

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.noise_variance
        return tensor + noise


class HistogramNormalize:
    """Apply per-channel histogram equalization to a PIL Image.

    Args:
        num_bins (int): Number of histogram bins.
    """

    def __init__(self, num_bins=256):
        self.num_bins = num_bins

    def __call__(self, img):
        np_img = np.array(img)
        normalized_img = np.zeros_like(np_img, dtype=np.uint8)
        for i in range(3):
            channel = np_img[..., i]
            hist, bins = np.histogram(channel.flatten(), self.num_bins, density=True)
            cdf = hist.cumsum()
            cdf = (self.num_bins - 1) * cdf / cdf[-1]
            normalized_channel = np.interp(channel.flatten(), bins[:-1], cdf)
            normalized_img[..., i] = normalized_channel.reshape(channel.shape)
        return Image.fromarray(normalized_img)


class EnhancedHistogramNormalize:
    """Apply CLAHE followed by global histogram equalization.

    Converts the image to LAB color space and applies Contrast Limited
    Adaptive Histogram Equalization (CLAHE) to the luminance channel,
    then performs global histogram equalization on each RGB channel.

    Args:
        clip_limit (float): CLAHE clip limit.
        grid_size (tuple): CLAHE tile grid size.
        num_bins (int): Number of histogram bins for global equalization.
    """

    def __init__(self, clip_limit=2.0, grid_size=(8, 8), num_bins=256):
        self.clip_limit = clip_limit
        self.grid_size = grid_size
        self.num_bins = num_bins

    def __call__(self, img):
        np_img = np.array(img)

        # Apply CLAHE on the luminance channel in LAB color space
        if len(np_img.shape) == 3 and np_img.shape[2] == 3:
            lab = cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)

            clahe = cv2.createCLAHE(
                clipLimit=self.clip_limit, tileGridSize=self.grid_size
            )
            cl = clahe.apply(l)

            enhanced_lab = cv2.merge((cl, a, b))
            enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        else:
            # Grayscale: apply CLAHE directly
            clahe = cv2.createCLAHE(
                clipLimit=self.clip_limit, tileGridSize=self.grid_size
            )
            enhanced_img = clahe.apply(np_img)

        # Global histogram equalization to standardize brightness distribution
        normalized_img = np.zeros_like(enhanced_img, dtype=np.uint8)
        if len(enhanced_img.shape) == 3:
            for i in range(3):
                channel = enhanced_img[..., i]
                hist, bins = np.histogram(
                    channel.flatten(), self.num_bins, density=True
                )
                cdf = hist.cumsum()
                cdf = (self.num_bins - 1) * cdf / cdf[-1]
                normalized_channel = np.interp(channel.flatten(), bins[:-1], cdf)
                normalized_img[..., i] = normalized_channel.reshape(channel.shape)
        else:
            hist, bins = np.histogram(
                enhanced_img.flatten(), self.num_bins, density=True
            )
            cdf = hist.cumsum()
            cdf = (self.num_bins - 1) * cdf / cdf[-1]
            normalized_img = np.interp(
                enhanced_img.flatten(), bins[:-1], cdf
            ).reshape(enhanced_img.shape)

        return Image.fromarray(normalized_img)


class UltrasoundNoiseAugmentation:
    """Add ultrasound-specific noise combining speckle and Gaussian noise.

    Applies multiplicative speckle noise (characteristic of ultrasound) and
    optionally adds Gaussian noise to simulate electronic interference.

    Args:
        noise_intensity (float): Intensity of the speckle noise.
        speckle_ratio (float): Probability of adding additional Gaussian noise.
    """

    def __init__(self, noise_intensity=0.1, speckle_ratio=0.3):
        self.noise_intensity = noise_intensity
        self.speckle_ratio = speckle_ratio

    def __call__(self, img):
        np_img = np.array(img).astype(np.float32) / 255.0

        # Add multiplicative speckle noise (appropriate for ultrasound images)
        speckle = np.random.randn(*np_img.shape) * self.noise_intensity
        noisy_img = np_img + np_img * speckle

        # Optionally add Gaussian noise to simulate electronic noise
        if np.random.random() < self.speckle_ratio:
            gaussian_noise = np.random.randn(*np_img.shape) * (
                self.noise_intensity / 2
            )
            noisy_img += gaussian_noise

        # Clip values to valid [0, 1] range
        noisy_img = np.clip(noisy_img, 0, 1)

        return Image.fromarray((noisy_img * 255).astype(np.uint8))


class UltrasoundSpeckleNoise:
    """Apply multiplicative speckle noise to a tensor with 50% probability.

    Simulates the granular speckle pattern characteristic of ultrasound imaging.

    Args:
        noise_variance (float): Standard deviation of the noise.
    """

    def __init__(self, noise_variance=0.05):
        self.noise_variance = noise_variance

    def __call__(self, tensor):
        if random.random() < 0.5:
            noise = torch.randn_like(tensor) * self.noise_variance
            return tensor + tensor * noise  # Multiplicative noise
        return tensor


class UltrasoundShadowAugmentation:
    """Simulate acoustic shadow artifacts in ultrasound images.

    Creates a vertical attenuation band at a random position to mimic
    the shadow effect caused by highly reflective structures.

    Args:
        shadow_intensity (float): Maximum attenuation factor (0-1).
        shadow_size (float): Width of the shadow as a fraction of image width.
    """

    def __init__(self, shadow_intensity=0.5, shadow_size=0.2):
        self.shadow_intensity = shadow_intensity
        self.shadow_size = shadow_size

    def __call__(self, tensor):
        # Apply with 20% probability
        if random.random() < 0.2:
            c, h, w = tensor.shape
            x = int(random.uniform(0, w * 0.8))
            width = int(w * self.shadow_size)

            # Create an attenuation mask simulating acoustic shadow
            shadow_mask = torch.ones_like(tensor)
            for i in range(width):
                if x + i < w:
                    attenuation = 1.0 - (self.shadow_intensity * (i / width))
                    shadow_mask[:, :, x + i] = attenuation

            return tensor * shadow_mask
        return tensor


class UltrasoundEnhancementAugmentation:
    """Simulate posterior acoustic enhancement in ultrasound images.

    Creates a localized brightness enhancement region with a Gaussian
    profile to mimic the enhancement seen behind fluid-filled structures.

    Args:
        enhancement_intensity (float): Peak enhancement factor.
        enhancement_size (float): Size of the enhancement region as a fraction.
    """

    def __init__(self, enhancement_intensity=1.5, enhancement_size=0.2):
        self.enhancement_intensity = enhancement_intensity
        self.enhancement_size = enhancement_size

    def __call__(self, tensor):
        # Apply with 20% probability
        if random.random() < 0.2:
            c, h, w = tensor.shape
            x = int(random.uniform(0, w * 0.8))
            y = int(random.uniform(0, h * 0.8))
            width = int(w * self.enhancement_size)
            height = int(h * self.enhancement_size)

            # Create Gaussian-shaped enhancement region
            enhancement_mask = torch.ones_like(tensor)
            for i in range(width):
                for j in range(height):
                    if x + i < w and y + j < h:
                        dist = ((i - width / 2) ** 2 + (j - height / 2) ** 2) / (
                            (width / 2) ** 2 + (height / 2) ** 2
                        )
                        factor = 1.0 + (self.enhancement_intensity - 1.0) * np.exp(
                            -dist * 4
                        )
                        enhancement_mask[:, y + j, x + i] = factor

            enhanced = tensor * enhancement_mask
            return torch.clamp(enhanced, 0, 1)
        return tensor


class AutoROICropWithAspectRatio:
    """Automatic ROI detection and cropping for ultrasound images.

    Uses Otsu's thresholding with safety bounds to detect the diagnostic
    region in ultrasound images, then crops and resizes while maintaining
    the original aspect ratio with zero-padding.

    Args:
        target_size (tuple): Target output size (width, height).
        margin (int): Extra margin in pixels around the detected ROI.
    """

    def __init__(self, target_size=(224, 224), margin=10):
        self.target_size = target_size
        self.margin = margin

    def __call__(self, img):
        np_img = np.array(img)
        roi = self._auto_detect_roi(np_img)
        resized_roi = self._resize_maintain_ratio(roi)
        return Image.fromarray(resized_roi.astype("uint8"))

    def _auto_detect_roi(self, image):
        """Detect and crop the ROI using Otsu's algorithm with safety bounds."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Compute optimal threshold using Otsu's method
        otsu_thresh, _ = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Safety bounds: ultrasound backgrounds are typically very dark,
        # so the threshold should be between 10 and 60
        final_thresh = otsu_thresh
        if otsu_thresh < 10:
            final_thresh = 10
        elif otsu_thresh > 60:
            final_thresh = 60

        _, thresh = cv2.threshold(gray, final_thresh, 255, cv2.THRESH_BINARY)

        # Morphological operations to remove noise
        kernel = np.ones((5, 5), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

        # Find the largest connected component (assumed to be the diagnostic area)
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Add margin
            x = max(0, x - self.margin)
            y = max(0, y - self.margin)
            w = min(image.shape[1] - x, w + 2 * self.margin)
            h = min(image.shape[0] - y, h + 2 * self.margin)

            roi = image[y : y + h, x : x + w]
            return roi

        return image  # Return original if ROI detection fails

    def _resize_maintain_ratio(self, img):
        """Resize image to target size while maintaining aspect ratio with zero-padding."""
        h, w = img.shape[:2]
        target_w, target_h = self.target_size

        ratio = min(target_w / w, target_h / h)
        new_size = (int(w * ratio), int(h * ratio))

        resized = cv2.resize(img, new_size)

        # Create black canvas and center the resized image
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        x_offset = (target_w - new_size[0]) // 2
        y_offset = (target_h - new_size[1]) // 2
        canvas[
            y_offset : y_offset + new_size[1], x_offset : x_offset + new_size[0]
        ] = resized

        return canvas
