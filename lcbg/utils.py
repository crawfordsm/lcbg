import os
import re

import numpy as np

from scipy.interpolate import interp1d

from astropy.stats import gaussian_sigma_to_fwhm
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy import units as u

from matplotlib import pyplot as plt

from .fitting import fit_gaussian2d, plot_fit


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)


def angular_to_pixel(angular_diameter, wcs):
    pixel_scales = proj_plane_pixel_scales(wcs)
    assert np.allclose(*pixel_scales)
    pixel_scale = pixel_scales[0] * wcs.wcs.cunit[0] / u.pix

    pixel_size = angular_diameter / pixel_scale.to(angular_diameter.unit / u.pix)
    pixel_size = pixel_size.value

    return pixel_size


def elliptical_area_to_r(area, e):
    a = np.sqrt(e * area / (np.pi))
    b = a / e
    return a, b


def circle_area_to_r(area):
    return np.sqrt(area / (np.pi))


def get_interpolated_values(x, y, num=5000, kind='cubic'):
    f = interp1d(x, y, kind=kind)
    x_new = np.linspace(min(x), max(x), num=num, endpoint=True)
    y_new = f(x_new)
    return x_new, y_new


def closest_value_index(value, array):
    """Return first index closes to value"""
    idx_list = np.where(array <= value)[0]

    idx = None
    if idx_list.size > 0:
        idx = idx_list[0]
        idx = (array[:idx + 1] - value).argmin()
    return idx


def plot_target(position, image, size, vmin=None, vmax=None):
    x, y = position
    if not isinstance(image, np.ndarray):
        image = image.data
    plt.imshow(image, vmin=vmin, vmax=vmax)
    plt.plot(x, y, '+', c='r', label='Target')
    plt.xlim(x-size, x+size)
    plt.ylim(y-size, y+size)


def cutout(image, x, y, dx, dy=None, vmin=None, vmax=None):
    """
    Clip and make a cutout of an image

    Parameters
    ----------
    image : array like
        Input image.
    x, y : int
        Center of cutout. The indexing is array[y, x] and
        x is the x axis when plotted.
    dx, dy : int
        Size of image in x and y direction. The indexing is array[y, x] and
        x is the x axis when plotted. If dy is not provided it will be set
        to the same size as dx.
    vmin, vmax : float
        max and min values to clip the input image.


    Returns
    -------
    Copied array
        Clipped and cropped image.
    """

    if dy is None:
        dy = dx

    vmin = vmin if vmin else image.min()
    vmax = vmax if vmax else image.max()
    image = np.clip(image, vmin, vmax)

    bounds = np.array([y - dy // 2, y + dy // 2, x - dx // 2, x + dx // 2])
    bounds[bounds < 0] = 0
    ymin, ymax, xmin, xmax = bounds

    return image[ymin:ymax, xmin:xmax].copy()


def cutout_subtract(image, target, x, y):
    """
    Subtract cutout from image
    Parameters
    ----------
    image : array like
        Main image
    target : array like
        Cutout image
    x, y : int
        Center to subtract from

    Returns
    -------
    Copied array
        subtracted
    """

    dy, dx = target.shape
    bounds = np.array([y - dy // 2, y + dy // 2, x - dx // 2, x + dx // 2])
    bounds[bounds < 0] = 0
    ymin, ymax, xmin, xmax = bounds
    image[ymin:ymax, xmin:xmax] -= target
    return image


def measure_fwhm(image, plot=True, printout=True):
    """
    Find the 2D FWHM of a background/continuum subtracted cutout image of a target.
    The target should be centered and cropped in the cutout.
    Use lcbg.utils.cutout for cropping targets.
    FWHM is estimated using the sigmas from a 2D gaussian fit of the target's flux.
    The FWHM is returned as a tuple of the FWHM in the x and y directions.

    Parameters
    ----------
    image : array like
        Input background/continuum subtracted cutout image.
    printout : bool
        Print out info.
    plot : bool
        To plot fit or not.

    Returns
    -------
    tuple : array of floats
        FWHM in x and y directions.
    """

    # Find FWHM
    # ----------

    fitted_line = fit_gaussian2d(image)

    # Find fitted center
    x_mean, y_mean = [i.value for i in [fitted_line.x_mean, fitted_line.y_mean]]

    # Estimate FWHM using gaussian_sigma_to_fwhm
    x_fwhm = fitted_line.x_stddev * gaussian_sigma_to_fwhm
    y_fwhm = fitted_line.y_stddev * gaussian_sigma_to_fwhm

    # Find half max
    hm = fitted_line(x_mean, y_mean) / 2.

    # Find the mean of the x and y direction
    mean_fwhm = np.mean([x_fwhm, y_fwhm])
    mean_fwhm = int(np.round(mean_fwhm))

    # Print info about fit and FWHM
    # ------------------------------

    if printout:
        print("Image Max: {}".format(image.max()))
        print("Amplitude: {}".format(fitted_line.amplitude.value))
        print("Center: ({}, {})".format(x_mean, y_mean))
        print("Sigma = ({}, {})".format(fitted_line.x_stddev.value,
                                        fitted_line.y_stddev.value, ))

        print("Mean FWHM: {} Pix ".format(mean_fwhm))
        print("FWHM: (x={}, y={}) Pix ".format(x_fwhm, y_fwhm))

    if plot:

        fig, [ax0, ax1, ax2, ax3] = plot_fit(image, fitted_line)

        # Make x and y grid to plot to
        y_arange, x_arange = np.mgrid[:image.shape[0], :image.shape[1]]

        # Plot input image with FWHM and center
        # -------------------------------------

        ax0.imshow(image, cmap='gray_r')

        ax0.axvline(x_mean - x_fwhm / 2, c='c', linestyle="--", label="X FWHM")
        ax0.axvline(x_mean + x_fwhm / 2, c='c', linestyle="--")

        ax0.axhline(y_mean - y_fwhm / 2, c='g', linestyle="--", label="Y FWHM")
        ax0.axhline(y_mean + y_fwhm / 2, c='g', linestyle="--")

        ax0.set_title("Center and FWHM Plot")
        ax0.legend()

        # Plot X fit
        # ----------

        ax2.axvline(x_mean, linestyle="-", label="Center")
        ax2.axvline(x_mean - x_fwhm / 2, c='c', linestyle="--", label="X FWHM")
        ax2.axvline(x_mean + x_fwhm / 2, c='c', linestyle="--")
        ax2.axhline(hm, c="black", linestyle="--", label="Half Max")

        ax2.legend()

        # Plot Y fit
        # ----------

        ax3.axvline(y_mean, linestyle="-", label="Center")
        ax3.axvline(y_mean - y_fwhm / 2, c='g', linestyle="--", label="Y FWHM")
        ax3.axvline(y_mean + y_fwhm / 2, c='g', linestyle="--")
        ax3.axhline(hm, c="black", linestyle="--", label="Half Max")

        ax3.legend()

        plt.show()

    return np.array([x_fwhm, y_fwhm])


