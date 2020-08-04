import numpy as np

from astropy.io import fits
from astropy import units as u
from astropy import cosmology

from .utils import angular_to_pixel


def cosmo_aperture_diameter(aperture_diameter, z, wcs, cosmo=cosmology.WMAP5):
    """
    Compute the aperture diameter in pix given the physical aperture diameter at a redshift.

    Parameters
    ----------
    aperture_diameter : astropy.units.quantity.Quantity
        Aperture diameter with units
    z : float
        Redshift of target
    wcs : astropy.wcs.wcs.WCS
        WCS of the image
    cosmo : astropy.cosmology.core.FlatLambdaCDM
        Comsology to use for computation. `cosmology.WMAP5` by default

    Returns
    -------
    pixel_size : float
        returns pixel size
    """

    angular_diameter = (cosmo.arcsec_per_kpc_proper(z) * aperture_diameter.to('kpc')).to('arcsec')

    return angular_to_pixel(angular_diameter, wcs)