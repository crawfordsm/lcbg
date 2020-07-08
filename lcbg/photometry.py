import numpy as np

from astropy.modeling import models, fitting, functional_models, Parameter, custom_model
from astropy import units as u

from lcbg.utils import cutout, measure_fwhm, plot_apertures
from lcbg.fitting import plot_fit, fit_model, model_subtract, Moffat2D, Nuker2D

from matplotlib import pyplot as plt

import ipywidgets as widgets
from IPython.display import display

plt.rcParams['figure.figsize'] = [12, 12]


def flux_to_abmag(flux, header):
    """Convert HST flux to AB Mag"""

    PHOTFLAM = header['PHOTFLAM']
    PHOTFNU = header['PHOTFNU']
    PHOTZPT = header['PHOTZPT']
    PHOTPLAM = header['PHOTPLAM']
    PHOTBW = header['PHOTBW']

    STMAG_ZPT = (-2.5 * np.log10(PHOTFLAM)) + PHOTZPT
    ABMAG_ZPT = STMAG_ZPT - (5. * np.log10(PHOTPLAM)) + 18.692

    return -2.5 * np.log10(flux) + ABMAG_ZPT


def order_cat(cat):
    """
    Sort a catalog by largest area and return the argsort
    Parameters
    ----------
    cat : `SourceCatalog` instance
        A `SourceCatalog` instance containing the properties of each
        source.

    Returns
    -------
    output : list
        A list of catalog indices ordered by largest area
    """
    table = cat.to_table()['area']
    order_all = table.argsort()
    order_all = list(reversed(order_all))
    return order_all



def remove_fitted_sources(image, image_residual, cat,
                          r_inner_mult=1,
                          r_outter_mult=7,
                          r_output_mult=20,
                          index_start=None, index_end=None,
                          n_models=2,
                          show_progress=True,
                          print_fit=False,
                          show_fit=True):
    """
    Fits and removes bright sersic profiles from image.

    Parameters
    ----------
    image : array
        Input image to remove sources from. This image should be noise subtracted
    image_residual : array
        Image resulting from subtracting the input image by all the catalog sources.
    cat : `SourceCatalog` instance
        A `SourceCatalog` instance containing the properties of each
        source.
    r_inner_mult : int
        Multiples the object radius to give the radius of the object's core. Should be >= 1.
    r_outter_mult : int
        Multiples the object radius to give the radius of the outer components. Should be >= 1.
    r_output_mult : int
        Multiples the object radius to give the radius of the final cutout that is subtracted from the image.
        Should be >= 1.
    index_start : int
        Index of the first source in the area sorted catalog. So if this is set to 0,
        it will start removing at the object with the largest area in the catalog.
    index_end : int
        Index of last source in the area sorted catalog.
    n_models : int
        Number of models to use for fitting
        1 = Sersic2D
        2 = Sersic2D_core + Sersic2D_outter
        3 = Sersic2D_core + Sersic2D_outter + models.Gaussian2D
        all models are centered around the same pixel (the core).
    show_progress : bool
        Show jupyter notebook progress bar
    print_fit : bool
        Print fit params
    show_fit : bool
        Plot fit

    Returns
    -------
    subtracted_image : array
        An image with the fitted sources subtracted from it.

    subtracted_image_residual : array
        The input residual image with the fitted sources subtracted from it.

    fitted_sources : array
        An image showing only the fitted sources from the models.
    """

    order = order_cat(cat)

    order = order[index_start:index_end]

    if show_progress:
        pb = widgets.IntProgress(
            value=0,
            min=0,
            max=len(order),
            step=1,
            description='Loading:',
            bar_style='',
            orientation='horizontal'
        )

        display(pb)

    # Copy image for output
    image_copy = image.copy()
    image_residual_copy = image_residual.copy()
    image_zero = np.zeros_like(image_copy)
    photometric_sums = {}

    for i, cat_index in enumerate(order):
        if show_fit or print_fit:
            print(index_start + i, cat_index)

        if show_progress:
            # Update progress bar
            pb.value = i + 1
            pb.description = "{}/{}".format(pb.value, len(order))

        # Load object and aperture
        obj = cat[cat_index]

        # Estimate center of bobject
        cut = obj.segment.make_cutout(image, masked_array=True)
        cy, cx = np.unravel_index(cut.argmax(), cut.shape)

        x = obj.segment.bbox.ixmin + cx
        y = obj.segment.bbox.iymin + cy

        # Estimate other parts
        theta = obj.orientation.to(u.rad).value
        ellip = (1 - obj.semiminor_axis_sigma.value / obj.semimajor_axis_sigma.value)
        amp = cut.data.max()

        # Define fitting radius
        r_object = int(np.round(obj.semimajor_axis_sigma.value))  # inner cutout radius

        r_inner = r_object * r_inner_mult

        r_outter = r_object * r_outter_mult

        # Make images to fit
        target_zoom = cutout(image_copy, x, y, r_inner)
        target = cutout(image_copy, x, y, r_outter)
        target_residual = cutout(image_residual_copy, x, y, r_outter)

        # Fit inner core
        # --------------

        # Find center of zoomed cutout image
        y_0, x_0 = np.array(target_zoom.shape) // 2

        # Make inner model
        xy_slack = 10  # x and y value range / 2

        model_1 = models.Sersic2D(
            amplitude=amp,
            n=2,
            r_eff=r_inner,
            ellip=ellip,
            theta=theta,
            x_0=x_0,
            y_0=y_0,
            # fixed={'theta':True, 'ellip':True},
            bounds={
                'amplitude': (0, None),
                'r_eff': (0, None),
                'n': (0, 6),
                'ellip': (0, 1),
                'theta': (0, 2 * np.pi),
                'x_0': (x_0 - xy_slack, x_0 + xy_slack),
                'y_0': (y_0 - xy_slack, y_0 + xy_slack),
            })

        # Fit core model to zoomed cutout to tighten first guess
        model_1, fit = fit_model(target_zoom, model_1, maxiter=10000, epsilon=1e-40)

        # Fix the center x and y of all models to match the fitted guess
        """
        for pn in model.param_names:
            model.fixed[pn] = True            
        """

        model_1.fixed.update({
            'x_0': True,
            'y_0': True,
        })

        del model_1.bounds['x_0']
        del model_1.bounds['y_0']

        # Fit glow around target
        # ----------------------

        # Remove old center and add new center
        y_0, x_0 = np.array(target_zoom.shape) // 2

        model_1.x_0 -= x_0
        model_1.y_0 -= y_0

        y_0, x_0 = np.array(target.shape) // 2

        model_1.x_0 += x_0
        model_1.y_0 += y_0

        # Setup second model for fitting
        xy_slack = 5

        model_2 = models.Sersic2D(
            amplitude=0,
            n=0.1,
            r_eff=model_1.r_eff * 3,
            ellip=ellip,
            theta=theta,
            x_0=model_1.x_0,
            y_0=model_1.y_0,
            fixed={'x_0': True, 'y_0': True, 'theta': True, 'ellip': True},
            bounds={
                'amplitude': (0, None),
                'r_eff': (0, None),
                'n': (0, 2),
                'ellip': (0, 1),
                'theta': (0, 2 * np.pi),
            })

        # Fit second model to the residual of the image (image with segmented area masked)
        model_2, fits = fit_model(target_residual, model_2, maxiter=10000, epsilon=1e-40)

        # PSF Models
        # ----------

        # Setup PSF image and fit it to image of target to estimate param guess

        # Normal sources
        model_3 = models.Gaussian2D(
            amplitude=amp,
            x_mean=model_1.x_0,
            y_mean=model_1.y_0,
            x_stddev=model_1.r_eff,
            y_stddev=model_1.r_eff,
            fixed={'x_mean': True, 'y_mean': True, }
        )

        model_3, fits = fit_model(target, model_3, maxiter=10000, epsilon=1e-40)

        # Final Fit
        # ---------

        # Combine models
        if n_models == 1:
            model = model_1
        elif n_models == 2:
            model = model_1 + model_2
        elif n_models == 3:
            model = model_1 + model_2 + model_3
        else:
            raise Exception("n_models should be 1, 2 or 3")

        # Fit combined model
        model, fit = fit_model(target, model, maxiter=10000, epsilon=1e-40)
        model, fit = fit_model(target, model, maxiter=10000, epsilon=1e-40)

        # Make center of model the center of model_1
        model.x_0 = model_1.x_0
        model.y_0 = model_1.y_0

        # Plot and Print
        # --------------
        if show_fit:
            fig, ax = plot_fit(target, model)  # , vmin=vmin, vmax=vmax)
            plt.show()

        if print_fit:
            print("\n".join([str(j) for j in zip(model.param_names, model.parameters)]) + "\n" * 3)

        # Subtract from image
        # -------------------

        if fit.fit_info['ierr'] > 4:
            # Fit failed
            continue

        # Make new image
        size = r_inner * r_output_mult
        y_arange, x_arange = np.mgrid[
                             int(model.y_0.value) - size:int(model.y_0.value) + size,
                             int(model.x_0.value) - size:int(model.x_0.value) + size, ]
        model_image = model(x_arange, y_arange)

        # Subtract cutout from main copy image
        image_copy = model_subtract(image_copy, np.array(model_image), x, y)
        image_copy = np.clip(image_copy, 0, image_copy.max())
        #image_copy[y - 8:y + 8, x - 8:x + 8] = np.nan

        # Add cutout to main cutout only image
        image_zero = model_subtract(image_zero, -1 * np.array(model_image), x, y)

        # Subtract cutout second component from residual image
        image_residual_copy = model_subtract(image_residual_copy, np.array(model_image), x, y)
        image_residual_copy = np.clip(image_residual_copy, 0, image_residual_copy.max())

        # Photometry
        photometric_sums[obj.label] = model_image.sum()

    return image_copy, image_residual_copy, image_zero, photometric_sums