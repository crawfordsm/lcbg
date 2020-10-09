import numpy as np

from scipy.interpolate import interp1d

from matplotlib import pyplot as plt

from .utils import closest_value_index, get_interpolated_values


def plot_petrosian(r_list, area_list, flux_list, eta=0.2):
    petrosian_list = calculate_petrosian(area_list, flux_list)

    r_list_new, petrosian_list_new = get_interpolated_values(r_list, petrosian_list)

    plt.plot(r_list, petrosian_list, marker='o', linestyle='None', label='Data')
    plt.plot(r_list_new, petrosian_list_new, label='Interpolated [cubic]')

    r_petrosian = calculate_petrosian_r(r_list, area_list, flux_list, eta=eta)
    if r_petrosian is not None:
        plt.axvline(r_petrosian, linestyle='--', label="r_petrosian={:0.4f} pix".format(r_petrosian))
        plt.axhline(eta, linestyle='--', label='Eta={:0.4f}'.format(eta))
    else:
        r_petrosian = 0

    plt.legend(loc='best')

    plt.title("Petrosian")
    plt.xlabel("Aperture Radius [Pix]")
    plt.ylabel("Petrosian Value")


def calculate_petrosian(area_list, flux_list):
    petrosian_list = []

    last_area = 0
    last_I = 0
    for i in range(len(area_list)):
        area = area_list[i]
        I = flux_list[i]

        area_of_slice = area - last_area
        I_at_r = (I - last_I) / area_of_slice

        area_within_r = area
        I_avg_within_r = (I / area_within_r)

        petrosian_value = I_at_r / I_avg_within_r

        petrosian_list.append(petrosian_value)

        last_area = area
        last_I = I

    return np.array(petrosian_list)


def calculate_petrosian_r(r_list, area_list, flux_list, eta=0.2):
    petrosian_list = calculate_petrosian(area_list, flux_list)

    r_list_new, petrosian_list_new = get_interpolated_values(r_list, petrosian_list)

    idx = closest_value_index(eta, petrosian_list_new)

    return None if idx is None else r_list_new[idx]


def discrete_petrosian_r(r_list, area_list, flux_list, eta=0.2):
    petrosian_list = calculate_petrosian(area_list, flux_list)
    idx_list = np.where(petrosian_list <= eta)[0]

    r_petrosian = None
    if idx_list.size > 0:
        idx = idx_list[0]
        r_petrosian = r_list[idx]

    return r_petrosian


def calculate_r_total_flux(r_list, area_list, flux_list, eta=0.2, verbose=False):
    r_petrosian = calculate_petrosian_r(r_list, area_list, flux_list, eta=eta)
    if r_petrosian is None:
        if verbose:
            print("r_petrosian could not be computed")
        return np.nan

    return r_petrosian * 2


def fraction_flux_to_r(r_list, area_list, flux_list, fraction=0.5, eta=0.2):
    r_total_flux = calculate_r_total_flux(r_list, area_list, flux_list, eta=eta)

    if r_total_flux > max(r_list):
        return None

    f = interp1d(r_list, flux_list, kind='cubic')
    total_flux = f(r_total_flux)
    fractional_flux = total_flux * fraction

    r_list_new, flux_list_new = get_interpolated_values(r_list, flux_list)

    # idx = abs(flux_list_new - fractional_flux).argmin()
    idx = closest_value_index(fractional_flux, flux_list_new, growing=True)
    return None if idx is None else r_list_new[idx]


def calculate_r_half_light(r_list, area_list, flux_list, eta=0.2):
    return fraction_flux_to_r(r_list, area_list, flux_list, fraction=0.5, eta=eta)


def calculate_concentration_index(r_list, area_list, flux_list, ratio1=0.2, ratio2=0.8, eta=0.2):
    r_total_flux = calculate_r_total_flux(r_list, area_list, flux_list, eta=eta)

    if r_total_flux > max(r_list):
        return None

    r1 = fraction_flux_to_r(r_list, area_list, flux_list, fraction=ratio1, eta=eta)
    r2 = fraction_flux_to_r(r_list, area_list, flux_list, fraction=ratio2, eta=eta)

    if None in [r1, r2]:
        return None

    return r1, r2, 5 * np.log10(r2 / r1)


def estimate_n(c2080pet, verbose=False):
    n_list = [0.5, 0.75, 1, 1.5, 2, 4, 6, 8]
    c_pet_list = [2.14, 2.49, 2.78, 3.26, 3.63, 4.50, 4.99, 5.31]
    f = interp1d(c_pet_list, n_list, kind='cubic')
    try:
        return f(c2080pet)
    except ValueError:
        if verbose:
            print("Could not estimate n for {}, returning closest".format(c2080pet))
        return 0.5 if c2080pet < 2.14 else 5.31


