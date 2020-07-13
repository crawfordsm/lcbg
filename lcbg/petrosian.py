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


def calculate_r_total_flux(r_list, area_list, flux_list, eta=0.2):
    r_petrosian = calculate_petrosian_r(r_list, area_list, flux_list, eta=eta)
    if r_petrosian is None:
        print("r_petrosian could not be computed")
        return np.nan

    return r_petrosian * 2


def calculate_r_half_light(r_list, area_list, flux_list, eta=0.2):
    r_total_flux = calculate_r_total_flux(r_list, area_list, flux_list, eta=eta)

    if r_total_flux > max(r_list):
        return None

    f = interp1d(r_list, flux_list, kind='cubic')
    total_flux = f(r_total_flux)
    half_flux = total_flux / 2.

    r_list_new, flux_list_new = get_interpolated_values(r_list, flux_list)

    idx = abs(flux_list_new - half_flux).argmin()
    return None if idx is None else r_list_new[idx]