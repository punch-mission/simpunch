import os
from glob import glob

import astropy.units as u
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from dateutil.parser import parse as parse_datetime


def keynote_figs():
    matplotlib.rcParams['lines.color'] = 'white'
    matplotlib.rcParams['patch.edgecolor'] = 'white'
    matplotlib.rcParams['text.color'] = 'white'
    matplotlib.rcParams['axes.facecolor'] = 'black'
    matplotlib.rcParams['axes.edgecolor'] = 'white'
    matplotlib.rcParams['axes.labelcolor'] = 'white'
    matplotlib.rcParams['xtick.color'] = 'white'
    matplotlib.rcParams['ytick.color'] = 'white'
    matplotlib.rcParams['grid.color'] = 'white'
    matplotlib.rcParams['figure.facecolor'] = 'black'
    matplotlib.rcParams['figure.edgecolor'] = 'black'
    matplotlib.rcParams['savefig.facecolor'] = 'black'
    matplotlib.rcParams['savefig.edgecolor'] = 'black'
    matplotlib.rcParams['font.size'] = 10
    matplotlib.rcParams['lines.linewidth'] = 1.5
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['text.usetex'] = False


def light_figs():
    matplotlib.rcParams['lines.color'] = 'black'
    matplotlib.rcParams['patch.edgecolor'] = 'black'
    matplotlib.rcParams['text.color'] = 'black'
    matplotlib.rcParams['axes.facecolor'] = 'white'
    matplotlib.rcParams['axes.edgecolor'] = 'black'
    matplotlib.rcParams['axes.labelcolor'] = 'black'
    matplotlib.rcParams['xtick.color'] = 'black'
    matplotlib.rcParams['ytick.color'] = 'black'
    matplotlib.rcParams['grid.color'] = 'black'
    matplotlib.rcParams['figure.facecolor'] = 'white'
    matplotlib.rcParams['figure.edgecolor'] = 'white'
    matplotlib.rcParams['savefig.facecolor'] = 'white'
    matplotlib.rcParams['savefig.edgecolor'] = 'white'
    matplotlib.rcParams['font.size'] = 10
    matplotlib.rcParams['lines.linewidth'] = 1.5
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['text.usetex'] = False


def radial_dist(h, w, center=None, radius=None):
    if center is None:
        center = (int(w/2), int(h/2))

    if radius is None:
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_arr = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    return dist_arr / dist_arr.max()


def _plot_image(path, title, image, wcs, vmin=0, vmax=5E-9):
    rarr = radial_dist(*image.shape)
    scaled_image = image * (rarr / 4) ** 2.5
    ax = plt.subplot(111, projection=wcs, slices=('x', 'y', 0))
    plt.imshow(scaled_image, cmap='magma', vmin=vmin, vmax=vmax)
    # plt.contour(rarr_qp < 0.68, alpha=0.3, colors='C0', linewidths=3)
    lon, lat, _ = ax.coords
    lat.set_ticks(np.arange(-90, 90, 10) * u.degree)
    lon.set_ticks(np.arange(-180, 180, 10) * u.degree)
    lat.set_major_formatter('dd')
    lon.set_major_formatter('dd')
    ax.set_facecolor('black')
    ax.coords.grid(color='white', alpha=.1)
    plt.xlabel("Helioprojective longitude")
    plt.ylabel("Helioprojective latitude")
    plt.scatter(0, 0, s=480, color='k', transform=ax.get_transform('world'))
    plt.title(title)
    plt.savefig(path, dpi=300)
    plt.close()


def plot(directory, glob_pattern, mode="keynote", vmin=0, vmax=5E-9):
    if mode == "keynote":
        keynote_figs()
    elif mode == "light":
        light_figs()
    else:
        raise ValueError("Invalid mode. Must be `keynote` or `light`.")

    paths = sorted(glob(os.path.join(directory, glob_pattern)))

    for path in paths:
        with fits.open(path) as hdul:
            total_brightness = hdul[1].data[0]
            polarized_brightness = hdul[1].data[1]
            wcs = WCS(hdul[1].header, hdul)
            time_obs = parse_datetime(hdul[1].header['DATE-OBS'])
            title = 'Total brightness - ' + time_obs.strftime('%Y/%m/%d %H:%M:%S' + 'UT')
            _plot_image(path.replace(".fits", "_total.png"),
                        title,
                        total_brightness, wcs, vmin=vmin, vmax=vmax)
            title = 'Polarized brightness - ' + time_obs.strftime('%Y/%m/%d %H:%M:%S' + 'UT')
            _plot_image(path.replace(".fits", "_polarized.png"),
                        title,
                        polarized_brightness, wcs, vmin=vmin, vmax=vmax)


if __name__ == "__main__":
    # plot("/Users/jhughes/Desktop/data/gamera_mosaic_jan2024/synthetic_L3/products/L3/PAN/",
    #      "*_PAN_*.fits", vmin=0, vmax=0.001)
    plot("/Users/jhughes/Desktop/data/gamera_mosaic_jan2024/synthetic_L3/products/L3/PAM/",
         "*_PAM_*.fits")
    plot("/Users/jhughes/Desktop/data/gamera_mosaic_jan2024/synthetic_L3/products/L3/PTM/",
         "*_PTM_*.fits")
