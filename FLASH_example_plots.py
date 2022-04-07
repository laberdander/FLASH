import sys
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd


# sys.path.insert(0, '/lustre/phx/lwegert/Data_Analysis')

import FLASH_PLOT


dict = {
    'dens_xmin': -20,
    'dens_xmax': 0,
    'dens_ymin': 0,
    'dens_ymax': 8,
    'dens_scale': 'linear',
    'tele_xmin': 0,
    'tele_xmax': 400,
    'tele_ymin': 0,
    'tele_ymax': 700,
    'tele_scale': 'linear',
    'nele_xmin': 0,
    'nele_xmax': 400,
    'nele_ymin': 1e18,
    'nele_ymax': 1e23,
    'nele_scale': 'log',
    'ye_xmin': -20,
    'ye_xmax': 100,
    'ye_ymin': 0,
    'ye_ymax': 100,
}


path1 = '/lustre/phx/lwegert/WorkDirectory/2D_Test_Ref7/lasslab_hdf5_plt_cnt_????'
path2 = 'D:/Simulation/FLASH/2D/testcase/testcase_2d_cylindrical/'
path3 = 'D:/Simulation/FLASH/2D/testcase/testcase_2d_cartesian/'
path4 = 'D:/Simulation/FLASH/1D/testcase/'
path5 = 'D:/Simulation/HELIOS/testcase/testcase_1d/'
name = 'lasslab_hdf5_plt_cnt_????'


def plot2d(path, file, variable, time=None, **kwargs):
    fig, ax = plt.subplots()
    plotter2d = FLASH_PLOT.FlashPlot2D(path+file, time=time, **kwargs)
    frb = plotter2d.data_2d()
    plotter2d.plot_2d(frb, variable, ax)
    plt.show()
    # plotter2d.save_plot(fig, path+variable+'_2d_'+str(round(time, 1))+'ns.png')


def plot1d(path, file, variable, time=None, slice=0, ax=None, grid='cartesian', **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        pass
    if time is None:
        plotter2d = FLASH_PLOT.FlashPlot2D(path+file, scale=1, grid=grid)
    else:
        plotter2d = FLASH_PLOT.FlashPlot2D(path+file, scale=1, time=time, grid=grid)
    ray = plotter2d.data_1d(r_slice=slice)
    plotter2d.plot_1d(ray, variable, ax, **kwargs)
    ax.set_xlim(dict[variable+'_xmin'], dict[variable+'_xmax'])
    ax.set_ylim(dict[variable+'_ymin'], dict[variable+'_ymax'])
    ax.set_yscale(dict[variable+'_scale'])
    return ax


def plot1d_from1dsim(path, file, variable, time=None, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt. subplots()
    else:
        pass
    if time is None:
        plotter1d = FLASH_PLOT.FlashPlot1D(path+file)
    else:
        plotter1d = FLASH_PLOT.FlashPlot1D(path + file, time=time)
    ray = plotter1d.data_1d()
    plotter1d.plot_1d(ray, variable, ax, **kwargs)
    ax.set_xlim(dict[variable + '_xmin'], dict[variable + '_xmax'])
    ax.set_ylim(dict[variable + '_ymin'], dict[variable + '_ymax'])
    ax.set_yscale(dict[variable + '_scale'])
    return ax


def import_helios(path, file, add_x_offset=-20):
    data = pd.read_csv(path+file, header=None, delim_whitespace=True, comment='#').to_numpy()
    x = data[:, 0] * 1e4 + add_x_offset
    y = data[:, 1]
    return x, y


def plot_helios(variable, ax, **kwargs):
    x, y = import_helios(path5, variable+'_1ns.ppd')
    ax.plot(x, y, **kwargs)
    ax.set_xlim(dict[variable + '_xmin'], dict[variable + '_xmax'])
    ax.set_ylim(dict[variable + '_ymin'], dict[variable + '_ymax'])
    ax.set_yscale(dict[variable + '_scale'])


def compare_cyl2d_cart2d_1d_helios(variable):
    ax1 = plot1d(path3, name, variable, time=1, slice=5, grid='cartesian', label='cartesian (2D)')
    plot1d(path2, name, variable, time=1, slice=5, ax=ax1, grid='cylindrical', label='cylindrical (2D)')
    plot1d_from1dsim(path4, name, variable, time=1, ax=ax1, label='1D')
    plot_helios(variable, ax1, label='HELIOS')
    plt.legend()
    plt.savefig('D:/Simulation/testcase_results/cyl_vs_cart_vs_1d_helios_'+variable+'.png')


def compare_cyl2d_cart2d(variable):
    ax1 = plot1d(path3, name, variable, time=1, slice=0, grid='cartesian', label='cartesian (0 µm)')
    plot1d(path2, name, variable, time=1, slice=0, ax=ax1, grid='cylindrical', label='cylindrical (0 µm)')
    plot1d(path3, name, variable, time=1, slice=5, ax=ax1, grid='cartesian', label='cartesian (5 µm)')
    plot1d(path2, name, variable, time=1, slice=5, ax=ax1, grid='cylindrical', label='cylindrical (5 µm)')
    plt.legend()
    plt.savefig('D:/Simulation/testcase_results/cyl_vs_cart_'+variable+'_different_slices.png')


# compare_cyl2d_cart2d('nele')
# compare_cyl2d_cart2d('dens')
# compare_cyl2d_cart2d('tele')

compare_cyl2d_cart2d_1d_helios('nele')
compare_cyl2d_cart2d_1d_helios('dens')
compare_cyl2d_cart2d_1d_helios('tele')
