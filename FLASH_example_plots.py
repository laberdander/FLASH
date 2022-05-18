import sys
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
# import pandas as pd
import scipy.constants as const
from scipy import interpolate
from scipy import optimize


sys.path.insert(0, '/lustre/phx/lwegert/Data_Analysis')
import FLASH_PLOT


name = 'lasslab_hdf5_plt_cnt_????'

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
    'nele_xmin': -20,
    'nele_xmax': 400,
    'nele_ymin': 1e18,
    'nele_ymax': 1e24,
    'nele_scale': 'log',
    'ye_xmin': -20,
    'ye_xmax': 100,
    'ye_ymin': 0,
    'ye_ymax': 100,
    'depo_xmin': -20,
    'depo_xmax': 400,
    'depo_ymin': 0,
    'depo_ymax': 1e12,
    'depo_scale': 'linear',
    'pres_xmin': -20,
    'pres_xmax': 400,
    'pres_scale': 'linear',
    'pres_ymin': 0,
    'pres_ymax': 3
}


def plot2d(path, file, variable, time=None, **kwargs):
    fig, ax = plt.subplots()
    plotter2d = FLASH_PLOT.FlashPlot2D(path+file, time=time, **kwargs)
    frb = plotter2d.data_2d()
    plotter2d.plot_2d(frb, variable, ax)
    plt.show()
    # plotter2d.save_plot(fig, path+variable+'_2d_'+str(round(time, 1))+'ns.png')


def load_flash(path, file, time=None, slice=0, grid='cartesian'):
    if time is None:
        plotter2d = FLASH_PLOT.FlashPlot2D(path+file, scale=1, grid=grid)
    else:
        plotter2d = FLASH_PLOT.FlashPlot2D(path+file, scale=1, time=time, grid=grid)
    ray = plotter2d.data_1d(r_slice=slice)
    return plotter2d, ray


def plot1d(variable, plotter2d, ray, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        pass
    ray = ray
    plotter2d = plotter2d
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


# def import_helios(path, variable, add_x_offset=-20):
#     data = pd.read_csv(path+variable+'_1ns.ppd', header=None, delim_whitespace=True, comment='#').to_numpy()
#     x = data[:, 0] * 1e4 + add_x_offset
#     y = data[:, 1]
#     return x, y


def find_critical_density(plotter2d, ray, wavelength=5.27e-7):
    x, nele = plotter2d.data_numpy_1d(ray, 'nele')
    ang_freq = const.c / wavelength * 2*const.pi
    crit_dens = const.epsilon_0 * const.electron_mass / const.elementary_charge**2 * ang_freq**2 * 1e-6
    f = interpolate.interp1d(x, nele - crit_dens)
    sol = optimize.root_scalar(f, bracket=[0, 100])
    print('Critical Density (' + str(crit_dens) + ' 1/cc) at x = ' + str(sol.root) + ' um')
    return sol.root, crit_dens


def find_max_temp(plotter2d, ray):
    x, tele = plotter2d.data_numpy_1d(ray, 'tele')
    ind_tele_max = np.argmax(tele)
    x_tele_max = x[ind_tele_max]
    tele_max = np.max(tele)
    print('Maximum electron temperature: ' + str(tele_max) + ' eV, at x: ' + str(x_tele_max) + ' um')
    return x_tele_max, tele_max


def find_shock_wave_pos(plotter2d, ray):
    x, dens = plotter2d.data_numpy_1d(ray, 'dens')
    ind_dens_max = np.argmax(dens)
    x_dens_max = x[ind_dens_max]
    dens_max = dens[ind_dens_max]
    x = x[:(ind_dens_max+10)]
    dens = dens[:(ind_dens_max+10)]
    spl = interpolate.InterpolatedUnivariateSpline(x, dens, k=3)
    x_dens = np.linspace(-20, x[-1], 1000)
    ind_shock_pos = np.argmax(spl.derivative()(x_dens))
    x_shock_pos = x_dens[ind_shock_pos]
    dens_shock_pos = spl(x_shock_pos)
    print('Position of shock wave: ' + str(x_shock_pos) + ' um    (density: ' + str(dens_shock_pos) + ' g/cc)')
    print('Maximum density: ' + str(dens_max) + ' g/cc)')
    return x_shock_pos, dens_shock_pos, x_dens_max, dens_max


def find_max_pressure(plotter2d, ray):
    x, pres = plotter2d.data_numpy_1d(ray, 'pres')
    ind_pres_max = np.argmax(pres)
    pres_max = pres[ind_pres_max] *1e-12
    x_pres_max = x[ind_pres_max]
    print('Maximum Pressure: ' + str(pres_max) + ' Mbar, at x: ' + str(x_pres_max) + ' um')
    return x_pres_max, pres_max


def analyse_shock_pos(plotter2d, ray, ax, plot=True):
    shock_pos, dens_shock_pos, x_dens_max, dens_max = find_shock_wave_pos(plotter2d, ray)
    x_max_pres, max_pres = find_max_pressure(plotter2d, ray)
    if plot:
        ax.hlines(dens_max, dict['dens_xmin'], x_dens_max, linestyles='dashed')
        ax.vlines(shock_pos, dict['dens_ymin'], dens_shock_pos, linestyles='dashed')
        ax.vlines(x_max_pres, dict['dens_ymin'], dict['dens_ymax'], linestyles='dashed', colors='red')
        ax.annotate('Shock position:  ' + str(round(shock_pos, 1)) + r'$ \mu m$',
                    xy=(shock_pos, dens_shock_pos),
                    xytext=(shock_pos+0.5, 0.3))
        ax.annotate(str(round(dens_max, 2)) + ' g/cc',
                    xy=(x_dens_max, dens_max),
                    xytext=(dict['dens_xmin']+1, dens_max + (dict['dens_ymax']-dict['dens_ymin'])*0.05))
        ax.annotate('Max Pressure:  ' + str(round(max_pres, 1)) + ' Mbar',
                    xy=(x_max_pres, dens_max),
                    xytext=(x_max_pres + (dict['dens_xmax']-dict['dens_xmin'])*0.05, dict['dens_ymax']-0.5)
                    )


def analyse_crit_dens(plotter2d, ray, ax, plot=True):
    x_crit_dens, crit_dens = find_critical_density(plotter2d, ray)
    if plot:
        ax.hlines(crit_dens, dict['nele_xmin'], x_crit_dens, linestyles='dashed')
        ax.vlines(x_crit_dens, dict['nele_ymin'], crit_dens, linestyles='dashed')
        ax.annotate(r'$z_{crit} = $' + str(round(x_crit_dens, 1)) + r'$ \mu m$',
                    xy=(x_crit_dens, crit_dens),
                    xytext=(x_crit_dens + 3, dict['nele_ymin'] + dict['nele_ymin']))


def analyse_max_temp(plotter2d, ray, ax, plot=True):
    x_max_temp, max_temp = find_max_temp(plotter2d, ray)
    if plot:
        ax.hlines(max_temp, dict['tele_xmin'], x_max_temp, linestyles='dashed')
        ax.vlines(x_max_temp, dict['tele_ymin'], max_temp, linestyles='dashed')
        ax.annotate(r'$T_{max} = $' + str(round(max_temp, 0)),
                    xy=(x_max_temp, max_temp),
                    xytext=(x_max_temp - 10, max_temp + 20))


def dataset_analysis(path_sim_data, path_results, label='cartesian (2D)'):
    plotter2d, ray_1ns = load_flash(path_sim_data, name, time=1, slice=0, grid='cartesian')
    ax_dens = plot1d('dens', plotter2d, ray_1ns, label=label)
    analyse_shock_pos(plotter2d, ray_1ns, ax_dens)
    plt.savefig(path_results+'dens_cart_1D.png')
    ax_nele = plot1d('nele', plotter2d, ray_1ns, label=label)
    analyse_crit_dens(plotter2d, ray_1ns, ax_nele)
    plt.savefig(path_results+'nele_cart_1D.png')
    ax_tele = plot1d('tele', plotter2d, ray_1ns, label=label)
    analyse_max_temp(plotter2d, ray_1ns, ax_tele)
    plt.savefig(path_results+'tele_cart_1D.png')


path_testcase_3720_prop_6800 = '/lustre/phx/lwegert/WorkDirectory/testcase_2d_cartesian_3720_Prop_6800/'
path_testcase_3720_prop_6800_results = '/u/lwegert/WorkDirectory/Data_Analysis/testcase_2d_cartesian_3720_Prop_6800/'

path_testcase_3720_flash_6800 = '/lustre/phx/lwegert/WorkDirectory/testcase_2d_cartesian_3720_FLASH_6800/'
path_testcase_3720_flash_6800_results = '/u/lwegert/WorkDirectory/Data_Analysis/testcase_2d_cartesian_3720_FLASH_6800/'

path_testcase_flash_prop_6800 = '/lustre/phx/lwegert/WorkDirectory/testcase_2d_cartesian_FLASH_Prop_6800/'
path_testcase_flash_prop_6800_results = '/u/lwegert/WorkDirectory/Data_Analysis/testcase_2d_cartesian_FLASH_Prop_6800/'


dataset_analysis(path_testcase_3720_prop_6800, path_testcase_3720_prop_6800_results)
dataset_analysis(path_testcase_3720_flash_6800, path_testcase_3720_flash_6800_results)
dataset_analysis(path_testcase_flash_prop_6800, path_testcase_flash_prop_6800_results)


'''
BEISPIEL: Verschiedene Code Ausführungen
'''

# path1 = '/lustre/phx/lwegert/WorkDirectory/2D_Test_Ref7/lasslab_hdf5_plt_cnt_????'
# path2 = 'D:/Simulation/FLASH/2D/testcase/testcase_2d_cylindrical/'
# path3 = 'D:/Simulation/FLASH/2D/testcase/testcase_2d_cartesian/'
# path_1D_6Groups = 'D:/Simulation/FLASH/1D/testcase/EnergyGroups6/'
# path_1D_50Groups = 'D:/Simulation/FLASH/1D/testcase/EnergyGroups50/'
# path5 = 'D:/Simulation/HELIOS/testcase/testcase_1d/'
#
#
# path_testcase_flash_prop_6800 = 'D:/Simulation/FLASH/2D/testcase/testcase_2d_cartesian_FLASH_Prop_6800/'
# path_testcase_flash_prop_6800_results = 'D:/Simulation/FLASH/2D/testcase/testcase_2d_cartesian_FLASH_Prop_6800/'
#
# path_testcase_3720_flash_6800 = 'D:/Simulation/FLASH/2D/testcase/testcase_2d_cartesian_3720_FLASH_6800/'
# path_testcase_3720_flash_6800_results = 'D:/Simulation/FLASH/2D/testcase/testcase_2d_cartesian_3720_FLASH_6800/'
#
#
#
# dataset_analysis(path_testcase_3720_flash_6800, path_testcase_3720_flash_6800_results)

# plotter2d, ray = load_flash(path3, name, time=1, slice = 0, grid = 'cartesian')
# # find_max_temp(plotter2d, ray)
# x, dens = plotter2d.data_numpy_1d(ray, 'dens')
#
# x_dens = np.linspace(-20, 0, 1000)
#
# ax = plot1d('dens', plotter2d, ray)
# spl = find_shock_wave_pos(plotter2d, ray)
# ax.plot(x_dens, spl(x_dens))
# plt.show()


# def plot_helios(variable, ax, **kwargs):
#     x, y = import_helios(path5, variable+'_1ns.ppd')
#     ax.plot(x, y, **kwargs)
#     ax.set_xlim(dict[variable + '_xmin'], dict[variable + '_xmax'])
#     ax.set_ylim(dict[variable + '_ymin'], dict[variable + '_ymax'])
#     ax.set_yscale(dict[variable + '_scale'])


# def compare_cyl2d_cart2d_1d_helios(variable):
#     ax1 = plot1d(path3, name, variable, time=1, slice=5, grid='cartesian', label='cartesian (2D)')
#     plot1d(path2, name, variable, time=1, slice=5, ax=ax1, grid='cylindrical', label='cylindrical (2D)')
#     plot1d_from1dsim(path_1D_6Groups, name, variable, time=1, ax=ax1, label='1D 6 Energy Groups')
#     plot1d_from1dsim(path_1D_50Groups, name, variable, time=1, ax=ax1, label='1D 50 Energy Groups')
#     plot_helios(variable, ax1, label='HELIOS')
#     plt.legend()
#     plt.show()
    # plt.savefig('D:/Simulation/testcase_results/cyl_vs_cart_vs_1d_helios_'+variable+'.png')


# def compare_energy_groups(variable):
#     ax1 = plot1d_from1dsim(path_1D_6Groups, name, variable, time=1, label='6 Energy Groups')
#     plot1d_from1dsim(path_1D_50Groups, name, variable, time=1, ax=ax1, label='50 Energy Groups')
#     plt.legend()
#     plt.show()


# def compare_cyl2d_cart2d(variable):
#     ax1 = plot1d(path3, name, variable, time=1, slice=0, grid='cartesian', label='cartesian (0 µm)')
#     plot1d(path2, name, variable, time=1, slice=0, ax=ax1, grid='cylindrical', label='cylindrical (0 µm)')
#     plot1d(path3, name, variable, time=1, slice=5, ax=ax1, grid='cartesian', label='cartesian (5 µm)')
#     plot1d(path2, name, variable, time=1, slice=5, ax=ax1, grid='cylindrical', label='cylindrical (5 µm)')
#     plt.legend()
#     plt.savefig('D:/Simulation/testcase_results/cyl_vs_cart_'+variable+'_different_slices.png')