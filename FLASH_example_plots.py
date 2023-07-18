import sys
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
# import pandas as pd
import scipy.constants as const
from scipy import interpolate
from scipy import optimize
import tables


sys.path.insert(0, '/lustre/phx/lwegert/Data_Analysis')

import FLASH_PLOT


dict = {
    'dens_xmin': 40,
    'dens_xmax': 53,
    'dens_ymin': 0,
    'dens_ymax': 8,
    'dens_x_scale': 'linear',
    'dens_scale': 'linear',
    'tele_xmin': 35,
    'tele_xmax': 310,
    'tele_ymin': 0,
    'tele_ymax': 500,
    'tele_scale': 'linear',
    'tele_x_scale': 'log',
    'tion_xmin': 35,
    'tion_xmax': 310,
    'tion_ymin': 0,
    'tion_ymax': 500,
    'tion_scale': 'linear',
    'tion_x_scale': 'log',
    'nele_xmin': 35,
    'nele_xmax': 310,
    'nele_ymin': 1e16,
    'nele_ymax': 1e24,
    'nele_scale': 'log',
    'nele_x_scale': 'log',
    'nion_xmin': 40,
    'nion_xmax': 52,
    'nion_ymin': 0,
    'nion_ymax': 12e22,
    'nion_scale': 'linear',
    'nion_x_scale': 'linear',
    'ye_xmin': -20,
    'ye_xmax': 100,
    'ye_ymin': 0,
    'ye_ymax': 100,
    'depo_xmin': -20,
    'depo_xmax': 400,
    'depo_ymin': 0,
    'depo_ymax': 1e12,
    'depo_scale': 'linear',
    'depo_x_scale': 'lin',
    'pres_xmin': -20,
    'pres_xmax': 400,
    'pres_scale': 'linear',
    'pres_x_scale': 'linear',
    'pres_ymin': 0,
    'pres_ymax': 3,
    'zavg_xmin': 35,
    'zavg_xmax': 310,
    'zavg_ymin': 0,
    'zavg_ymax': 18,
    'zavg_scale': 'linear',
    'zavg_x_scale': 'log'
}


def plot2d(variable, plotter2d, frb, path, **kwargs):
    frb = frb
    plotter2d = plotter2d
    fig, ax = plt.subplots()
    plotter2d.plot_2d(frb, variable, ax, **kwargs)
    return ax


def load_flash(path, file, time=None, slice=0, grid='cartesian'):
    if time is None:
        plotter2d = FLASH_PLOT.FlashPlot2D(path+file, scale=1, grid=grid)
    else:
        plotter2d = FLASH_PLOT.FlashPlot2D(path+file, scale=1, time=time, grid=grid)
    ray = plotter2d.data_1d(r_slice=slice)
    return plotter2d, ray


def load_flash_2d(path, file, time=None, slice=0, grid='cartesian'):
    if time is None:
        plotter2d = FLASH_PLOT.FlashPlot2D(path+file, scale=1, grid=grid)
    else:
        plotter2d = FLASH_PLOT.FlashPlot2D(path+file, scale=1, time=time, grid=grid, n_x = 100, n_r = 5)
    frb = plotter2d.data_2d()
    return plotter2d, frb



def plot1d(variable, plotter2d, ray, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        pass
    ray = ray
    plotter2d = plotter2d
    plotter2d.plot_1d(ray, variable, ax, **kwargs)
    x, np_ray = plotter2d.data_numpy_1d(ray, variable)
    # print(variable + ' max:   ' + np.max(np_ray))
    # print(variable+' maximum x value:   ' + x[np.argmax(np_ray)])
    ax.set_xlim(dict[variable+'_xmin'], dict[variable+'_xmax'])
    ax.set_ylim(dict[variable+'_ymin'], dict[variable+'_ymax'])
    ax.set_xscale(dict[variable + '_x_scale'])
    ax.set_yscale(dict[variable+'_scale'])
    print(plotter2d.t)
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
    ax.set_xscale(dict[variable + '_x_scale'])
    ax.set_yscale(dict[variable + '_scale'])
    return ax


def import_helios(path, file, add_x_offset=-20):
    data = np.load(path+file, allow_pickle=True)
    x = data[:, 0] * 1e4 + add_x_offset
    y = data[:, 1]
    return x, y


def plot_helios(path, variable, time, ax, **kwargs):
    x, y = import_helios(path, variable+'_'+str(time)+'ns.npy')
    ax.plot(x, y, **kwargs)
    ax.set_xlim(dict[variable + '_xmin'], dict[variable + '_xmax'])
    ax.set_ylim(dict[variable + '_ymin'], dict[variable + '_ymax'])
    ax.set_yscale(dict[variable + '_scale'])


def find_critical_density(plotter2d, ray, wavelength=5.27e-7):
    x, nele = plotter2d.data_numpy_1d(ray, 'nele')
    ang_freq = const.c / wavelength * 2*const.pi
    crit_dens = const.epsilon_0 * const.electron_mass / const.elementary_charge**2 * ang_freq**2 * 1e-6
    f = interpolate.interp1d(x, nele - crit_dens)
    sol = optimize.root_scalar(f, bracket=[50, 150])
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
    x_dens = np.linspace(x[0], x[-1], 1000)
    ind_shock_pos = np.argmax(spl.derivative()(x_dens))
    x_shock_pos = x_dens[ind_shock_pos]
    dens_shock_pos = spl(x_shock_pos)
    print('Position of shock wave: ' + str(x_shock_pos) + ' um    (density: ' + str(dens_shock_pos) + ' g/cc)')
    print('Maximum density: ' + str(dens_max) + ' g/cc)')
    return x_shock_pos, dens_shock_pos, x_dens_max, dens_max

def find_max_pressure(plotter2d, ray):
    x, pres = plotter2d.data_numpy_1d(ray, 'pres')
    ind_pres_max = np.argmax(pres)
    pres_max = pres[ind_pres_max]
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

def analyse_crit_dens(plotter2d, ray, ax, wavelength=5.27e-7, plot=True):
    x_crit_dens, crit_dens = find_critical_density(plotter2d, ray, wavelength=wavelength)
    x_max_pres, max_pres = find_max_pressure(plotter2d, ray)
    if plot:
        ax.hlines(crit_dens, dict['nele_xmin'], x_crit_dens, linestyles='dashed')
        ax.vlines(x_crit_dens, dict['nele_ymin'], crit_dens, linestyles='dashed')
        ax.annotate(r'$z_{crit} = $' + str(round(x_crit_dens, 1)) + r'$ \mu m$',
                    xy=(x_crit_dens, crit_dens),
                    xytext=(x_crit_dens + 10, dict['nele_ymin'] + dict['nele_ymin']))


def analyse_max_temp(plotter2d, ray, ax, plot=True):
    x_max_temp, max_temp = find_max_temp(plotter2d, ray)
    if plot:
        ax.hlines(max_temp, dict['tele_xmin'], x_max_temp, linestyles='dashed')
        ax.vlines(x_max_temp, dict['tele_ymin'], max_temp, linestyles='dashed')
        ax.annotate(r'$T_{max} = $' + str(round(max_temp, 0)),
                    xy=(x_max_temp, max_temp),
                    xytext=(x_max_temp - 10, max_temp + 20))


def dataset_analysis(path_sim_data, path_results, label='cartesian (2D)', t=1, g='cartesian'):
    plotter2d, ray_1ns = load_flash(path_sim_data, name, time=t, slice=2, grid=g)
    ax_dens = plot1d('dens', plotter2d, ray_1ns, label=label)
    analyse_shock_pos(plotter2d, ray_1ns, ax_dens)
    plt.savefig(path_results+str(t)+'ns_dens_1D.png')
    ax_nele = plot1d('nele', plotter2d, ray_1ns, label=label)
    analyse_crit_dens(plotter2d, ray_1ns, ax_nele, wavelength=8e-7)
    plt.savefig(path_results+str(t)+'ns_nele_1D.png')
    ax_tele = plot1d('tele', plotter2d, ray_1ns, label=label)
    plot1d('tion', plotter2d, ray_1ns, label=label, ax=ax_tele, linestyle='dashed')
    analyse_max_temp(plotter2d, ray_1ns, ax_tele)
    plt.savefig(path_results+str(t)+'ns_tele_1D.png')
    ax_nion = plot1d('nion', plotter2d, ray_1ns, label=label)
    plt.savefig(path_results+str(t)+'ns_nion_1D.png')
    ax_zavg = plot1d('zavg', plotter2d, ray_1ns, label=label)
    plt.savefig(path_results+str(t)+'ns_zavg_1D.png')



def ray_data(file):
    h5file = tables.open_file(file, "r")
    data = h5file.root.RayData[:, :]
    nrow = data.shape[0]
    h5file.close()

    tags = data[:, 0]
    indx = tags.argsort(kind='mergesort')

    sorted_data = np.empty((nrow, 5))
    for i in range(len(indx)):
        sorted_data[i, :] = data[indx[i], :]
    # print(sorted_data)
    print(sorted_data.shape)
    ray_number = 0
    i_prev = 32.0
    loc_counter = 0
    loc_counter_array = []
    for i in sorted_data[:, 0]:
        if i != i_prev:
            loc_counter_array.append(loc_counter)
            # print('i            ' + str(i))
            # print('i_prev       ' + str(i_prev))
            ray_number += 1
            i_prev = i
            # print('loc_counter  ' + str(loc_counter))
            loc_counter = 0
        loc_counter += 1
    print(len(loc_counter_array))
    print(loc_counter_array)
    final_data = np.empty((ray_number, max(loc_counter_array), 5))

    f = 0
    for i in range(ray_number):
        for j in range(loc_counter_array[i]):
            final_data[i, j, :] = sorted_data[f+j, :]
        f += loc_counter_array[i]
    print(final_data.shape)
    return final_data, ray_number, loc_counter_array


def plot_rays(file):
    final_data, ray_number, loc_counter_array = ray_data(file)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for i in range(ray_number):
        ax.scatter(final_data[i, :, 1], final_data[i, :, 2], final_data[i, :, 4])
    plt.show()


def find_central_ray(data, number):
    final_data, ray_number = data, number

    x_list = np.empty(ray_number)
    i_list = np.empty(ray_number)
    for i in range(ray_number):
        x_list[i] = final_data[i, 2, 1]
        i_list[i] = i
    filter_list = x_list[x_list != 0.0]
    ind_list = i_list[x_list != 0.0]
    x_0_ind = np.argmin(np.absolute(filter_list))
    return int(ind_list[x_0_ind])


def plot_central_ray(file):
    final_data, ray_number, loc_counter_array = ray_data(file)
    ind = find_central_ray(final_data, ray_number)
    fig = plt.figure()
    ax = fig.add_subplot()
    x = final_data[ind, :, 2]*10000
    ray_power = final_data[ind, :, 4]
    ax.plot(x, ray_power/max(ray_power))
    return fig, ax


def plot_crit_dens(path_sim_data, path_results, label='cylindrical (2D)', g='cylindrical'):
    time = []
    cd = []
    for t in np.arange(0.1, 1.5, 0.1):
        plotter2d, ray = load_flash(path_sim_data, name, time=t, slice=2, grid=g)
        x_crit_dens, crit_dens = find_critical_density(plotter2d, ray, wavelength=8.0e-7)
        time.append(plotter2d.t)
        cd.append(x_crit_dens)
    plt.plot(time, cd)
    plt.xlabel('time (ns)')
    plt.ylabel('z_crit (' + r'$ \mu m$' + ')')
    plt.savefig(path_results + 'crit_dens.png')


def load_and_plot_2d(path_data, path_result, time, variable, grid = 'cylindrical'):
    plotter2d, frb = load_flash_2d(path_data, name, time=time, grid=grid)
    plot2d(variable, plotter2d, frb, path_result)
    plt.savefig(path_result+variable+'_2d_'+str(round(time, 1))+'ns.png')



path_2D_cartesian_6groups = '/lustre/phx/lwegert/WorkDirectory/testcase_2d_cartesian_ref7/'
path_2D_cartesian_50groups = '/lustre/phx/lwegert/WorkDirectory/testcase_2d_cartesian_50energygroups/'
path_2D_cylindrical_6groups = '/lustre/phx/lwegert/WorkDirectory/testcase_2d_cylindrical/'
path_2D_cylindrical_50groups = '/lustre/phx/lwegert/WorkDirectory/testcase_2d_cylindrical_50energygroups/'
name = 'lasslab_hdf5_plt_cnt_????'

path_helios_results = '/u/lwegert/WorkDirectory/Data/HELIOS/testcase/'

path_testcase_results = '/u/lwegert/WorkDirectory/Data_Analysis/'


path_testcase_3720_prop_6800 = '/lustre/phx/lwegert/WorkDirectory/testcase_2d_cartesian_3720_Prop_6800/'
path_testcase_3720_prop_6800_results = '/u/lwegert/WorkDirectory/Data_Analysis/testcase_2d_cartesian_3720_Prop_6800/'

path_testcase_3720_flash_6800 = '/lustre/phx/lwegert/WorkDirectory/testcase_2d_cartesian_3720_FLASH_6800/'
path_testcase_3720_flash_6800_results = '/u/lwegert/WorkDirectory/Data_Analysis/testcase_2d_cartesian_3720_FLASH_6800/'

path_testcase_flash_prop_6800 = '/lustre/phx/lwegert/WorkDirectory/testcase_2d_cartesian_FLASH_Prop_6800/'
path_testcase_flash_prop_6800_results = '/u/lwegert/WorkDirectory/Data_Analysis/testcase_2d_cartesian_FLASH_Prop_6800/'

path_testcase2_3720_prop_6800 = '/lustre/phx/lwegert/WorkDirectory/testcase2_2d_cylindrical_3720_Prop_6800/'
path_testcase2_3720_prop_6800_results = '/u/lwegert/WorkDirectory/Data_Analysis/testcase2_2d_cylindrical_3720_Prop_6800/'

path_testcase2_3720_prop25_6800 = '/lustre/phx/lwegert/WorkDirectory/testcase2_2d_cylindrical_3720_Prop25_6800/'
path_testcase2_3720_prop25_6800_results = '/u/lwegert/WorkDirectory/Data_Analysis/testcase2_2d_cylindrical_3720_Prop25_6800/'

path_testcase3_3720_prop_6800 = '/lustre/phx/lwegert/WorkDirectory/testcase3_2d_cylindrical_3720_Prop_6800/'
path_testcase3_3720_prop_6800_results = '/u/lwegert/WorkDirectory/Data_Analysis/testcase3_2d_cylindrical_3720_Prop_6800/'

path_testcase3_3720_prop25_6800 = '/lustre/phx/lwegert/WorkDirectory/testcase3_2d_cylindrical_3720_Prop25_6800/'
path_testcase3_3720_prop25_6800_results = '/u/lwegert/WorkDirectory/Data_Analysis/testcase3_2d_cylindrical_3720_Prop25_6800/'

path_LULI = '/lustre/phx/lwegert/WorkDirectory/LULI_Simulation/'
path_LULI_results = '/u/lwegert/WorkDirectory/Data_Analysis/LULI/'

path_foamtube_abl = 'D:/Analysis/FLASH_Sim/PHELIX_foamtube500_25umablator/'
path_test_cart = 'D:/Simulation/FLASH/2D/testcase/testcase_2d_cartesian_3720_Prop_6800/'

load_and_plot_2d(path_test_cart, path_test_cart, 5, 'dens', grid='cartesian')


# dataset_analysis(path_testcase2_3720_prop25_6800, path_testcase2_3720_prop25_6800_results, t=0.2, g='cylindrical')
# dataset_analysis(path_testcase3_3720_prop_6800, path_testcase3_3720_prop_6800_results, t=0.6, g='cylindrical')

# dataset_analysis(path_testcase3_3720_prop25_6800, path_testcase3_3720_prop25_6800_results, t=0.6, g='cylindrical')

# plot_crit_dens(path_testcase2_3720_prop_6800, path_testcase2_3720_prop_6800_results, g='cylindrical')

# plot2d(path_2D_cartesian_50groups, name, 'tele', time=1)
# plt.savefig(path_testcase_results+'tele_cart_50_2d.png')
