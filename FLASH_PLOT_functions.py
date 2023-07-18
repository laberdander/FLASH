import sys
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
# import pandas as pd
import scipy.constants as const
from scipy import interpolate
from scipy import optimize


sys.path.insert(0, '/lustre/phx/lwegert/Data_Analysis')
plt.rcParams.update({'font.size': 20})
plt.rc('legend', fontsize=12)

import FLASH_PLOT

class FLASHPLOT2d_analyse:
    def __init__(self, plotter2d, ray, plot_dict):
        self.plotter2d = plotter2d
        self.ray = ray
        self.plot_dict = plot_dict


    def find_critical_density(self, bracket = [500, 1000],wavelength=5.27e-7):
        x, nele = self.plotter2d.data_numpy_1d(self.ray, 'nele')
        ang_freq = const.c / wavelength * 2*const.pi
        crit_dens = const.epsilon_0 * const.electron_mass / const.elementary_charge**2 * ang_freq**2 * 1e-6
        f = interpolate.interp1d(x, nele - crit_dens)
        sol = optimize.root_scalar(f, bracket=bracket)
        print('Critical Density (' + str(crit_dens) + ' 1/cc) at x = ' + str(sol.root) + ' um')
        return sol.root, crit_dens 


    def find_max_temp(self):
        x, tele = self.plotter2d.data_numpy_1d(self.ray, 'tele')
        ind_tele_max = np.argmax(tele)
        x_tele_max = x[ind_tele_max]
        tele_max = np.max(tele)
        print('Maximum electron temperature: ' + str(tele_max) + ' eV, at x: ' + str(x_tele_max) + ' um')
        return x_tele_max, tele_max


    def find_shock_wave_pos(self):
        x, dens = self.plotter2d.data_numpy_1d(self.ray, 'dens')
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


    def find_max_pressure(self):
        x, pres = self.plotter2d.data_numpy_1d(self.ray, 'pres')
        ind_pres_max = np.argmax(pres)
        pres_max = pres[ind_pres_max]
        x_pres_max = x[ind_pres_max]
        print('Maximum Pressure: ' + str(pres_max) + ' Mbar, at x: ' + str(x_pres_max) + ' um')
        return x_pres_max, pres_max


    def analyse_shock_pos(self, ax, plot=True):
        shock_pos, dens_shock_pos, x_dens_max, dens_max = self.find_shock_wave_pos()
        x_max_pres, max_pres = self.find_max_pressure()
        if plot:
            dens_xmin = self.plot_dict['dens_xmin']
            dens_xmax = self.plot_dict['dens_xmax']
            dens_ymin = self.plot_dict['dens_ymin']
            dens_ymax = self.plot_dict['dens_ymax']
            ax.hlines(dens_max, dens_xmin, x_dens_max, linestyles='dashed')
            ax.vlines(shock_pos, dens_ymin, dens_shock_pos, linestyles='dashed')
            ax.vlines(x_max_pres, dens_ymin, dens_ymax, linestyles='dashed', colors='red')
            ax.annotate('Shock position:  ' + str(round(shock_pos, 1)) + r'$ \mu m$',
                        xy=(shock_pos, dens_shock_pos),
                        xytext=(shock_pos+0.5, 0.3))
            ax.annotate(str(round(dens_max, 2)) + ' g/cc',
                        xy=(x_dens_max, dens_max),
                        xytext=(dens_xmin+1, dens_max + (dens_ymax-dens_ymin)*0.05))
            ax.annotate('Max Pressure:  ' + str(round(max_pres, 1)) + ' Mbar',
                        xy=(x_max_pres, dens_max),
                        xytext=(x_max_pres + (dens_xmax-dens_xmin)*0.05, dens_ymax-0.5))
        return ax

    def analyse_crit_dens(self, ax, bracket = [800, 1000],wavelength=5.27e-7, plot=True):
        x_crit_dens, crit_dens = self.find_critical_density(bracket = bracket, wavelength=wavelength)
        x_max_pres, max_pres = self.find_max_pressure()
        if plot:
            ax.hlines(crit_dens, self.plot_dict['nele_xmin'], x_crit_dens, linestyles='dashed')
            ax.vlines(x_crit_dens, self.plot_dict['nele_ymin'], crit_dens, linestyles='dashed')
            ax.annotate(r'$z_{crit} = $' + str(round(x_crit_dens, 1)) + r'$ \mu m$',
                        xy=(x_crit_dens, crit_dens),
                        xytext=(x_crit_dens + 10, self.plot_dict['nele_ymin'] + self.plot_dict['nele_ymin']))
        return ax


    def analyse_max_temp(self, ax, plot=True):
        x_max_temp, max_temp = self.find_max_temp()
        if plot:
            # ax.hlines(max_temp, self.plot_dict['tele_xmin'], x_max_temp, linestyles='dashed')
            # ax.vlines(x_max_temp, self.plot_dict['tele_ymin'], max_temp, linestyles='dashed')
            ax.annotate(r'$T_{max} = $' + str(round(max_temp, 0)) + ' eV',
                        xy=(x_max_temp, max_temp),
                        xytext=(x_max_temp - 10, max_temp + 100))
        return ax



class FLASHPLOT2d_raytrace:
    
    def __init__(self, filename):
        self.file = filename

    def ray_data(self):
        h5file = tables.open_file(self.file, "r")
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


    def plot_rays(self):
        final_data, ray_number, loc_counter_array = self.ray_data(self.file)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for i in range(ray_number):
            ax.scatter(final_data[i, :, 1], final_data[i, :, 2], final_data[i, :, 4])
        return fig, ax


    def find_central_ray(self, data, number):
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


    def plot_central_ray(self, ray_number):
        final_data, ray_number, loc_counter_array = self.ray_data()
        ind = self.find_central_ray(final_data, ray_number)
        fig = plt.figure()
        ax = fig.add_subplot()
        x = final_data[ind, :, 2]*10000
        ray_power = final_data[ind, :, 4]
        ax.plot(x, ray_power/max(ray_power))
        return fig, ax