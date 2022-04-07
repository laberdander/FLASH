import sys
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy as sc
import scipy.integrate as integrate


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
    'tele_ymax': 500,
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
name = 'lasslab_hdf5_plt_cnt_????'


def plot2d(path, file, variable, time=None, **kwargs):
    fig, ax = plt.subplots()
    plotter2d = FLASH_PLOT.FlashPlot2D(path+file, time=time, **kwargs)
    frb = plotter2d.data_2d()
    plotter2d.plot_2d(frb, variable, ax)
    plt.show()
    # plotter2d.save_plot(fig, path+variable+'_2d_'+str(round(time, 1))+'ns.png')


def plot1d(path, file, variable, time=None, slice=0, ax= None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        pass
    if time is None:
        plotter2d = FLASH_PLOT.FlashPlot2D(path+file, scale=1)
    else:
        plotter2d = FLASH_PLOT.FlashPlot2D(path+file, scale=1, time=time)
    ray = plotter2d.data_1d(r_slice=slice)
    plotter2d.plot_1d(ray, variable, ax)
    ax.set_xlim(dict[variable+'_xmin'], dict[variable+'_xmax'])
    ax.set_ylim(dict[variable+'_ymin'], dict[variable+'_ymax'])
    ax.set_yscale(dict[variable+'_scale'])
    # plotter2d.save_plot(fig, path + variable + '_1d_' + str(round(time, 1)) + 'ns.png')
    return ax


def gauss(y, y_0, sigma):
    return np.exp(-(y - y_0)**2 / (2*sigma**2))


def shock_wave_function(x, y_0, a, b, sigma, c):
    result = gauss(x, y_0, sigma)*abs(a)+b
    return np.heaviside(x-y_0, 0.5) * result + c


def shock_wave_function_test(x, y_0, a, b, sigma, c):
    if x > y_0 + 8*sigma:
        integral = integrate.quad(gauss, -np.inf, y_0 + 8 * sigma, args=(y_0, sigma,))[0]
        behind_shock = b - abs(a) * np.array(integral)
    else:
        integral = integrate.quad(gauss, -np.inf, x, args=(y_0, sigma,))[0]
        behind_shock = b - abs(a) * np.array(integral)
    return np.heaviside(x-y_0, 1)*behind_shock + c


def shock_wave_array(x, y_0, a, b, sigma, c):
    list = [shock_wave_function(x_0, y_0, a, b, sigma, c) for x_0 in x]
    return np.array(list)


def fit_shock_wave(xdata, ydata, guess, bounds=None):

    popt, pcov = sc.optimize.curve_fit(shock_wave_array, xdata, ydata, p0=guess, bounds=bounds)
    return popt


def doit(path, file, time, guess, **kwargs):
    plotter2d = FLASH_PLOT.FlashPlot2D(path + file, scale=1, time=time)
    ray = plotter2d.data_1d(r_slice=5)
    x, y = plotter2d.data_numpy_1d(ray, 'dens')
    popt = fit_shock_wave(x, y, guess, bounds=([-40, 3.7, -40, 0, 2.6], [50, 50, 50, 50, 2.75]))
    plt.plot(x, y, **kwargs)
    plt.plot(x, shock_wave_array(x, *popt))
    print('Time: ' + str(time) + '   Position: ' + str(popt[0]))
    print(popt)


