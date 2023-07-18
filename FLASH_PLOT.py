import yt
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import sys
import scipy.constants as const
from scipy import interpolate
from scipy import optimize

# import scipy as sc

yt.set_log_level('critical')


# matplotlib.use('Agg')


class FlashPlot2D:
    """
    Class handles FLASH plot files from 2D simulations.
    """
    k_b = 8.617333262 * 1e-5
    n_a = 6.02214076 * 1e23

    # Dictionary that contains the information about the different plot variables
    var_dict = {
        'dens_label': 'material density ' + r'$\left(\frac{g}{cm^3}\right)$',
        'dens_scale': 1,
        'tele_label': 'electron temperature (eV)',
        'tele_scale': k_b,
        'tion_label': 'electron temperature (eV)',
        'tion_scale': k_b,
        'nele_label': 'electron density ' + r'$\left(\frac{1}{cm^3}\right)$',
        'nele_scale': n_a,
        'nele_numdens': 'ye',
        'nion_label': 'test',
        'nion_scale': n_a,
        'nion_numdens': 'sumy',
        'ye_scale': 1,
        'ye_label': 'test',
        'zavg_label': 'average ionization level',
        'zavg_scale': 1,
        'depo_label': 'deposited laser energy' + r'$\left(\frac{erg}{g}\right)$',
        'depo_scale': 1,
        'pres_label': 'pressure (Mbar)',
        'pres_scale': 1e-12,
        'targ_label': 'target contribution',
        'targ_scale': 1,
        'cartesian_slice': 'y',
        'cylindrical_slice': 'z',
    }

    def __init__(self, path, time=None, scale=10, n_x=1, n_r=1, grid='cartesian'):
        """
        Init command loads a single datafile and loads some information of this simulation into class variables.

        :param path: string; path of FLASH plot file, example: .../lasslab_hdf5_plt_cnt_0001,
        also possible: load all timesteps of a simulation via : .../lasslab_hdf5_plt_cnt_????,
        then specify via time parameter, what time step to plot
        :param time: float; default: None, If all timesteps are loaded,
        time (in ns) specifies the plotfile that is loaded
        :param scale: float; default: 10,
        scale*x_width(in µm) and scale * y_width(in µm) specifies the number of sampling
        points in x and y direction, this is only used for 2D plot data
        :param n_x: int; default: 1 (will not be used if n_x = 1)
        Number of sampling points in x direction, if not spefified, scale will be used
        :param n_r: int; default: 1 (will not be used if n_r = 1)
        Number of sampling points in r direction, if not spefified, scale will be used
        :param grid: string; default: 'cartesian', data import different for different grid structures
        """

        self.data_path = path

        # import, if a single plotfile path is given
        if time is None:
            self.ds = yt.load(path)  # self.ds: full hdf5 dataset
            self.r_min = self.ds.domain_left_edge.in_units('um')[0]
            self.r_max = self.ds.domain_width.in_units('um')[0]  # self.r_max: grid width in r-direction in µm
            self.x_min = self.ds.domain_right_edge.in_units('um')[1]
            self.x_max = self.ds.domain_width.in_units('um')[1]  # self.x_max: grid width in x-direction in µm
            self.t = self.ds.current_time.in_units('ns')  # self.t: Current time step of loaded plotfile in ns

        # import, if many plotfiles are given, time parameter required
        else:
            # load time series and find the file that is closest to the given parameter time
            self.ts = yt.load(path)
            times = []
            for ds in self.ts:
                times.append(ds.current_time.in_units('ns'))
            ts_ind = self.find_nearest(np.array(times), time)
            # load identical class parameters as above
            self.ds = self.ts[ts_ind]
            self.r_min = self.ds.domain_left_edge.in_units('um')[0]
            self.r_max = self.ds.domain_right_edge.in_units('um')[0]  # self.r_max: grid width in r-direction in µm
            self.x_min = self.ds.domain_left_edge.in_units('um')[1]
            self.x_max = self.ds.domain_right_edge.in_units('um')[1]
            self.t = self.ds.current_time.in_units('ns')

        self.scale = scale
        self.r_width = self.r_max - self.r_min
        self.x_width = self.x_max - self.x_min
        # Create grid of position values with same dimensions as the sampling grid
        self.grid_style = grid
        print(self.grid_style)

        # Change number of sample points in cylindrical case (needs to be odd)
        if grid == 'cylindrical' and n_r%2 == 0:
            self.n_r = n_r +1
        else:
            self.n_r = n_r
        self.n_x = n_x

        # Use number of sampling points defined by scale, when n_r (and n_x) is not specified
        if self.n_r == 1:
            r = np.linspace(int(self.r_min), int(self.r_max), int(self.r_width * self.scale))
            x = np.linspace(int(self.x_min), int(self.x_max), int(self.x_width * self.scale))
            if grid == 'cartesian':
                self.grid = np.meshgrid(r, x)
            elif grid == 'cylindrical':
                r = np.linspace(-int(self.r_max), int(self.r_max), 2 * int(self.r_width * self.scale) - 1)
                self.grid = np.meshgrid(r, x)
            else:
                sys.exit('No valid grid style')
        else:
            r = np.linspace(int(self.r_min), int(self.r_max), self.n_r)
            x = np.linspace(int(self.x_min), int(self.x_max), self.n_x)
            if grid == 'cartesian':
                self.grid = np.meshgrid(r, x)
            elif grid == 'cylindrical':
                r = np.linspace(-int(self.r_max), int(self.r_max), self.n_r)
                self.grid = np.meshgrid(r, x)
            else:
                sys.exit('No valid grid style')

    @staticmethod
    def find_nearest(array, value):
        """
        Function finds array element closest to the given value and returns the index of this element.
        Used, to find the plot file to a given time.

        :param array: np.array; array, that is compared with value
        :param value: float; function finds element in array, that is closest to value
        :return: int; index of element in array that is closest to value
        """
        idx = (np.abs(array - value)).argmin()
        return idx

    def data_1d(self, r_slice=0):
        """
        Takes a one dimensional slice parallel to x-axis at r=r_slice
        Resolution corresponds to simulation resolution.

        :param r_slice: float; Default: 0; r value in µm
        :return: yt.data_objects; slice ray, still contains all plot variables
        """
        ds = self.ds
        ray_unsrtd = ds.ray([r_slice * 1e-4, -1, 0], [r_slice * 1e-4, 1, 0])
        return ray_unsrtd

    def data_2d(self):
        """
        Returns a grid with dimensions (xmax*self.scale, rmax*self.scale) that contains the 2D simulation results

        :return: yt.data_objects; grid with simulation data, still contains all plot variables
        """
        
        rwidth = self.r_width
        xwidth = self.x_width

        ds = self.ds
        slc = ds.slice(2, 0)
        if self.n_r == 1:    
            if self.grid_style == 'cartesian':
                frb = slc.to_frb((rwidth, 'um'), (int(rwidth * self.scale), int(xwidth * self.scale)),
                                 height=(xwidth, 'um'))
            else:
                frb = slc.to_frb((rwidth, 'um'), (int(xwidth * self.scale), int(rwidth * self.scale)),
                                 height=(xwidth, 'um'))
        else:
            if self.grid_style == 'cartesian':
                frb = slc.to_frb((rwidth, 'um'), (int(self.n_r), int(self.n_x)),
                                 height=(xwidth, 'um'))
            else:
                frb = slc.to_frb((rwidth, 'um'), (int(self.n_x), int((self.n_r+1)/2)),
                                 height=(xwidth, 'um'))
        return frb

    def data_numpy_1d(self, ray_unsrtd, variable):
        """
        Returns 1D numpy array of the given plot variable.

        :param ray_unsrtd: yt.data_objects; 1D ray produced by self.data_1d
        :param variable: string; variable indicates whether to get density data('dens'),
        electron temperature data ('tele'), or others
        :return: x, ray (both np.array); returns 1D numpy array of the given data and the corresponding x values
        """
        srt = np.argsort(ray_unsrtd['t'])
        if variable == 'nele' or variable == 'nion':
            ray = np.array(ray_unsrtd['flash', 'dens'][srt]) \
                  * np.array(ray_unsrtd['flash', self.var_dict[variable + '_numdens']][srt]) \
                  * self.var_dict[variable + '_scale']
            x = np.array(ray_unsrtd['index', self.var_dict[self.grid_style + '_slice']][srt].in_units('um'))
        elif variable == 'zavg':
            ray = np.array(ray_unsrtd['flash', 'ye'][srt]) \
                  / np.array(ray_unsrtd['flash', 'sumy'][srt]) \
                  * self.var_dict[variable + '_scale']
            x = np.array(ray_unsrtd['index', self.var_dict[self.grid_style + '_slice']][srt].in_units('um'))
        else:
            ray = np.array(ray_unsrtd['flash', variable][srt]) * self.var_dict[variable + '_scale']
            x = np.array(ray_unsrtd['index', self.var_dict[self.grid_style + '_slice']][srt].in_units('um'))
        return x, ray

    
    def data_numpy_2d_save(self, frb, variable):
        """
        Returns 2D numpy array of the given plot variable.

        :param frb: yt.data_objects; 2D grid produced by self.data_2d
        :param variable: string; variable indicates whether to get density data('dens'),
        electron temperature data ('tele'), or others
        :return: np.array; 2D numpy array that contains the simulation data of the given variable
        """
        if variable == 'nele' or variable == 'nion':
            data = np.array(frb['flash', 'dens']) \
                   * np.array(frb['flash', self.var_dict[variable + '_numdens']]) \
                   * self.var_dict[variable + '_scale']
        elif variable == 'zavg':
            data = np.array(frb['flash', 'ye']) \
                   / np.array(frb['flash', 'sumy']) \
                   * self.var_dict[variable + '_scale']
        else:
            data = np.array(frb['flash', variable]) * self.var_dict[variable + '_scale']

        return data


    def data_numpy_2d(self, frb, variable):
        """
        Returns 2D numpy array of the given plot variable.

        :param frb: yt.data_objects; 2D grid produced by self.data_2d
        :param variable: string; variable indicates whether to get density data('dens'),
        electron temperature data ('tele'), or others
        :return: np.array; 2D numpy array that contains the simulation data of the given variable
        """
        if variable == 'nele' or variable == 'nion':
            data = np.array(frb['flash', 'dens']) \
                   * np.array(frb['flash', self.var_dict[variable + '_numdens']]) \
                   * self.var_dict[variable + '_scale']
        elif variable == 'zavg':
            data = np.array(frb['flash', 'ye']) \
                   / np.array(frb['flash', 'sumy']) \
                   * self.var_dict[variable + '_scale']
        else:
            data = np.array(frb['flash', variable]) * self.var_dict[variable + '_scale']

        if self.grid_style == 'cartesian':
            return data
        else:
            data_mirrored = []
            for i in data:
                data_mirrored.append(np.append(i[::-1], i[1:]))
            data = np.array(data_mirrored)
            # print('Numpy shape:  ' + str(np.shape(data)))
            return data

    def plot_1d(self, ray_unsrtd, variable, ax, **kwargs):
        """
        Plots 1D graph of given plot variable to the given axes.

        :param ray_unsrtd: t.data_objects; 1D ray produced by self.data_1d
        :param variable: string; variable indicates whether to get density data('dens'),
        electron temperature data ('tele'), or others
        :param ax: matplotlib.axes.Axes; axes where the 1d graph is plotted to
        :param kwargs: are given to ax.plot(,**kwargs)
        :return: matplotlib.axes.Axes
        """
        x, ray = self.data_numpy_1d(ray_unsrtd, variable)
        ax.plot(x, ray, **kwargs)
        ax.set_xlabel('length (µm)')
        ax.set_ylabel(self.var_dict[variable + '_label'])
        return ax

    def plot_2d(self, frb, variable, ax, **kwargs):
        """
        Plots 2D colorplot of given plot variable to the given axes
        :param frb: yt.data_objects; 2D grid produced by self.data_2d
        :param variable: string; variable indicates whether to get density data('dens'),
        electron temperature data ('tele'), or others
        :param ax: ax: matplotlib.axes.Axes; axes where the 2d graph is plotted to
        :param kwargs: are given to ax.pcolormesh(,**kwargs) (for example norm=matplotlib.colors.LogNorm())
        :return: matplotlib.axes.Axes
        """
        data = self.data_numpy_2d(frb, variable)
        cplot = ax.pcolormesh(*self.grid, data, **kwargs)
        cbar = plt.colorbar(cplot)
        cbar.set_label(self.var_dict[variable+'_label'])
        ax.set_xlabel('length (µm)')
        ax.set_ylabel('length (µm)')
        return ax, cplot

    @staticmethod
    def save_plot(figure, save_path):
        """
        Saves figure to save_path

        :param figure: matplotlib.figure;
        :param save_path: string; Has to include either .pdf or .png or .jpg as ending
        :return: Nothing
        """
        figure.savefig(save_path)


class FlashPlot1D:
    """
    Class handles FLASH plot files from 1D simulations
    """
    var_dict = FlashPlot2D.var_dict

    def __init__(self, path, time=None):
        """
        Init command loads either a single or a bunch of datafiles
        and loads some information of this simulation into class variables.

        :param path: string; path of FLASH plot file, example: .../lasslab_hdf5_plt_cnt_0001,
        also possible: load all timesteps of a simulation via : .../lasslab_hdf5_plt_cnt_????,
        then specify via time parameter, what time step to plot
        :param time: float; default: None, If all timesteps are given (via ????),
        time (in ns) specifies the plotfile that is loaded
        """

        self.data_path = path

        # import, if a single plotfile path is given
        if time is None:
            self.ds = yt.load(path)  # self.ds: full hdf5 dataset
            self.t = self.ds.current_time.in_units('ns')  # self.t: Current time step of loaded plotfile in ns

        # import, if many plotfiles are given, time parameter required
        else:
            # load time series and find the file that is closest to the given parameter time
            self.ts = yt.load(path)
            times = []
            for ds in self.ts:
                times.append(ds.current_time.in_units('ns'))
            ts_ind = FlashPlot2D.find_nearest(np.array(times), time)
            # same class variables as above
            self.ds = self.ts[ts_ind]
            self.t = self.ds.current_time.in_units('ns')

    def data_1d(self):
        """
        Processes the 1D data of the hdf5 and returns 1D ray, that contains all simulation data

        :return: yt.data_objects; slice ray, still contains all plot variables
        """
        ds = self.ds
        ray_unsrtd = ds.ray([-1, 0, 0], [1, 0, 0])
        return ray_unsrtd

    def data_numpy_1d(self, ray_unsrtd, variable):
        """
        Returns 1D numpy array of the given plot variable.

        :param ray_unsrtd: yt.data_objects; 1D ray produced by self.data_1d
        :param variable: string; variable indicates whether to get density data('dens'),
        electron temperature data ('tele'), or others
        :return: x, ray (both np.array); returns 1D numpy array of the given data and the corresponding x values
        """
        srt = np.argsort(ray_unsrtd['t'])
        if variable == 'nele' or variable == 'nion':
            ray = np.array(ray_unsrtd['flash', 'dens'][srt]) \
                  * np.array(ray_unsrtd['flash', self.var_dict[variable + '_numdens']][srt]) \
                  * self.var_dict[variable + '_scale']
            x = np.array(ray_unsrtd['index', 'x'][srt].in_units('um'))
        else:
            ray = np.array(ray_unsrtd['flash', variable][srt]) * self.var_dict[variable + '_scale']
            x = np.array(ray_unsrtd['index', 'x'][srt].in_units('um'))
        return x, ray

    def plot_1d(self, ray_unsrtd, variable, ax, **kwargs):
        """
        Plots 1D graph of given plot variable to the given axes.

        :param ray_unsrtd: t.data_objects; 1D ray produced by self.data_1d
        :param variable: string; variable indicates whether to get density data('dens'),
        electron temperature data ('tele'), or others
        :param ax: matplotlib.axes.Axes; axes where the 1d graph is plotted to
        :param xmin: float; Default: 0; Minimum x-value (in µm)
        :param xmax: float; Default: 300; Maximum x-value (in µm)
        :param kwargs: are given to ax.plot(,**kwargs)
        :return: matplotlib.axes.Axes
        """
        x, ray = self.data_numpy_1d(ray_unsrtd, variable)
        ax.plot(x, ray, **kwargs)
        ax.set_xlabel('length (µm)')
        ax.set_ylabel(self.var_dict[variable + '_label'])
        return ax

class FLASHPLOT2d_analyse:
    def __init__(self, plotter2d, ray, plot_dict):
        self.plotter2d = plotter2d
        self.ray = ray
        self.plot_dict = plot_dict


    def find_critical_density(self, wavelength=5.27e-7):
        x, nele = self.plotter2d.data_numpy_1d(self.ray, 'nele')
        ang_freq = const.c / wavelength * 2*const.pi
        crit_dens = const.epsilon_0 * const.electron_mass / const.elementary_charge**2 * ang_freq**2 * 1e-6
        f = interpolate.interp1d(x, nele - crit_dens)
        sol = optimize.root_scalar(f) # , bracket=[1900, 2100])
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

    def analyse_crit_dens(self, ax, wavelength=5.27e-7, plot=True):
        x_crit_dens, crit_dens = self.find_critical_density(wavelength=wavelength)
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
            ax.hlines(max_temp, self.plot_dict['tele_xmin'], x_max_temp, linestyles='dashed')
            ax.vlines(x_max_temp, self.plot_dict['tele_ymin'], max_temp, linestyles='dashed')
            ax.annotate(r'$T_{max} = $' + str(round(max_temp, 0)),
                        xy=(x_max_temp, max_temp),
                        xytext=(x_max_temp - 10, max_temp + 20))
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
