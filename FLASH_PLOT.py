import yt
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import sys

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
        'cartesian_slice': 'y',
        'cylindrical_slice': 'z',
        'targ_scale': 1,
        'targ_label': 'target',
        'cham_scale': 1,
        'cham_label': 'chamber'
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
            self.r_max = self.ds.domain_right_edge.in_units('um')[0]  # self.r_max: grid width in r-direction in µm
            self.x_min = self.ds.domain_left_edge.in_units('um')[1]
            self.x_max = self.ds.domain_right_edge.in_units('um')[1]  # self.x_max: grid width in x-direction in µm
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
            self.grid = np.meshgrid(r, x)
            if grid == 'cylindrical':
                r_extended = np.linspace(-int(self.r_max), int(self.r_max), 2 * int(self.r_width * self.scale) - 1)
                self.grid_extended = np.meshgrid(r_extended, x)
        else:
            r = np.linspace(self.r_min, self.r_max, self.n_r)
            x = np.linspace(self.x_min, self.x_max, self.n_x)
            self.grid = np.meshgrid(r, x)
            if grid == 'cylindrical':
                r_extended = np.linspace(-int(self.r_max), int(self.r_max), self.n_r)
                self.grid_extended = np.meshgrid(r_extended, x)
            else:
                pass
            # sys.exit('No valid grid style')

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
                frb = slc.to_frb((rwidth, 'um'), (int(self.n_x), int(self.n_r)),
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

    def data_numpy_2d(self, frb, variable, extended=False):
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

        if extended:
            data_mirrored = []
            for i in data:
                data_mirrored.append(np.append(i[::-1], i[1:]))
            data = np.array(data_mirrored)
            # print('Numpy shape:  ' + str(np.shape(data)))
            return data
        else:
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

    def plot_2d(self, frb, variable, ax, extended = False, **kwargs):
        """
        Plots 2D colorplot of given plot variable to the given axes
        :param frb: yt.data_objects; 2D grid produced by self.data_2d
        :param variable: string; variable indicates whether to get density data('dens'),
        electron temperature data ('tele'), or others
        :param ax: ax: matplotlib.axes.Axes; axes where the 2d graph is plotted to
        :param kwargs: are given to ax.pcolormesh(,**kwargs) (for example norm=matplotlib.colors.LogNorm())
        :return: matplotlib.axes.Axes
        """
        data = self.data_numpy_2d(frb, variable, extended=extended)
        if extended:
            cplot = ax.pcolormesh(*self.grid_extended, data, **kwargs)
        else:
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
