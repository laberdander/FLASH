import yt
from matplotlib import pyplot as plt
import numpy as np


class FlashPlot2D:
    """
    Class handles FLASH plot files from 2D simulations.
    """
    k_b = 8.617333262 * 1e-5

    # Dictionary that contains the information about the different plot variables
    var_dict = {
        'dens_label': 'material density ' + r'$\left(\frac{g}{cm^3}\right)$',
        'dens_scale': 1,
        'tele_label': 'electron temperature (eV))',
        'tele_scale': k_b,
        'nele_label': 'test',
        'nele_scale': 1,
        'nele_numdens': 'ye',
        'nion_label': 'test',
        'nion_scale': 1,
        'nion_numdens': 'sumy',
        'ye_scale': 1,
        'ye_label': 'test'
    }

    def __init__(self, path, time=None, scale=10):
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
        """

        self.data_path = path

        # import, if a single plotfile path is given
        if time is None:
            self.ds = yt.load(path)  # self.ds: full hdf5 dataset
            self.r_max = self.ds.domain_width.in_units('um')[0]  # self.r_max: grid width in r-direction in µm
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
            self.r_max = self.ds.domain_width.in_units('um')[0]
            self.x_max = self.ds.domain_width.in_units('um')[1]
            self.t = self.ds.current_time.in_units('ns')

        self.scale = scale

        # Create grid of position values with same dimensions as the sampling grid
        r = np.linspace(0, int(self.r_max), int(self.r_max * self.scale))
        x = np.linspace(0, int(self.x_max), int(self.x_max * self.scale))
        self.grid = np.meshgrid(r, x)

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
        ray_unsrtd = ds.ray([r_slice/float(self.r_max), 0, 0], [r_slice/float(self.r_max), 1, 0])
        return ray_unsrtd

    def data_2d(self, rmax=None, xmax=None):
        """
        Returns a grid with dimensions (xmax*self.scale, rmax*self.scale) that contains the 2D simulation results

        :param rmax: float; default: self.r_max, grid sample size, no need to change parameter
        :param xmax: float; default: self.x_max, grid sample size, no need to change parameter
        :return: yt.data_objects; grid with simulation data, still contains all plot variables
        """
        if rmax is None:
            rmax = self.r_max
        if xmax is None:
            xmax = self.x_max
        ds = self.ds
        slc = ds.slice(2, 0)
        frb = slc.to_frb((rmax, 'um'), (int(xmax*self.scale), int(rmax*self.scale)), height=(xmax, 'um'))
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
            ray = np.array(ray_unsrtd['flash', 'dens'][srt])\
                           * np.array(ray_unsrtd['flash', self.var_dict[variable+'_numdens']][srt])\
                           * self.var_dict[variable + '_scale']
            x = np.array(ray_unsrtd['index', 'z'][srt].in_units('um'))
        else:
            ray = np.array(ray_unsrtd['flash', variable][srt])*self.var_dict[variable+'_scale']
            x = np.array(ray_unsrtd['index', 'z'][srt].in_units('um'))
        return x, ray

    def data_numpy_2d(self, frb, variable):
        """
        Returns 2D numpy array of the given plot variable.

        :param frb: yt.data_objects; 2D grid produced by self.data_2d
        :param variable: string; variable indicates whether to get density data('dens'),
        electron temperature data ('tele'), or others
        :return: np.array; 2D numpy array that contains the simulation data of the given variable
        """
        if variable == 'nele' or variable == 'nion':
            data = np.array(frb['flash', 'dens'])\
                   * np.array(frb['flash', self.var_dict[variable+'_numdens']])\
                   * self.var_dict[variable + '_scale']
        else:
            data = np.array(frb['flash', variable])*self.var_dict[variable+'_scale']
        return data

    def plot_1d(self, ray_unsrtd, variable, ax, xmin=0, xmax=300, **kwargs):
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
        ax.set_xlim(xmin, xmax)
        ax.set_xlabel('length (µm)')
        ax.set_ylabel(self.var_dict[variable+'_label'])
        return ax

    def plot_2d(self, frb, variable, ax, rmin=0, rmax=200, xmin=0, xmax=300, **kwargs):
        """
        Plots 2D colorplot of given plot variable to the given axes
        :param frb: yt.data_objects; 2D grid produced by self.data_2d
        :param variable: string; variable indicates whether to get density data('dens'),
        electron temperature data ('tele'), or others
        :param ax: ax: matplotlib.axes.Axes; axes where the 2d graph is plotted to
        :param rmin: float; Default: 0; Minimum r value (in µm)
        :param rmax: float; Default: 200; Maximum r value (in µm)
        :param xmin: float; Default: 0; Minimum x value (in µm)
        :param xmax: float; Default: 300; Maximum x value (in µm)
        :param kwargs: are given to ax.pcolormesh(,**kwargs) (for example norm=matplotlib.colors.LogNorm())
        :return: matplotlib.axes.Axes
        """
        data = self.data_numpy_2d(frb, variable)
        cplot = ax.pcolormesh(*self.grid, data, **kwargs)
        cbar = plt.colorbar(cplot)
        cbar.set_label(self.var_dict[variable+'_label'])
        ax.set_xlim(rmin, rmax)
        ax.set_ylim(xmin, xmax)
        ax.set_xlabel('length (µm)')
        ax.set_ylabel('length (µm)')
        return ax

    @staticmethod
    def save_plot(figure, save_path):
        """
        Saves figure to save_path

        :param figure: matplotlib.figure;
        :param save_path: string; Has to include either .pdf or .png or .jpg as ending
        :return: Nothing
        """
        figure.savefig(save_path)
        figure.axes.cla()


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
        ray_unsrtd = ds.ray([0, 0, 0], [1, 0, 0])
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
            ray = np.array(ray_unsrtd['flash', 'dens'][srt])\
                           * np.array(ray_unsrtd['flash', self.var_dict[variable+'_numdens']][srt])\
                           * self.var_dict[variable + '_scale']
            x = np.array(ray_unsrtd['index', 'x'][srt].in_units('um'))
        else:
            ray = np.array(ray_unsrtd['flash', variable][srt])*self.var_dict[variable+'_scale']
            x = np.array(ray_unsrtd['index', 'x'][srt].in_units('um'))
        return x, ray

    def plot_1d(self, ray_unsrtd, variable, ax, xmin=0, xmax=300, **kwargs):
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
        ax.set_xlim(xmin, xmax)
        ax.set_xlabel('length (µm)')
        ax.set_ylabel(self.var_dict[variable+'_label'])
        return ax
