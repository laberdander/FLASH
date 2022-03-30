import sys
import matplotlib.pyplot as plt

# sys.path.insert(0, '/lustre/phx/lwegert/Data_Analysis')

import FLASH_PLOT


path1 = '/lustre/phx/lwegert/WorkDirectory/2D_Test_Ref7/lasslab_hdf5_plt_cnt_????'
path2 = 'D:/Simulation/FLASH/2D/1TW/lasslab_hdf5_plt_cnt_????'


def plot2d(path, variable, time=None):
    fig, ax = plt.subplots()
    plotter2d = FLASH_PLOT.FlashPlot2D(path, time=time, scale=1)
    frb = plotter2d.data_2d()
    plotter2d.plot_2d(frb, 'tele',ax)
    plotter2d.save_plot(fig, variable+'_2d_'+str(round(time, 1))+'ns.png')


def plot1d(path, variable, time=None):
    fig, ax = plt.subplots()
    plotter2d = FLASH_PLOT.FlashPlot2D(path, time=time, scale=1)
    ray = plotter2d.data_1d()
    plotter2d.plot_1d(ray, 'tele', ax)
    plotter2d.save_plot(fig, variable + '_1d_' + str(round(time, 1)) + 'ns.png')


plot1d(path2, 'tele', time=1)


