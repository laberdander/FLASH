# FLASH

FLASH_PLOT contains all the relevant functions to get FLASH Data and to create plots.

FLASH_example_plots.py shows how to possibly use FLASH_PLOT.

FLASH_data_analysis.py is currently just used to play around.


Here a few notes:

An instance of FLASH_PLOT.FlashPlot2D or FLASH_PLOT.FlashPlot1D contains only a single hdf5 file. If many HDF5 files are provided in the
loading path ('.../lasslab_hdf5_plt_cnt_????'), a single file has to be specified with the time parameter.
An instance is created for example by:
    plotter2d = FLASH_PLOT.FlashPlot2D('lasslab_hdf5_plt_cnt_????', time=time, scale=1)

The yt data objects (in this case ray), that are for example generated by:
    ray = plotter2d.data_1d()
still contain all different quantities ('dens', 'tele', ...). Hence they have to be loaded only once, when wanting to plot more than one
quantity for a given time step.

The plotters do always plot to an axis that has to be provided. By that, it is possible to plot several information into a single figure.

FLASH_PLOT.FlashPlot2d.save_plot(figure, 'testplot.png') is only doing fig.savefig('testplot.png')
