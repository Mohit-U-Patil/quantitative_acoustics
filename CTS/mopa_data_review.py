import os
import sys
sys.path.append("D:\ETH\Work\Latest Download\ETH\python")
import bsmd.algorithm.bsmdAlgorithm as bsmdAlgorithm
# import bsmd.algorithm.bsmdAlgorithmRestructured as bsmdAlgorithm
import bsmd.measurement.bsmdMeasurement as bsmdMeasurement
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
import pandas as pd
import logging
from itertools import cycle
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import CTS.mopa_2dgauss


def find_files(filter, topDir):
    '''searches in the topDir for the files with filter and returns list of all files as matches'''
    matches = []

    # check if it exists, otherwise quit
    if not os.path.exists(topDir):
        sys.exit("The directory {} was not found! Quitting...".format(topDir))

    for root, dirs, files in os.walk(topDir):
        # get all files satisfying the filter
        for file in glob.glob(os.path.join(root, filter)):
            matches.append(file)
    return matches

def raw_data_plot(meas, filelist, freq):
    '''reads the file, then you can window the data in both space and time directions, calculates fft, plots multiple options'''

    for file in filelist:
        logging.info("Started working on file {}".format(file))
        meas.readStepPhantomXMLData(file)
#         colNr = meas.data.shape[1]
#         rowNr = meas.data.shape[0]
#         window = np.hamming(36)
#         copy = meas.data.copy()
#         copy = np.transpose(copy)
# #        copy = copy * window
#         copy = np.transpose(copy)
#         meas.data = copy
#         meas.data = meas.movingTimeFilter(np.hamming(400), starttime=0.011, speed=480,
#                                         showImages=None)

        'meas.plot2D_all plots time-pos as well as freq-wavenumber data'
        #meas.plot2D_all("Raw measurement", clickEvent=True, showImages=True, colorbar=True)

        'Raw data plot'
        # meas.plot2D(meas.data, xLabel="Time (s)", yLabel="Pos (m)", title=os.path.basename(file) + "_raw", clickEvent=None,
        #         showImages=True, outFile = outdir + "\{}_raw.eps".format(os.path.basename(file)), colorbar=True, xRange=(0.01, 0.014), yRange=(-20, 20))

        # meas.plot2D(meas.data, xLabel="Time (s)", yLabel="Pos (m)", title=os.path.basename(file) + "_raw without windowing", clickEvent=None,
        #             showImages=None, outFile=outdir + "\{}_raw without windowing.png".format(os.path.basename(file)),
        #             colorbar=True, xRange=(0.01, 0.014), yRange=None)
        #
        # meas.plot2D(abs(meas.dataFFT), "Frequency (Hz)", "Wavenumber (1/m)", title=os.path.basename(file) + "_fft without windowing", clickEvent=None,
        #             showImages=None, colorbar=True , xRange=(2000, 4000), yRange=(0, 20), outFile = outdir + "\{}_fft without windowing.png".format(os.path.basename(file)))

        meas.data = meas.movingTimeFilter(np.hamming(400), starttime=0.011, speed=200, showImages=None)
        #
        # meas.plot2D(meas.data, xLabel="Time (s)", yLabel="Pos (m)", title=os.path.basename(file) + "raw after windowing 400 0.011 200", clickEvent=None,
        #             showImages=None, outFile=outdir + "\{}_raw after windowing 400 0.011 200.png".format(os.path.basename(file)),
        #             colorbar=True, xRange=(0.01, 0.014), yRange=None)

        meas.plot2D(abs(meas.dataFFT), "Frequency (Hz)", "Wavenumber (1/m)",
                    title=os.path.basename(file) + "_fft after windowing 400 0.011 200", clickEvent=None,
                    showImages=None, colorbar=True, xRange=(2000, 4000), yRange=(0, 20),
                    outFile=outdir + "\{}_fft after windowing 400 0.011 200.png".format(os.path.basename(file)))

        'Basic fft data- Frequency-wavenumber'
        # meas.plot2D(abs(meas.dataFFT), "Frequency (Hz)", "Wavenumber (1/m)", title=os.path.basename(file) + "_basic_fft", clickEvent=None,
        #             showImages=True, colorbar=True , xRange=(2000, 4000), yRange=(-20, 20), outFile = outdir + "\{}_basic_fft all fft.png".format(os.path.basename(file)))
        # outFile = outdir + "\{}_basic_fft.png".format(os.path.basename(file)),

        'Wavenumber slice, frequency-amplitude data'
        # wavenumber = meas.findNearestIndex(meas.dataFFT.index, 7.14)
        # meas.plot1D(abs(meas.dataFFT.iloc[wavenumber]), "Frequency [Hz]", "Amplitude [g]", title=os.path.basename(file) + "_wavenumber_slice",
        #     showImages=None, outFile=outdir + "\{}_wavenumber_slice.png".format(os.path.basename(file)), xRange=(1500, 4500))

        'Frequency slice, Wavenumber-amplitude data'
        # frequency = meas.findNearestIndex(meas.dataFFT.columns, freq)
        # meas.plot1D(abs(meas.dataFFT)[meas.dataFFT.columns[frequency]], "Wavenumber [1/m]", "Amplitude [g]", title=os.path.basename(file) + "_fequency_slice_{}".format(freq),
        #                     showImages=True, outFile=outdir + "\{}_frequency_slice_{}.png".format(os.path.basename(file), frequency), xRange=(0, 20))

        'this plots time-amplitude for a specific sensor positions ex.0.06'
        #meas.plot1D(meas.data.ix[0.06], "Time (s)", "Amplitude (g)", title=os.path.basename(file), showImages=None)

        plt.close()


if __name__ == "__main__":
   #  logging.basicConfig(level=logging.DEBUG)
   #  topDir = "T:\Data\Bug-shooter\Bugshooter XML\Phantom 2.5 redo 15.06.17"
   #  outdir = "T:\Data\Bug-shooter\Bugshooter Plots\Without any preprocessing\Phantom 2.5 redo"
   # meas = bsmdMeasurement.bsmdMeasurement(xFFTNr = 25000, yFFTNr = 3500)
   #  filelist = find_files("*.xml", topDir)
   # # point = (4,5,6)
   # # number = (1,2,3,4,5,6,7)
   # # for point in point:
   # #     for n in number:
   # #         filelist = find_files("point{}_r{}5mm.xml".format(point,n),topDir)
   # #         raw_data_plot(meas, filelist, 3000)
   # # filelist = filelist[3]
   #  raw_data_plot(meas,filelist, 3000)
   #
   logging.basicConfig(level=logging.DEBUG)
   logging.basicConfig(level=logging.DEBUG)
   # topDir = r"T:\Project data and results (svn)\Cortical Thickness Mohit\XML data\Phantom_radial_1.5-7.5mm"
   # outdir = r"T:\Project data and results (svn)\Cortical Thickness Mohit\testing directory Temporary"
   topDir = r"D:\ETH\Work\Latest Download\ETH\project data and results\Cortical Thickness Mohit\XML data\Phantom_radial_1.5-7.5mm"
   outdir = r"D:\ETH\Work\Latest Download\Manuscript\Manuscript revision"

   # meas = bsmdMeasurement.bsmdMeasurement(xFFTNr=25000, yFFTNr=3500)
   # # filelist = find_files("point1_r75mm.xml", topDir)
   # point = (0,2)
   # number = (5,6,7)
   # for point in point:
   #     for n in number:
   #         filelist = find_files("point{}_r{}5mm.xml".format(point,n),topDir)
   #         raw_data_plot(meas, filelist, 3000)
   # filelist = filelist[3]


   raw_data_plot(meas, filelist, 3000)
   '''change bsmdAlgorithm import statement'''
   # algo = bsmdAlgorithm.PhaseVelocity_2DFFTGauss()
   # result = algo.phaseVelocity(meas, freq=(2999, 3001), wavenumber=(0,20), debug_plot=True, sumOfSquares_threshold=1)
   # print(result)

