import os
import sys
# file_dir = os.path.dirname(__file__)
sys.path.append("D:\ETH\Work\Latest Download\ETH\python")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import bsmd.algorithm.bsmdAlgorithmRestructured as bsmdAlgorithm
# import bsmd.algorithm.bsmdAlgorithm as bsmdAlgorithm
import bsmd.measurement.bsmdMeasurement as bsmdMeasurement
import matplotlib.pyplot as plt
import CTS.mopa_data_review as mopa_data_review

import pandas as pd
import logging
from itertools import cycle
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from CTS.mopa_2dgauss import twoD_gaussian as gauss
import argparse
import pickle
import fnmatch
from pandas.tools.plotting import scatter_matrix
from  itertools import combinations
from pandas.tools.plotting import table
import seaborn as sns
from matplotlib import pylab
import csv
'''for pandas pickle version fail'''
import sys
import pandas.core.indexes
sys.modules['pandas.indexes'] = pandas.core.indexes



####################################
##      functions with verified input values for Radial Phantom study
####################################
def saving_dataframes():
    '''function finds the .xml data files from topdir and runs analysis. saves the dataframe of results as pickles to outdir.
    dispersion_plot=True and outfile=file saves the plot of discpersion relation plotted over 2Dfft plot'''

    logging.basicConfig(level=logging.DEBUG)
    meas = bsmdMeasurement.bsmdMeasurement(xFFTNr=25000, yFFTNr=3500)
    topDir = r"D:\ETH\Work\Latest Download\ETH\project data and results\Cortical Thickness Mohit\XML data\Phantom_radial_1.5-7.5mm"
    outdir = r"D:\ETH\Work\Latest Download\Work from home\Radial"
    filelist = mopa_data_review.find_files("point1_r75mm.xml", topDir)

    parameter_list = pd.read_excel(r'D:\ETH\Work\Latest Download\ETH\project data and results\Cortical Thickness Mohit\Results\Radial\All algorithms\Result Dataframe Pickles\parameter record.xlsx')
    parameter_list.index = parameter_list['filename']



    for file in filelist:
        print("Analyzing file: {}".format(file))
        meas.readStepPhantomXMLData(file)
        freq1 = parameter_list['freq1'][os.path.basename(file)]
        freq2 = parameter_list['freq2'][os.path.basename(file)]
        iW1 = parameter_list['iW1'][os.path.basename(file)]
        iW2 = parameter_list['iW2'][os.path.basename(file)]
        print('parameters for {}- frquency range = ({},{}), wavenumber range = ({},{})'.format(os.path.basename(file),freq1, freq2, iW1, iW2))



        '''PhaseVelocity_2DFFTGauss,( try and except makes sure that subsequent files are run even if error occurs in one of them)'''
        # try:
        #     result = algo.phaseVelocity(meas, freq=(2500, 3500), wavenumber=(2, 14),  p0=(10000., 8., 0.7), mode='linear',
        #                             dispersion_plot=True, showImages=None, window= True,
        #                             sliced_plot=None, debug_plot=None, outdir=None, sumOfSquares_threshold=0.1,
        #                             normalize=True, oldschool=None, conditional_gauss=True, outFile = outdir + "\{}_{}_linear_dispersion overlap.png".format(os.path.basename(file),
        #                                                                  algo.__class__.__name__),
        #                             title="{}_{}_linear_dispersion overlap.png".format(os.path.basename(file),
        #                                                                                algo.__class__.__name__))
        #
        #
        #     pickle.dump(result, open(outdir + "\{}_{}_basicData.p".format(os.path.basename(file), algo.__class__.__name__), "wb"))
        # except:
        #     pass

        '''PhaseVelocity_2DFFTSinc,( try and except makes sure that subsequent files are run even if error occurs in one of them)'''
        # try:
        #     result = algo.phaseVelocity(meas,  freq=(2600, 3400), wavenumber=(4, 10), p0=(10000., 8., 0.7, 1000), mode='linear',
        #                             dispersion_plot=True, showImages=None, window= True,
        #                             sliced_plot=None, debug_plot=None, outdir=None, sumOfSquares_threshold=0.5,
        #                             normalize=True, oldschool=None, conditional_gauss=True, outFile = outdir + "\{}_{}_linear_dispersion overlap.png".format(os.path.basename(file),
        #                                                                  algo.__class__.__name__),
        #                             title="{}_{}_linear_dispersion overlap.png".format(os.path.basename(file),
        #                                                                                algo.__class__.__name__))
        #
        #
        #     pickle.dump(result, open(outdir + "\{}_{}_basicData.p".format(os.path.basename(file), algo.__class__.__name__), "wb"))
        # except:
        #     pass

        '''PhaseVelocity_2DFFTMax'''
        # try:
        #     result = algo.phaseVelocity(meas, freq=(freq1, freq2),wavenumber=(2, 16), mode='linear',
        #                                 dispersion_plot=True, showImages=None, window=True,
        #                                 sliced_plot=None, debug_plot=None, outdir=None, sumOfSquares_threshold=0.1,
        #                                 normalize=True, oldschool=None, conditional_gauss=True,
        #                                 outFile=outdir + "\{}_{}_linear_dispersion overlap.png".format(os.path.basename(file), algo.__class__.__name__),
        #                                 title="{}_{}_linear_dispersion overlap.png".format(os.path.basename(file), algo.__class__.__name__))
        #
        #     pickle.dump(result, open(outdir + "\{}_{}_basicData.p".format(os.path.basename(file), algo.__class__.__name__), "wb"))
        #
        # except:
        #     pass

        '''PhaseVelocity_2DGaussianfit'''
        # try:
        #     result = algo.phaseVelocity(meas, freq=(freq1, freq2), wavenumber=(iW1, iW2), p0=(15000, (freq1+freq2)/2 , 11, 1, 1, 0.5),dispersion_plot=True, showImages=None, window= True,
        #                             sliced_plot=None, debug_plot=None, sumOfSquares_threshold=None ,
        #                                 outFile=outdir + "\{}_{}_linear_dispersion overlap.png".format(os.path.basename(file), algo.__class__.__name__),
        #                                 title="{}_{}_linear_dispersion overlap.png".format(os.path.basename(file),
        #                                                                                algo.__class__.__name__))
        #     pickle.dump(result,
        #                 open(outdir + "\{}_{}_basicData.p".format(os.path.basename(file), algo.__class__.__name__), "wb"))
        #
        # except:
        #     pass

        '''ampconst_sigmafixed'''
        # try:
        #     result = algo.phaseVelocity(meas, freq=(freq1, freq2), wavenumber=(iW1, iW2),  p0=(15000,  0.001, 1), mode='linear',
        #                             dispersion_plot=True, showImages=True, window= True,
        #                             sliced_plot=None, debug_plot=None, outdir=None, sumOfSquares_threshold=0.1,
        #                             normalize=True, oldschool=None, conditional_gauss=True, outFile = outdir + "\{}_{}_linear_dispersion overlap.png".format(os.path.basename(file),
        #                                                                  algo.__class__.__name__),
        #                             title="{}_{}_linear_dispersion overlap.png".format(os.path.basename(file),
        #                                                                                algo.__class__.__name__))
        #
        #
        #     pickle.dump(result, open(outdir + "\{}_{}_basicData.p".format(os.path.basename(file), algo.__class__.__name__), "wb"))
        # except:
        #     pass

        '''ampConst_sigmafree'''
        # try:
        #     result = algo.phaseVelocity(meas, freq=(freq1, freq2), wavenumber=(freq1, freq2),  p0=(15000, 0.001, 1, 1), mode='linear',
        #                             dispersion_plot=True, showImages=None, window= True,
        #                             sliced_plot=None, debug_plot=None, outdir=None, sumOfSquares_threshold=0.1,
        #                             normalize=True,outFile = outdir + "\{}_{}_linear_dispersion overlap.png".format(os.path.basename(file),
        #                                                                      algo.__class__.__name__),
        #                             title="{}_{}_linear_dispersion overlap.png".format(os.path.basename(file),
        #                                                                                algo.__class__.__name__))
        #
        #     pickle.dump(result, open(outdir + "\{}_{}_basicData.p".format(os.path.basename(file), algo.__class__.__name__), "wb"))
        # except:
        #     pass

        '''ampgaussian_sigmafixed'''
        # try:
        #     result = algo.phaseVelocity(meas, freq=(freq1, freq2), wavenumber=(iW1, iW2), p0=(15000, 0.001, 1, 3000, 1),
        #                                 mode='linear',
        #                                 dispersion_plot=True, showImages=None, window=True,
        #                                 sliced_plot=None, debug_plot=None, sumOfSquares_threshold=0.1,
        #                                 normalize=None, outFile=outdir + "\{}_{}_linear_dispersion overlap.png".format(os.path.basename(file), algo.__class__.__name__),
        #                                 title="{}_{}_linear_dispersion overlap.png".format(os.path.basename(file),
        #                                                                                    algo.__class__.__name__))
        #
        #     pickle.dump(result,
        #                 open(outdir + "\{}_{}_basicData.p".format(os.path.basename(file), algo.__class__.__name__),
        #                      "wb"))
        # except:
        #     pass

        '''ampgaussian_sigmafree'''
        try:
            result = algo.phaseVelocity(meas, freq=(freq1, freq2), wavenumber=(iW1,iW2),
                                        p0=(20000, 0.001, 1, (freq1 + freq2) / 2, 550, 5), mode='linear',
                                        dispersion_plot=True, showImages=None, window=True,
                                        sliced_plot=None, debug_plot=None, outdir=None, sumOfSquares_threshold=None,
                                        normalize=None, outFile=outdir + "\{}_{}_linear_dispersion overlap dual whole.eps".format(
                    os.path.basename(file),
                    algo.__class__.__name__),
                                        title="{}_{}_linear_dispersion overlap dual whole.png".format(os.path.basename(file),
                                                                                           algo.__class__.__name__))

            pickle.dump(result, open(outdir + "\{}_{}_basicData.p".format(os.path.basename(file), algo.__class__.__name__), "wb"))
        except:
            pass

        '''ampsemigaussian_sigmafixed'''
        # try:
        # result = algo.phaseVelocity(meas, freq=(freq1, freq2), wavenumber=(iW1, iW2), p0=(15000, 0.001, 1),
        #                             mode='linear',
        #                             dispersion_plot=True, showImages=None, window=True,
        #                             sliced_plot=None, debug_plot=None, outdir=None, sumOfSquares_threshold=0.1,
        #                             normalize=None, outFile=outdir + "\{}_{}_linear_dispersion overlap.png".format(
        #         os.path.basename(file),
        #         algo.__class__.__name__),
        #                             title="{}_{}_linear_dispersion overlap.png".format(os.path.basename(file),
        #                                                                                algo.__class__.__name__))
        #
        # pickle.dump(result,
        #             open(outdir + "\{}_{}_basicData.p".format(os.path.basename(file), algo.__class__.__name__),
        #                  "wb"))
        # except:
        #     pass

        '''ampsemigaussian_sigmafree'''
        # try:
        # result = algo.phaseVelocity(meas, freq=(freq1, freq2), wavenumber=(4.2, 7.5), p0=(15000, 0.001, 1, 1),
        #                             mode='linear',
        #                             dispersion_plot=True, showImages=None, window=True,
        #                             sliced_plot=None, debug_plot=None, outdir=None, sumOfSquares_threshold=0.1,
        #                             normalize=None, outFile=outdir + "\{}_{}_linear_dispersion overlap.png".format(
        #         os.path.basename(file),
        #         algo.__class__.__name__),
        #                             title="{}_{}_linear_dispersion overlap.png".format(os.path.basename(file),
        #                                                                                algo.__class__.__name__))
        #
        # pickle.dump(result,
        #                 open(outdir + "\{}_{}_basicData.p".format(os.path.basename(file), algo.__class__.__name__),
        #                      "wb"))
        # except:
        #     pass

def file_listing(algo):
    logging.basicConfig(level=logging.DEBUG)
    meas = bsmdMeasurement.bsmdMeasurement(xFFTNr=25000, yFFTNr=3500)
    topDir = r"D:\ETH\Work\Latest Download\Work from home\Radial\Uncert pickles"


    filelist = []
    fileclass = []
    for phantom in (15,25,35,45,55,65,75):
        for point in (1,2,3,4,5,6):
            file = mopa_data_review.find_files("point{}_r{}mm.xml_{}_basicData.p".format(point, phantom , algo.__class__.__name__), topDir)
            filelist = filelist + file
        fileclass.append(filelist)
        filelist = [] #resets the filelist so that each phantom has 6 element string in fileclass, otherwise the list goes on adding filenames for next phantoms
    print(fileclass)
    return(fileclass)

def frequency_vs_velocity(algo):

    topDir = r"T:\Project data and results (svn)\Cortical Thickness Mohit\Results\Radial\All algorithms\Result Dataframe Pickles"
    outdir = r"T:\Project data and results (svn)\Cortical Thickness Mohit\Results\Radial\Final 3 Algorithms\frequency vs velocity"

    fileclass = file_listing(algo)
    for phantom in (0,1,2,3,4,5,6):
        point = 0
        for file in fileclass[phantom]:
            point = point + 1
            print("Analyzing file: {}".format(file))
            result = pickle.load(open(file, "rb")).T
            # print(result["velocity"])
            plt.plot(result["velocity"], label= "point{} phantom {}".format(point , phantom + 1))
            plt.legend()
            plt.grid(True)
            plt.xlabel('frequency')
            plt.ylabel('phase velocity')
            plt.title('phantom {}_{}_frequency_vs_velocity'.format(phantom+1, algo.__class__.__name__))
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(8, 6)

        outFile = outdir + "\phantom {}_{}_frequency_vs_velocity.png".format(phantom+1, algo.__class__.__name__)
        plt.savefig(outFile)
        # plt.show()
        plt.close()

def sensorpoint_vs_velocity(algo, analysis_freq):

    outdir = r"T:\Project data and results (svn)\Cortical Thickness Mohit\Results\Radial\Final 3 Algorithms\Sensor point vs velocity"

    fileclass = file_listing(algo)
    velocity_collect = []
    uncert_collect = []
    for phantom in (0,1,2,3,4,5,6):
        point = 0
        for file in fileclass[phantom]:
            point = point + 1
            print("Analyzing file: {}".format(file))
            result = pickle.load(open(file, "rb")).T
            frequency = result.index[bsmdMeasurement.bsmdMeasurement.findNearestIndex(result.index, analysis_freq)]
            # print(result)
            uncert = result.ix[frequency][3]
            velocity = result.ix[frequency][0]
            velocity_collect.append(velocity)
            uncert_collect.append(uncert)
            print(frequency, velocity, uncert)

        points = [1,2,3,4,5,6]
        print(velocity_collect)
        plt.errorbar(points, velocity_collect, yerr=uncert_collect, label= "phantom {}".format(phantom + 1))
        plt.legend()
        plt.grid(True)
        plt.xlabel('sensor points')
        plt.ylabel('phase velocity')
        plt.title("phantom {}_{}_Sensor point vs Velocity @3000Hz".format(phantom + 1, algo.__class__.__name__))

        figure = plt.gcf()  # get current figure
        figure.set_size_inches(8, 6)
        outFile = outdir + "\phantom {}_{}_Sensor_point_vs_Velocity.png".format(phantom + 1, algo.__class__.__name__)

        # plt.savefig(outFile)
        plt.show()
        plt.close()
        uncert_collect = []
        velocity_collect = []

def sensorpoint_vs_velocity_all(analysis_freq):

    outdir = r"T:\Project data and results (svn)\Cortical Thickness Mohit\Results\Radial\temp destination"

    algo_list = (bsmdAlgorithm.ampgaussian_sigmafree(), bsmdAlgorithm.ampsemigaussian_sigmafixed(),
                 bsmdAlgorithm.PhaseVelocity_2DGaussianfit())

    for algo in algo_list:
        fileclass = file_listing(algo)
        velocity_collect = []
        uncert_collect = []
        for phantom in (0,1,2,3,4,5,6):
            point = 0
            for file in fileclass[phantom]:
                point = point + 1
                print("Analyzing file: {}".format(file))
                result = pickle.load(open(file, "rb")).T
                frequency = result.index[bsmdMeasurement.bsmdMeasurement.findNearestIndex(result.index, analysis_freq)]
                # print(result)
                uncert = result.ix[frequency][3]
                velocity = result.ix[frequency][0]
                velocity_collect.append(velocity)
                uncert_collect.append(uncert)
                print(frequency, velocity, uncert)

            points = [1,2,3,4,5,6]
            plt.errorbar(points, velocity_collect, yerr=uncert_collect, label= "phantom {}".format(phantom + 1))
            plt.legend()
            plt.grid(True)
            plt.xlabel('sensor points')
            plt.ylabel('phase velocity')
            plt.title("Sensor point vs Velocity _{}_@3000Hz".format( algo.__class__.__name__))
            velocity_collect = []
            uncert_collect = []

        figure = plt.gcf()  # get current figure
        figure.set_size_inches(8, 6)
        outFile = outdir + "\{}_Sensor_point_vs_Velocity.png".format( algo.__class__.__name__)

        plt.savefig(outFile)
        plt.show()
        plt.close()
        uncert_collect = []
        velocity_collect = []

def average_ss_listings(algo):
    '''lists all the average_ss values for all the files and all the algorithms'''
    outdir = r"T:\Project data and results (svn)\Cortical Thickness Mohit\Results\Radial\Final 3 Algorithms\Avg_ss listings"

    fileclass = file_listing(algo)
    for phantom in (0, 1, 2, 3, 4, 5, 6):
        point = 0
        for file in fileclass[phantom]:
            point = point + 1
            print("Analyzing file: {}".format(file))
            result = pickle.load(open(file, "rb")).T
            # print(result["velocity"])
            plt.plot(result["average_SS"], label="point{} phantom {}".format(point, phantom + 1))
            plt.legend()
            plt.grid(True)
            plt.xlabel('frequency')
            plt.ylabel('Average_Ss')
            plt.title("phantom {}_{}_avg_ss.png".format(phantom + 1, algo.__class__.__name__))
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(8, 6)

        outFile = outdir + "\phantom {}_{}_avg_ss.png".format(phantom + 1, algo.__class__.__name__)
        plt.savefig(outFile)
        # plt.show()
        plt.close()

def average_ss_check():

    bsmdAlgorithm.PhaseVelocity_2DFFTGauss(), bsmdAlgorithm.ampConst_sigmafixed(), bsmdAlgorithm.ampConst_sigmafree(),
    algo_list = (bsmdAlgorithm.PhaseVelocity_2DGaussianfit(), bsmdAlgorithm.ampgaussian_sigmafree(), bsmdAlgorithm.ampsemigaussian_sigmafixed())

    algo_avg_ss = []
    for algo in algo_list:
        fileclass = file_listing(algo)
        avg_ss_combined = []
        for phantom in (0, 1, 2, 3, 4, 5, 6):
            point = 0
            for file in fileclass[phantom]:
                point = point + 1
                print("Analyzing file: {}".format(file))
                result = pickle.load(open(file, "rb")).T
                avg_ss = np.mean(result["average_SS"])
            avg_ss_combined.append(avg_ss) #list of avg_ss for each phantom
        algo_avg_ss.append(np.mean(avg_ss_combined)) #list of averages of avg_ss of all phantoms for each algorithm
    # plt.plot(algo_avg_ss, (1,2,3,4,5,6,7,8))
    # plt.legend()
    # plt.grid(True)
    # plt.xlabel('algorithm')
    # plt.ylabel('Average_Ss')
    # plt.title("algo_comparison_avg_ss.png")
    # figure = plt.gcf()  # get current figure
    # figure.set_size_inches(8, 6)
    #
    # outFile = outdir + r"\algo_comparison_avg_ss.png"
    # plt.savefig(outFile)
    # plt.show()
    print(algo_avg_ss)

def local_thickness_vs_velocity(algo, analysis_freq):
    '''min thickness at a measurement point vs velocity'''
    outdir = r"P:\Student\Mohit\project data and results\Cortical Thickness Mohit\Results\Radial\Negative FFT analysis"

    fileclass = file_listing(algo)
    velocity_collect = []
    uncert_collect = []
    for phantom in (0, 1, 2, 3, 4, 5, 6):
        point = 0
        for file in fileclass[phantom]:
            point = point + 1
            print("Analyzing file: {}".format(file))
            result = pickle.load(open(file, "rb")).T
            frequency = result.index[bsmdMeasurement.bsmdMeasurement.findNearestIndex(result.index, analysis_freq)]
            # print(result)
            uncert = result.ix[frequency][3]
            velocity = result.ix[frequency][0]
            velocity_collect.append(velocity)
            uncert_collect.append(uncert)
            print(frequency, velocity, uncert)

        local_thickness = [[1.72,1.34,1.58,1.52,1.49,1.41], [2.95,2.18,2.66,2.54,2.48,2.32], [4.18,3.02,3.74,3.53,3.45,3.23],[5.4,3.86,4.82,4.54,4.43,4.14],[6.62,4.7,5.9,5.55,5.41,5.05], [7.85,5.54,6.99,6.57,6.4,5.96], [8.96,6.38,8.06,7.58,7.38,6.87]]


        plt.errorbar(local_thickness[phantom], velocity_collect, yerr=uncert_collect, label="phantom {}".format(phantom + 1), fmt='o')
        plt.legend()
        plt.grid(True)
        plt.xlabel('local thickness')
        plt.ylabel('phase velocity')
        plt.title("phantom {}_{}_local thickness_vs_velocity".format(phantom + 1, algo.__class__.__name__))

        figure = plt.gcf()  # get current figure
        figure.set_size_inches(8, 6)
        outFile = outdir + "\phantom {}_{}_local thickness_vs_velocity.png".format(phantom + 1, algo.__class__.__name__)
        plt.savefig(outFile)
        # plt.show()
        plt.close()
        uncert_collect = []
        velocity_collect = []

def local_thickness_42_points(algo, analysis_freq):
    '''min thickness at a measurement point vs velocity'''
    outdir = r"D:\ETH\Work\Latest Download\Work from home"

    fileclass = file_listing(algo)
    velocity_collect = []
    uncert_collect = []
    iW_collect = []
    for phantom in (0, 1, 2, 3, 4, 5, 6):
        point = 0
        for file in fileclass[phantom]:
            point = point + 1
            print("Analyzing file: {}".format(file))
            # result = pickle.load(open(file, "rb"), encoding= "latin1" ).T
            result = pickle.load(open(file, "rb")).T
            frequency = result.index[bsmdMeasurement.bsmdMeasurement.findNearestIndex(result.index, analysis_freq)]
            # print(result)
            uncert = result.ix[frequency][3]
            velocity = result.ix[frequency][0]
            iW = result.ix[frequency][5]
            velocity_collect.append(velocity)
            uncert_collect.append(uncert)
            iW_collect.append(iW)
            print(frequency, velocity, uncert)

    local_thickness = [1.72,1.34,1.58,1.52,1.49,1.41, 2.95,2.18,2.66,2.54,2.48,2.32, 4.18,3.02,3.74,3.53,3.45,3.23,5.4,3.86,4.82,4.54,4.43,4.14,6.62,4.7,5.9,5.55,5.41,5.05, 7.85,5.54,6.99,6.57,6.4,5.96, 8.96,6.38,8.06,7.58,7.38,6.87]

    '''print out the local thickness vs veloctiy values'''
    # for i in np.arange(len(local_thickness)):
    #     print(local_thickness[i],velocity_collect[i],iW_collect[i])
    # print(np.shape(local_thickness))
    # print(np.shape(velocity_collect))
    # print(np.shape(uncert_collect))



    plt.errorbar(local_thickness, velocity_collect, yerr=uncert_collect, label="{}".format(algo.__class__.__name__), fmt='o')
    plt.legend()
    plt.grid(True)
    plt.xlabel('local thickness')
    plt.ylabel('phase velocity')
    plt.title("{}_local thickness_42_points @{}.png".format(algo.__class__.__name__, analysis_freq))

    csv_file = open("D:\downloads-D\ETH\Work\Latest Download\ETH\project data and results\Cortical Thickness Mohit\Mathematica plots\id_tubetheory_mohit", "r")
    reader = csv.reader(csv_file)
    x = []
    y = []
    for line in reader:
        t = int(line[0])
        p = line[1]
        x.append(t)
        y.append(p)
    x = np.array(x) / 2000
    y = np.array(y)
    plt.plot(x, y)

    csv_file = open("D:\downloads-D\ETH\Work\Latest Download\ETH\project data and results\Cortical Thickness Mohit\Mathematica plots\id_tubetheory_mohit_perfectfit", "r")
    reader = csv.reader(csv_file)
    x = []
    y = []
    for line in reader:
        t = int(line[0])
        p = line[1]
        x.append(t)
        y.append(p)
    x = np.array(x) / 2000
    y = np.array(y)
    plt.plot(x, y)

    csv_file = open("D:\downloads-D\ETH\Work\Latest Download\ETH\project data and results\Cortical Thickness Mohit\Mathematica plots\id_lambtheory_mohit", "r")
    reader = csv.reader(csv_file)
    x_1 = []
    y_1 = []
    for line in reader:
        t = int(line[0])
        p = line[1]

        if int(float(p)) < 1000:
            y_1.append(p)
            x_1.append(t)
    x_1 = np.array(x_1) / 2000
    # print(x, y)
    # print(np.shape(x))
    # print(np.shape(y))
    plt.plot(x_1, y_1, '.')
    #
    plt.ylim(100, 1000)
    # plt.xlim(1, 9)

    figure = plt.gcf()  # get current figure
    figure.set_size_inches(8, 6)
    outFile = outdir + "\{}_local thickness_42_points @{} kx0.927.png".format(algo.__class__.__name__, analysis_freq)

    plt.savefig(outFile)
    plt.show()
    plt.close()
    uncert_collect = []
    velocity_collect = []

def each_phantom_separate(algo, analysis_freq, phantom):
    '''min thickness at a measurement point vs velocity'''
    outdir = r"D:\ETH\Work\Latest Download\Work from home"

    fileclass = file_listing(algo)
    velocity_collect = []
    uncert_collect = []
    iW_collect = []
    # for phantom in (0, 1, 2, 3, 4, 5, 6):
    point = 0
    for file in fileclass[phantom-1]:
        point = point + 1
        # print("Analyzing file: {}".format(file))
        result = pickle.load(open(file, "rb")).T
        frequency = result.index[bsmdMeasurement.bsmdMeasurement.findNearestIndex(result.index, analysis_freq)]
        # print(result)
        uncert = result.ix[frequency][3]
        velocity = result.ix[frequency][0]
        iW = result.ix[frequency][5]
        velocity_collect.append(velocity)
        uncert_collect.append(uncert)
        iW_collect.append(iW)
        print(file, frequency, velocity, iW, uncert)

    local_thickness = [[1.72,1.34,1.58,1.52,1.49,1.41], [2.95,2.18,2.66,2.54,2.48,2.32], [4.18,3.02,3.74,3.53,3.45,3.23], [5.4,3.86,4.82,4.54,4.43,4.14], [6.62,4.7,5.9,5.55,5.41,5.05], [7.85,5.54,6.99,6.57,6.4,5.96], [8.96,6.38,8.06,7.58,7.38,6.87]]

    '''print out the local thickness vs veloctiy values'''
    # for i in np.arange(len(local_thickness)):
    #     print(local_thickness[i],velocity_collect[i],iW_collect[i])
    # print(np.shape(local_thickness))
    # print(np.shape(velocity_collect))
    # print(np.shape(uncert_collect))



    plt.errorbar(local_thickness[phantom-1], velocity_collect, yerr=uncert_collect, label="{}".format(algo.__class__.__name__), fmt='o')
    plt.legend()
    plt.grid(True)
    plt.xlabel('local thickness')
    plt.ylabel('phase velocity')
    plt.title("\{}_phantom{} vs tube theory @{}.png".format(algo.__class__.__name__,phantom, analysis_freq))

    csv_file = open("T:\mathematica-svn\phantomResults\id_tubetheory_mohit_phantom{}".format(phantom), "r")
    reader = csv.reader(csv_file)
    x = []
    y = []
    for line in reader:
        t = int(line[0])
        p = line[1]
        x.append(t)
        y.append(p)
    x = np.array(x) / 1500
    y = np.array(y)
    plt.plot(x, y)

    csv_file = open("T:\mathematica-svn\phantomResults\id_lambtheory_mohit", "r")
    reader = csv.reader(csv_file)
    x_1 = []
    y_1 = []
    for line in reader:
        t = int(line[0])
        p = line[1]

        if int(float(p)) < 1000:
            y_1.append(p)
            x_1.append(t)
    x_1 = np.array(x_1) / 1500
    # print(x, y)
    # print(np.shape(x))
    # print(np.shape(y))
    plt.plot(x_1, y_1, '.')

    plt.ylim(100, 600)
    plt.xlim(1, 8)

    figure = plt.gcf()  # get current figure
    figure.set_size_inches(8, 6)
    outFile = outdir + "\{}_phantom{} vs tube theory @{} kx0.95.png".format(algo.__class__.__name__,phantom, analysis_freq)

    plt.savefig(outFile)
    plt.show()
    plt.close()
    uncert_collect = []
    velocity_collect = []

def selective_thickness_42_points(algo, analysis_freq):
    '''min thickness at a measurement point vs velocity'''
    outdir = r"T:\Project data and results (svn)\Cortical Thickness Mohit\Results\Radial\temp destination"

    fileclass = file_listing(algo)
    velocity_collect = []
    uncert_collect = []
    iW_collect = []
    selective_velocity = []
    selective_thickness = []
    selective_uncert = []

    for phantom in (0, 1, 2, 3, 4, 5, 6):
        point = 0
        for file in fileclass[phantom]:
            point = point + 1
            print("Analyzing file: {}".format(file))
            result = pickle.load(open(file, "rb")).T
            frequency = result.index[bsmdMeasurement.bsmdMeasurement.findNearestIndex(result.index, analysis_freq)]
            # print(result)
            uncert = result.ix[frequency][3]
            velocity = result.ix[frequency][0]
            iW = result.ix[frequency][5]
            velocity_collect.append(velocity)
            uncert_collect.append(uncert)
            iW_collect.append(iW)
            # print(frequency, velocity, uncert)

    local_thickness = [1.72,1.34,1.58,1.52,1.49,1.41, 2.95,2.18,2.66,2.54,2.48,2.32, 4.18,3.02,3.74,3.53,3.45,3.23,5.4,3.86,4.82,4.54,4.43,4.14,6.62,4.7,5.9,5.55,5.41,5.05, 7.85,5.54,6.99,6.57,6.4,5.96, 8.96,6.38,8.06,7.58,7.38,6.87]

    'choose specific sensor postions'
    next = [5]
    points = [next]
    for i in [1, 2, 3, 4, 5, 6]:
        next = [x + 6 for x in next]
        points.append(next)
    points = [item for sublist in points for item in sublist]
    print(points)

    for point in points:
        selective_velocity.append(velocity_collect[point])
        selective_thickness.append(local_thickness[point])
        selective_uncert.append(uncert_collect[point])


    '''print out the local thickness vs veloctiy values'''
    for i in np.arange(len(selective_thickness)):
        print(selective_thickness[i],selective_velocity[i])
    # print(np.shape(local_thickness))
    # print(np.shape(velocity_collect))
    # print(np.shape(uncert_collect))



    plt.errorbar(selective_thickness, selective_velocity, yerr=selective_uncert, label="{}".format(algo.__class__.__name__), fmt='o')
    plt.legend()
    plt.grid(True)
    plt.xlabel('selected points')
    plt.ylabel('phase velocity')
    plt.title("selected points {} @{}_{}.png".format(6, analysis_freq,algo.__class__.__name__))

    csv_file = open("T:\mathematica-svn\phantomResults\id_tubetheory_mohit", "r")
    reader = csv.reader(csv_file)
    x = []
    y = []
    for line in reader:
        t = int(line[0])
        p = line[1]
        x.append(t)
        y.append(p)
    x = np.array(x) / 1500
    y = np.array(y)
    plt.plot(x, y)

    csv_file = open("T:\mathematica-svn\phantomResults\id_lambtheory_mohit", "r")
    reader = csv.reader(csv_file)
    x_1 = []
    y_1 = []
    for line in reader:
        t = int(line[0])
        p = line[1]

        if int(float(p)) < 1000:
            y_1.append(p)
            x_1.append(t)
    x_1 = np.array(x_1) / 1500
    # print(x, y)
    # print(np.shape(x))
    # print(np.shape(y))
    plt.plot(x_1, y_1, '.')

    plt.ylim(100, 900)
    plt.xlim(1, 10)

    figure = plt.gcf()  # get current figure
    figure.set_size_inches(8, 6)
    outFile = outdir + "\selected points {} @{}_{}.png".format(6, analysis_freq,algo.__class__.__name__)

    plt.savefig(outFile)
    plt.show()
    plt.close()
    uncert_collect = []
    velocity_collect = []

def velocity_vs_thickness_lambda(algo, analysis_freq):
    '''min thickness at a measurement point vs velocity'''
    outdir = r"T:\Project data and results (svn)\Cortical Thickness Mohit\Results\Radial\velocity vs thickness-lambda"

    fileclass = file_listing(algo)
    velocity_collect = []
    iW_collect = []
    uncert_collect = []
    for phantom in (0, 1, 2, 3, 4, 5, 6):
        point = 0
        for file in fileclass[phantom]:
            point = point + 1
            print("Analyzing file: {}".format(file))
            result = pickle.load(open(file, "rb")).T
            frequency = result.index[bsmdMeasurement.bsmdMeasurement.findNearestIndex(result.index, analysis_freq)]
            # print(result)
            uncert = result.ix[frequency][3]
            velocity = result.ix[frequency][0]
            iW = result.ix[frequency][5]
            iW_collect.append(iW)
            velocity_collect.append(velocity)
            uncert_collect.append(uncert)
            # print(frequency, velocity, uncert)

    local_thickness = [1.72,1.34,1.58,1.52,1.49,1.41, 2.95,2.18,2.66,2.54,2.48,2.32, 4.18,3.02,3.74,3.53,3.45,3.23,5.4,3.86,4.82,4.54,4.43,4.14,6.62,4.7,5.9,5.55,5.41,5.05, 7.85,5.54,6.99,6.57,6.4,5.96, 8.96,6.38,8.06,7.58,7.38,6.87]
    thickness_lambda = np.array(local_thickness)*np.array(iW_collect)

    plt.errorbar(thickness_lambda, velocity_collect, yerr=uncert_collect, label="{}".format(algo.__class__.__name__), fmt='o')
    plt.legend()
    plt.grid(True)
    plt.xlabel('thickness/lambda')
    plt.ylabel('phase velocity')
    plt.title("{}_thickness-lambda vs velocity @{}.png".format(algo.__class__.__name__, analysis_freq))

    # # Calculate trendline
    # coeffs = np.polyfit(thickness_lambda, velocity_collect, 2)
    #
    # intercept = coeffs[-1]
    # slope = coeffs[-2]
    # power = coeffs[0]
    #
    # minxd = np.min(thickness_lambda)
    # maxxd = np.max(thickness_lambda)
    #
    # xl = np.array([minxd, maxxd])
    # yl = power * xl ** 2 + slope * xl + intercept
    #
    # # Plot trendline
    # plt.plot(xl, yl, 'r', alpha=1)

    z = np.polyfit(thickness_lambda, velocity_collect, 3)
    f = np.poly1d(z)

    # calculate new x's and y's
    minxd = np.min(thickness_lambda)
    maxxd = np.max(thickness_lambda)
    x_new = np.array([minxd, maxxd])
    y_new = f(x_new)

    plt.plot(thickness_lambda, velocity_collect, 'o', x_new, y_new)
    ax = plt.gca()
    ax.set_axis_bgcolor((0.898, 0.898, 0.898))

    figure = plt.gcf()  # get current figure
    figure.set_size_inches(8, 6)
    outFile = outdir + "\{}_thickness-lamda @{}.png".format(algo.__class__.__name__, analysis_freq)
    plt.savefig(outFile)
    plt.show()
    plt.close()
    uncert_collect = []
    velocity_collect = []

def cross_area_vs_velocity(algo, analysis_freq):
    '''avg velocity of each phantom vs cross sectional area of each phantom'''
    outdir = r"T:\Project data and results (svn)\Cortical Thickness Mohit\Results\Radial\Final 3 Algorithms\Cross sectional area vs velocity"

    fileclass = file_listing(algo)
    velocity_collect = []
    uncert_collect = []
    velocity_avg = []
    uncert_avg = []
    for phantom in (0, 1, 2, 3, 4, 5, 6):

        for file in fileclass[phantom]:

            print("Analyzing file: {}".format(file))
            result = pickle.load(open(file, "rb")).T
            frequency = result.index[bsmdMeasurement.bsmdMeasurement.findNearestIndex(result.index, analysis_freq)]
            # print(result)
            uncert = result.ix[frequency][3]
            velocity = result.ix[frequency][0]
            velocity_collect.append(velocity)
            uncert_collect.append(uncert)


        avg_vel = np.average(velocity_collect)
        avg_uncert = np.average(uncert_collect)
        velocity_avg.append(avg_vel)
        uncert_avg.append(avg_uncert)
        velocity_collect = []
        uncert_collect = []

    cross_area = [124.47,198.89,265.88,328.23,383.56,432.59,474.56]
    plt.errorbar(cross_area, velocity_avg, label= "algorithm {}".format(algo.__class__.__name__), yerr=uncert_avg)
    plt.legend()
    plt.grid(True)
    plt.xlabel('Cross sectional area of phantom')
    plt.ylabel('avg velocity')
    plt.title('avg velocity vs cross sectional area')

    figure = plt.gcf()  # get current figure
    figure.set_size_inches(8, 6)
    outFile = outdir + "\{}_cross_area_vs_velocity.png".format(algo.__class__.__name__)
    plt.savefig(outFile)
    # plt.show()
    plt.close()

def Algo_comparison_cross_area_vs_velocity(analysis_freq):
    '''avg velocity of each phantom vs cross sectional area of each phantom'''
    outdir = r"T:\Project data and results (svn)\Cortical Thickness Mohit\Results\Radial\temp destination\doule peak removal\cross section"

    algo_list = (bsmdAlgorithm.ampgaussian_sigmafree(), bsmdAlgorithm.ampsemigaussian_sigmafixed(),
                bsmdAlgorithm.PhaseVelocity_2DGaussianfit())

    for algo in algo_list:
        fileclass = file_listing(algo)
        velocity_collect = []
        uncert_collect = []
        velocity_avg = []
        uncert_avg = []
        for phantom in (0, 1, 2, 3, 4, 5, 6):

            for file in fileclass[phantom]:

                print("Analyzing file: {}".format(file))
                result = pickle.load(open(file, "rb")).T
                frequency = result.index[bsmdMeasurement.bsmdMeasurement.findNearestIndex(result.index, analysis_freq)]
                # print(result)
                uncert = result.ix[frequency][3]
                velocity = result.ix[frequency][0]
                velocity_collect.append(velocity)
                uncert_collect.append(uncert)


            avg_vel = np.average(velocity_collect)
            avg_uncert = np.average(uncert_collect)
            velocity_avg.append(avg_vel)
            uncert_avg.append(avg_uncert)
            velocity_collect = []
            uncert_collect = []

        cross_area = [124.47,198.89,265.88,328.23,383.56,432.59,474.56]
        plt.errorbar(cross_area, velocity_avg, label= "algorithm {}".format(algo.__class__.__name__), yerr=uncert_avg)
        plt.legend()
        plt.grid(True)
        plt.xlabel('Cross sectional area of phantom')
        plt.ylabel('avg velocity')
        plt.title('avg velocity vs cross sectional area')

    figure = plt.gcf()  # get current figure
    figure.set_size_inches(8, 6)
    outFile = outdir + "\{}_cross_area_vs_velocity.png".format(algo.__class__.__name__)
    plt.savefig(outFile)
    plt.show()
    # plt.close()

def avg_velocity_vs_avg_thickness(algo, analysis_freq):

    outdir = r"T:\Project data and results (svn)\Cortical Thickness Mohit\Results\Radial\Final 3 Algorithms\avg velocity vs avg thickness"

    fileclass = file_listing(algo)
    velocity_collect = []
    uncert_collect = []
    velocity_avg = []
    uncert_avg = []
    for phantom in (0, 1, 2, 3, 4, 5, 6):

        for file in fileclass[phantom]:

            print("Analyzing file: {}".format(file))
            result = pickle.load(open(file, "rb")).T
            frequency = result.index[bsmdMeasurement.bsmdMeasurement.findNearestIndex(result.index, analysis_freq)]
            # print(result)
            uncert = result.ix[frequency][3]
            velocity = result.ix[frequency][0]
            velocity_collect.append(velocity)
            uncert_collect.append(uncert)


        avg_vel = np.average(velocity_collect)
        avg_uncert = np.average(uncert_collect)
        velocity_avg.append(avg_vel)
        uncert_avg.append(avg_uncert)
        velocity_collect = []
        uncert_collect = []

    avg_thickness = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
    plt.errorbar(avg_thickness, velocity_avg, label= "algorithm {}".format(algo.__class__.__name__), yerr=uncert_avg)
    plt.legend()
    plt.grid(True)
    plt.xlabel('avg thickness of phantom')
    plt.ylabel('avg velocity')
    plt.title("{}_avg_velocity_vs_avg_thickness.png".format(algo.__class__.__name__))

    figure = plt.gcf()  # get current figure
    figure.set_size_inches(8, 6)
    outFile = outdir + "\{}_avg_velocity_vs_avg_thickness.png".format(algo.__class__.__name__)
    plt.savefig(outFile)
    # plt.show()
    plt.close()

def avg_velocity_vs_avg_thickness_algoComparison(analysis_freq):

    outdir = r"T:\Project data and results (svn)\Cortical Thickness Mohit\Results\Radial\temp destination"

    # algo_list = (bsmdAlgorithm.PhaseVelocity_2DFFTGauss(), bsmdAlgorithm.PhaseVelocity_2DGaussianfit(),
    #              bsmdAlgorithm.PhaseVelocity_2DFFTMax(), bsmdAlgorithm.ampConst_sigmafixed(),
    #              bsmdAlgorithm.ampConst_sigmafree(),
    #              bsmdAlgorithm.ampgaussian_sigmafixed(), bsmdAlgorithm.ampgaussian_sigmafree(),
    #              bsmdAlgorithm.ampsemigaussian_sigmafixed(), bsmdAlgorithm.ampsemigaussian_sigmafree())

    algo_list = (bsmdAlgorithm.ampgaussian_sigmafree(), bsmdAlgorithm.ampsemigaussian_sigmafixed(),
                 bsmdAlgorithm.PhaseVelocity_2DGaussianfit())

    for algo in algo_list:
        fileclass = file_listing(algo)
        velocity_collect = []
        uncert_collect = []
        velocity_avg = []
        uncert_avg = []
        for phantom in (0, 1, 2, 3, 4, 5, 6):

            for file in fileclass[phantom]:
                print("Analyzing file: {}".format(file))
                result = pickle.load(open(file, "rb")).T
                frequency = result.index[bsmdMeasurement.bsmdMeasurement.findNearestIndex(result.index, analysis_freq)]
                # print(result)
                uncert = result.ix[frequency][3]
                velocity = result.ix[frequency][0]
                velocity_collect.append(velocity)
                uncert_collect.append(uncert)

            avg_vel = np.average(velocity_collect)
            avg_uncert = np.average(uncert_collect)
            velocity_avg.append(avg_vel)
            uncert_avg.append(avg_uncert)
            velocity_collect = []
            uncert_collect = []

        avg_thickness = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
        plt.errorbar(avg_thickness, velocity_avg, label="algorithm {}".format(algo.__class__.__name__), yerr=uncert_avg)
        plt.legend()
        plt.grid(True)
        plt.xlabel('avg thickness of phantom')
        plt.ylabel('avg velocity')
        plt.title('avg velocity vs avg thickness')

    csv_file = open("T:\mathematica-svn\phantomResults\id_tubetheory_mohit", "r")
    reader = csv.reader(csv_file)
    x = []
    y = []
    for line in reader:
        t = int(line[0])
        p = line[1]
        x.append(t)
        y.append(p)
    x = np.array(x) / 1500
    y = np.array(y)
    plt.plot(x, y)

    csv_file = open("T:\mathematica-svn\phantomResults\id_tubetheory_mohit_perfectfit", "r")
    reader = csv.reader(csv_file)
    x = []
    y = []
    for line in reader:
        t = int(line[0])
        p = line[1]
        x.append(t)
        y.append(p)
    x = np.array(x) / 1500
    y = np.array(y)
    plt.plot(x, y)

    csv_file = open("T:\mathematica-svn\phantomResults\id_lambtheory_mohit", "r")
    reader = csv.reader(csv_file)
    x_1 = []
    y_1 = []
    for line in reader:
        t = int(line[0])
        p = line[1]

        if int(float(p)) < 1000:
            y_1.append(p)
            x_1.append(t)
    x_1 = np.array(x_1) / 1500
    # print(x, y)
    # print(np.shape(x))
    # print(np.shape(y))
    plt.plot(x_1, y_1, '.')

    plt.ylim(200, 900)
    plt.xlim(1, 8)

    outFile = outdir + r"\avg_velocity_vs_avg_thickness_all_phantoms_together.png"
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(8, 6)
    plt.savefig(outFile)
    plt.show()
    plt.close()

def mode_velocity_vs_avg_thickness_algoComparison(mode, analysis_freq):

    outdir = r"T:\Project data and results (svn)\Cortical Thickness Mohit\Results\Radial\temp destination"

    # algo_list = (bsmdAlgorithm.PhaseVelocity_2DFFTGauss(), bsmdAlgorithm.PhaseVelocity_2DGaussianfit(),
    #              bsmdAlgorithm.PhaseVelocity_2DFFTMax(), bsmdAlgorithm.ampConst_sigmafixed(),
    #              bsmdAlgorithm.ampConst_sigmafree(),
    #              bsmdAlgorithm.ampgaussian_sigmafixed(), bsmdAlgorithm.ampgaussian_sigmafree(),
    #              bsmdAlgorithm.ampsemigaussian_sigmafixed(), bsmdAlgorithm.ampsemigaussian_sigmafree())

    algo_list = (bsmdAlgorithm.ampgaussian_sigmafree(), bsmdAlgorithm.ampsemigaussian_sigmafixed(),
                 bsmdAlgorithm.PhaseVelocity_2DGaussianfit())

    for algo in algo_list:
        fileclass = file_listing(algo)
        velocity_collect = []
        uncert_collect = []
        velocity_avg = []
        uncert_avg = []
        for phantom in (0, 1, 2, 3, 4, 5, 6):

            for file in fileclass[phantom]:
                print("Analyzing file: {}".format(file))
                result = pickle.load(open(file, "rb")).T
                frequency = result.index[bsmdMeasurement.bsmdMeasurement.findNearestIndex(result.index, analysis_freq)]
                # print(result)
                uncert = result.ix[frequency][3]
                velocity = result.ix[frequency][0]
                velocity_collect.append(velocity)
                uncert_collect.append(uncert)

            sorted_vel = np.sort(velocity_collect)
            sorted_uncert = np.sort(uncert_collect)
            mode_vel = sorted_vel[mode]
            mode_uncert = sorted_uncert[mode]
            velocity_avg.append(mode_vel)
            uncert_avg.append(mode_uncert)
            velocity_collect = []
            uncert_collect = []

        avg_thickness = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
        plt.errorbar(avg_thickness, velocity_avg, label="algorithm {}".format(algo.__class__.__name__), yerr=uncert_avg)
        plt.legend()
        plt.grid(True)
        plt.xlabel('local thickness')
        plt.ylabel('mode velocity')
        plt.title('mode velocity vs local thickness')

    csv_file = open("T:\mathematica-svn\phantomResults\id_tubetheory_mohit", "r")
    reader = csv.reader(csv_file)
    x = []
    y = []
    for line in reader:
        t = int(line[0])
        p = line[1]
        x.append(t)
        y.append(p)
    x = np.array(x) / 1500
    y = np.array(y)
    plt.plot(x, y)

    csv_file = open("T:\mathematica-svn\phantomResults\id_tubetheory_mohit_perfectfit", "r")
    reader = csv.reader(csv_file)
    x = []
    y = []
    for line in reader:
        t = int(line[0])
        p = line[1]
        x.append(t)
        y.append(p)
    x = np.array(x) / 1500
    y = np.array(y)
    plt.plot(x, y)

    csv_file = open("T:\mathematica-svn\phantomResults\id_lambtheory_mohit", "r")
    reader = csv.reader(csv_file)
    x_1 = []
    y_1 = []
    for line in reader:
        t = int(line[0])
        p = line[1]

        if int(float(p)) < 1000:
            y_1.append(p)
            x_1.append(t)
    x_1 = np.array(x_1) / 1500
    # print(x, y)
    # print(np.shape(x))
    # print(np.shape(y))
    plt.plot(x_1, y_1, '.')

    plt.ylim(5,700)
    plt.xlim(1,8)
    outFile = outdir + r"\mode_velocity {}_vs_avg_thickness_all_phantoms_together.png".format(mode)
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(8, 6)
    plt.savefig(outFile)
    plt.show()
    plt.close()

def scatterplot_algo_comparison(freq_range):

    outdir = r"T:\Project data and results (svn)\Cortical Thickness Mohit\Results\Radial\All algorithms\scatterplot_algo_comparison"

    scatterplot_dataframe = pd.DataFrame()

    algo_list = (bsmdAlgorithm.PhaseVelocity_2DFFTGauss(), bsmdAlgorithm.PhaseVelocity_2DGaussianfit(),
                 bsmdAlgorithm.PhaseVelocity_2DFFTMax(), bsmdAlgorithm.ampConst_sigmafixed(),
                 bsmdAlgorithm.ampConst_sigmafree(),
                 bsmdAlgorithm.ampgaussian_sigmafixed(), bsmdAlgorithm.ampgaussian_sigmafree(),
                 bsmdAlgorithm.ampsemigaussian_sigmafixed(), bsmdAlgorithm.ampsemigaussian_sigmafree())
    for algo in algo_list:
        fileclass = file_listing(algo)
        velocity_collect = []
        uncert_collect = []
        for phantom in (0, 1, 2, 3, 4, 5, 6):
            point = 0
            for file in fileclass[phantom]:
                point = point + 1
                print("Analyzing file: {}".format(file))
                result = pickle.load(open(file, "rb")).T
                frequency0 = result.index[bsmdMeasurement.bsmdMeasurement.findNearestIndex(result.index, freq_range[0])]
                frequency1 = result.index[bsmdMeasurement.bsmdMeasurement.findNearestIndex(result.index, freq_range[1])]
                velocity = (result["velocity"][frequency0:frequency1].values)
                velocity_collect.append(velocity)
        flat_list = [item for sublist in velocity_collect for item in sublist]
        scatterplot_dataframe["{}".format(algo.__class__.__name__)] = flat_list
    corr_coeff_mat = pd.DataFrame.corr(scatterplot_dataframe, method='pearson')
    print(corr_coeff_mat)
    scatterplot_matrix = scatter_matrix(scatterplot_dataframe, alpha= 1, figsize=(6, 6), diagonal='kde') #
    outFile = outdir + r"\scatterplot_algo_comparison.png"
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(32, 24)
    matplotlib.rc('xtick', labelsize=8)
    matplotlib.rc('ytick', labelsize=8)
    # plt.savefig(outFile)
    # plt.show()
    plt.close()
    # print(scatterplot_matrix)

    '''correlatin coefficient matrix plot'''
    sns.set(font_scale=1.5)
    sns.heatmap(corr_coeff_mat, annot=True, annot_kws={"size":12})
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(32, 24)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.savefig(outdir + r"\correlation coefficient matrix _algo_comparison.png")
    plt.show()
    plt.close()

def bland_altman_plot(freq_range):

    outdir = r"T:\Project data and results (svn)\Cortical Thickness Mohit\Results\Radial\bland-altman algo comparison"

    scatterplot_dataframe = pd.DataFrame()

    algo_list = (bsmdAlgorithm.PhaseVelocity_2DFFTGauss(), bsmdAlgorithm.PhaseVelocity_2DGaussianfit(),
                 bsmdAlgorithm.PhaseVelocity_2DFFTMax(), bsmdAlgorithm.ampConst_sigmafixed(),
                 bsmdAlgorithm.ampConst_sigmafree(),
                 bsmdAlgorithm.ampgaussian_sigmafixed(), bsmdAlgorithm.ampgaussian_sigmafree(),
                 bsmdAlgorithm.ampsemigaussian_sigmafixed(), bsmdAlgorithm.ampsemigaussian_sigmafree())
    for algo in algo_list:
        fileclass = file_listing(algo)
        velocity_collect = []
        uncert_collect = []
        for phantom in (0, 1, 2, 3, 4, 5, 6):
            point = 0
            for file in fileclass[phantom]:
                point = point + 1
                print("Analyzing file: {}".format(file))
                result = pickle.load(open(file, "rb")).T
                frequency0 = result.index[bsmdMeasurement.bsmdMeasurement.findNearestIndex(result.index, freq_range[0])]
                frequency1 = result.index[bsmdMeasurement.bsmdMeasurement.findNearestIndex(result.index, freq_range[1])]
                velocity = (result["velocity"][frequency0:frequency1].values)
                velocity_collect.append(velocity)
        flat_list = [item for sublist in velocity_collect for item in sublist]
        scatterplot_dataframe["{}".format(algo.__class__.__name__)] = flat_list

    cc = list(combinations(scatterplot_dataframe.columns, 2))
    # print(np.shape(cc))

    # bland_altman = pd.concat([df[c[1]].sub(df[c[0]]) for c in cc], axis=1, keys=cc)
    # df.columns = df.columns.map(''.join)
    # print(df)

    for c in cc:

        data1 = scatterplot_dataframe[c[1]]
        data2 = scatterplot_dataframe[c[0]]
        mean = np.mean([data1, data2], axis=0)
        diff = data1 - data2  # Difference between data1 and data2
        md = np.mean(diff)  # Mean of the difference
        sd = np.std(diff, axis=0)  # Standard deviation of the difference

        plt.scatter(mean, diff)
        plt.axhline(md, color='gray', linestyle='--')
        plt.axhline(md + 1.96 * sd, color='gray', linestyle='--')
        plt.axhline(md - 1.96 * sd, color='gray', linestyle='--')
        plt.xlabel('mean')
        plt.ylabel('differemce')
        plt.title("bland_altman_{}-{}.png".format(c[1],c[0]))

        outFile = outdir + r"\bland_altman_{}-{}.png".format(c[1],c[0])
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(8, 6)
        plt.savefig(outFile)
        plt.close()

def radius_of_curvature_42_points(algo, analysis_freq):
    '''min thickness at a measurement point vs velocity'''
    outdir = r"T:\Project data and results (svn)\Cortical Thickness Mohit\Results\Radial"

    fileclass = file_listing(algo)
    velocity_collect = []
    uncert_collect = []
    iW_collect = []
    for phantom in (0, 1, 2, 3, 4, 5, 6):
        point = 0
        for file in fileclass[phantom]:
            point = point + 1
            print("Analyzing file: {}".format(file))
            result = pickle.load(open(file, "rb")).T
            frequency = result.index[bsmdMeasurement.bsmdMeasurement.findNearestIndex(result.index, analysis_freq)]
            # print(result)
            uncert = result.ix[frequency][3]
            velocity = result.ix[frequency][0]
            iW = result.ix[frequency][5]
            velocity_collect.append(velocity)
            uncert_collect.append(uncert)
            iW_collect.append(iW)
            # print(frequency, velocity, uncert)

    hr = [14.6764848,11.3188058,13.1328494,12.5647,12.4372386,11.8833813,25.0012212,18.8365201,22.8578075,20.996275,19.9496646,
          18.8785632,35.2406292,26.185297,32.416527,29.1798625,27.545562,25.8737451,46.077336,32.6049182,40.0635026,37.528775,
          36.9778302,34.8916302,56.4874008,39.700289,49.040387,45.8776875,45.1580274,42.5610465,66.982794,46.7956598,58.1003907,
          54.3092625,53.421696,50.2304628,76.4542464,53.8910306,66.9941558,62.658175,61.6018932,57.8998791]
    hrinverse = [0.201574154,0.158638644,0.190088223,0.183880236,0.178504254,0.167300867,0.343379227,0.264003116,0.330849755,
                 0.307273552,0.286325616,0.265782938,0.484012357,0.366999847,0.469205106,0.427037653,0.395344992,0.364265009,
                 0.632849087,0.456974003,0.579889388,0.549221231,0.530720702,0.491223824,0.775826102,0.556419123,0.70982311,
                 0.671404809,0.648126185,0.599198142,0.919975061,0.655864243,0.840959921,0.794798125,0.766729682,0.707172461,
                 1.050060707,0.755309363,0.969690553,0.916981703,0.884135165,0.81514678]
    hrmeaninverse = [0.107634543,0.122710623,0.111267606,0.115064345,0.117184428,0.119948958,0.190569106,0.212887828,0.201983107,
                     0.2,0.194863433,0.197530864,0.279526227,0.30876494,0.299079755,0.289225727,0.279424217,0.281006865,
                     0.381895332,0.399585921,0.383147854,0.388034188,0.393952868,0.398460058,0.489283075,0.508658009,0.490033223,
                     0.495757034,0.50302185,0.508303976,0.607820364,0.628117914,0.608090474,0.614880674,0.623781676,0.628691983,
                     0.724919094,0.75952381,0.73540146,0.74459725,0.755373593,0.761218837]
    inner_rad_outer_rad = [0.897862233,0.884383089,0.894596398,0.891195419,0.889301634,0.886837881,0.826009501,0.807592752,
                           0.816544363,0.818181818,0.82243685,0.820224719,0.754750594,0.732528041,0.739826551,0.747315676,
                           0.754829123,0.753611557,0.679334917,0.666954271,0.678452302,0.675017895,0.670876672,0.667736758,
                          0.606888361,0.594477998,0.60640427,0.602720115,0.598068351,0.59470305,0.533847981,0.522001726,
                          0.533689126,0.529706514,0.524517088,0.521669342,0.467933492,0.449525453,0.462308205,0.457408733,0.451708767,0.448635634]

    '''print out the parameter vs veloctiy values'''
    for i in np.arange(len(hr)):
        print(hr[i],velocity_collect[i],iW_collect[i])
    # print(np.shape(local_thickness))
    # print(np.shape(velocity_collect))
    # print(np.shape(uncert_collect))

    plt.errorbar(hr, velocity_collect, yerr=uncert_collect, label="{}".format(algo.__class__.__name__), fmt='o')
    plt.legend()
    plt.grid(True)
    plt.xlabel('h*r-curv')
    plt.ylabel('phase velocity')
    plt.title("{}_h*r_42_points @{}.png".format(algo.__class__.__name__, analysis_freq))
    plt.ylim(100, 800)
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(8, 6)
    outFile = outdir + "\{}_hr_42_points @{} kx0.927.png".format(algo.__class__.__name__, analysis_freq)
    plt.savefig(outFile)
    # plt.show()
    plt.close()

    plt.errorbar(hrinverse, velocity_collect, yerr=uncert_collect, label="{}".format(algo.__class__.__name__), fmt='o')
    plt.legend()
    plt.grid(True)
    plt.xlabel('h/r-curv')
    plt.ylabel('phase velocity')
    plt.title("{}_h/r_42_points @{}.png".format(algo.__class__.__name__, analysis_freq))
    plt.ylim(100, 800)
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(8, 6)
    outFile = outdir + "\{}_hr_inverse_42_points @{}.png".format(algo.__class__.__name__, analysis_freq)
    plt.savefig(outFile)
    # plt.show()
    plt.close()

    plt.errorbar(hrmeaninverse, velocity_collect, yerr=uncert_collect, label="{}".format(algo.__class__.__name__), fmt='o')
    plt.legend()
    plt.grid(True)
    plt.xlabel('h/rmean')
    plt.ylabel('phase velocity')
    plt.title("{}_hrmean_42_points @{}.png".format(algo.__class__.__name__, analysis_freq))
    plt.ylim(100, 800)
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(8, 6)
    outFile = outdir + "\{}_hrmean_42_points @{} kx0.927.png".format(algo.__class__.__name__, analysis_freq)
    plt.savefig(outFile)
    # plt.show()
    plt.close()

    plt.errorbar(inner_rad_outer_rad, velocity_collect, yerr=uncert_collect, label="{}".format(algo.__class__.__name__), fmt='o')
    plt.legend()
    plt.grid(True)
    plt.xlabel('a/b')
    plt.ylabel('phase velocity')
    plt.title("{}_inner_outer_rad_42_points @{}.png".format(algo.__class__.__name__, analysis_freq))
    plt.ylim(100, 800)
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(8, 6)
    outFile = outdir + "\{}_inner_outer_rad_42_points @{} kx0.927.png".format(algo.__class__.__name__, analysis_freq)
    plt.savefig(outFile)
    # plt.show()
    plt.close()

    uncert_collect = []
    velocity_collect = []

#work from home
def local_thickness_42_points_exact_theory(algo, analysis_freq):
    '''min thickness at a measurement point vs velocity'''
    outdir = r"D:\ETH\Work\Latest Download\Work from home"

    fileclass = file_listing(algo)
    velocity_collect = []
    uncert_collect = []
    iW_collect = []
    for phantom in (0, 1, 2, 3, 4, 5, 6):
        point = 0
        for file in fileclass[phantom]:
            point = point + 1
            print("Analyzing file: {}".format(file))
            # result = pickle.load(open(file, "rb"), encoding= "latin1" ).T
            result = pickle.load(open(file, "rb")).T
            frequency = result.index[bsmdMeasurement.bsmdMeasurement.findNearestIndex(result.index, analysis_freq)]
            # print(result)
            uncert = result.ix[frequency][3]
            velocity = result.ix[frequency][0]
            iW = result.ix[frequency][5]
            velocity_collect.append(velocity)
            uncert_collect.append(uncert)
            iW_collect.append(iW)
            print(frequency, velocity, uncert)

    local_thickness = [1.72,1.34,1.58,1.52,1.49,1.41, 2.95,2.18,2.66,2.54,2.48,2.32, 4.18,3.02,3.74,3.53,3.45,3.23,5.4,3.86,4.82,4.54,4.43,4.14,6.62,4.7,5.9,5.55,5.41,5.05, 7.85,5.54,6.99,6.57,6.4,5.96, 8.96,6.38,8.06,7.58,7.38,6.87]

    '''print out the local thickness vs veloctiy values'''
    # for i in np.arange(len(local_thickness)):
    #     print(local_thickness[i],velocity_collect[i],iW_collect[i])
    # print(np.shape(local_thickness))
    # print(np.shape(velocity_collect))
    # print(np.shape(uncert_collect))



    plt.errorbar(local_thickness, velocity_collect, yerr=uncert_collect, label="{}".format(algo.__class__.__name__), fmt='o')
    plt.legend()
    plt.grid(True)
    plt.xlabel('local thickness')
    plt.ylabel('phase velocity')
    plt.title("{}_local thickness_42_points @{}.png".format(algo.__class__.__name__, analysis_freq))

    csv_file = open("cts_exact_tube_results_0.csv", mode="r")
    reader = csv.reader(csv_file)
    x = []
    for line in reader:
        t = float(line[4])
        x.append(t)
    x = np.array(x)
    plt.plot(local_thickness, x, 'ro')

    plt.ylim(100, 1000)
    plt.xlim(1, 9)

    figure = plt.gcf()  # get current figure
    figure.set_size_inches(8, 6)
    outFile = outdir + "\{}_local thickness_42_points @{} with exact theory.png".format(algo.__class__.__name__, analysis_freq)

    plt.savefig(outFile)
    plt.show()
    plt.close()
    uncert_collect = []
    velocity_collect = []

def one_file_velocity_run(algo,analysis_freq):
    logging.basicConfig(level=logging.DEBUG)
    topDir = r"D:\ETH\Work\Latest Download\Work from home\Radial\Results"
    filelist = mopa_data_review.find_files("*_{}_*.p".format(algo.__class__.__name__), topDir)
    # print(filelist)

    for file in filelist:
        print("Analyzing file: {}".format(file))
        # result = pickle.load(open(file, "rb"), encoding= "latin1" ).T
        result = pickle.load(open(file, "rb")).T
        frequency = result.index[bsmdMeasurement.bsmdMeasurement.findNearestIndex(result.index, analysis_freq)]
        # print(result)
        uncert = result.ix[frequency][3]
        velocity = result.ix[frequency][0]
        iW = result.ix[frequency][5]
        print(file, frequency, velocity, iW, uncert)

def local_thickness_42_points_dualpeak_corrected(algo, analysis_freq):
    '''min thickness at a measurement point vs velocity'''
    outdir = r"D:\ETH\Work\Latest Download\Work from home\Radial"

    fileclass = file_listing(algo)
    velocity_collect = []
    uncert_collect = []
    iW_collect = []
    for phantom in (0, 1, 2, 3, 4, 5, 6):
        point = 0
        for file in fileclass[phantom]:
            point = point + 1
            print("Analyzing file: {}".format(file))
            # result = pickle.load(open(file, "rb"), encoding= "latin1" ).T
            result = pickle.load(open(file, "rb")).T
            frequency = result.index[bsmdMeasurement.bsmdMeasurement.findNearestIndex(result.index, analysis_freq)]
            # print(result)
            uncert = result.ix[frequency][3]
            velocity = result.ix[frequency][0]
            iW = result.ix[frequency][5]
            velocity_collect.append(velocity)
            uncert_collect.append(uncert)
            iW_collect.append(iW)
            print(frequency, velocity, uncert)

    # local_thickness = [1.72,1.34,1.58,1.52,1.49,1.41, 2.95,2.18,2.66,2.54,2.48,2.32, 4.18,3.02,3.74,3.53,3.45,3.23,5.4,3.86,4.82,4.54,4.43,4.14,6.62,4.7,5.9,5.55,5.41,5.05, 7.85,5.54,6.99,6.57,6.4,5.96, 8.96,6.38,8.06,7.58,7.38,6.87]
    local_thickness = [1.72,1.34,1.58, 2.95,2.18,2.66, 2.32, 4.18,3.02,3.74,3.53,3.45,3.23,5.4,3.86,4.82,4.54,4.43,4.14,6.62,4.7,5.9,5.55,5.41,5.05, 7.85,5.54,6.99,6.57,6.4,5.96, 8.96,6.38,8.06,7.58,7.38,6.87]

    '''print out the local thickness vs veloctiy values'''
    # for i in np.arange(len(local_thickness)):
    #     print(local_thickness[i],velocity_collect[i],iW_collect[i])
    # print(np.shape(local_thickness))
    # print(np.shape(velocity_collect))
    # print(np.shape(uncert_collect))



    plt.errorbar(local_thickness, velocity_collect, yerr=uncert_collect, label="{}".format(algo.__class__.__name__), fmt='o')
    plt.legend()
    plt.grid(True)
    plt.xlabel('local thickness')
    plt.ylabel('phase velocity')
    plt.title("{}_local thickness_42_points.png".format(algo.__class__.__name__, analysis_freq))

    local_thickness_dual_peak = [1.52, 1.49, 1.41, 2.54, 2.39]
    '''following values are calculated using excel sheet with name dual peak camparisons'''
    velocity_average = [343.927756, 339.157214, 250.7759862, 409.3519692, 410.9038323]
    velocity_upper = [282.5039229, 280.7885204, 213.4924219, 332.3717876, 320.333089]
    velocity_lower = [439.4829963, 428.1608058, 303.8370112, 532.7389099, 572.8793094]
    # plt.plot(local_thickness_dual_peak, velocity_average, 'ro', label = 'average of peaks')
    plt.plot(local_thickness_dual_peak, velocity_upper, 'go', label = 'upper peak')
    # plt.plot(local_thickness_dual_peak, velocity_lower, 'co', label = 'lower peak')
    plt.legend()

    # csv_file = open(r"D:\downloads-D\ETH\Work\Latest Download\Work from home\Results\cts_exact_tube_results_0.csv", "r")
    # reader = csv.reader(csv_file)
    # x = []
    # thickness = []
    # for line in reader:
    #     t = float(line[4])
    #     x.append(t)
    #     m = float(line[2])
    #     thickness.append(m)
    # x = np.array(x)
    # thickness = np.array(thickness)*1000
    # plt.plot(thickness, x, 'mo', label= 'exact tube theory')
    # plt.legend()

    plt.ylim(100, 600)
    plt.xlim(1, 9)

    figure = plt.gcf()  # get current figure
    figure.set_size_inches(8, 6)
    outFile = outdir + "\{}_local thickness_42_points dual peaks corrected only upper.png".format(algo.__class__.__name__, analysis_freq)

    plt.savefig(outFile)
    plt.show()
    plt.close()
    uncert_collect = []
    velocity_collect = []

def local_thickness_vs_velocity_colorcoded(algo, analysis_freq):
    '''min thickness at a measurement point vs velocity'''
    outdir = r"D:\ETH\Work\Latest Download\Work from home\Radial"

    fileclass = file_listing(algo)
    velocity_collect = [[],[],[],[],[],[],[]]
    uncert_collect = [[],[],[],[],[],[],[]]
    for phantom in (0, 1, 2, 3, 4, 5, 6):
        point = 0
        for file in fileclass[phantom]:
            point = point + 1
            print("Analyzing file: {}".format(file))
            result = pickle.load(open(file, "rb")).T
            frequency = result.index[bsmdMeasurement.bsmdMeasurement.findNearestIndex(result.index, analysis_freq)]
            # print(result)
            uncert = result.ix[frequency][3]
            velocity = result.ix[frequency][0]
            velocity_collect[phantom].append(velocity)
            uncert_collect[phantom].append(uncert)
            # print(frequency, velocity, uncert)
    print(velocity_collect,uncert_collect)

    local_thickness = [[1.72,1.34,1.58,1.52,1.49,1.41], [2.95,2.18,2.66,2.54,2.48,2.32], [4.18,3.02,3.74,3.53,3.45,3.23],[5.4,3.86,4.82,4.54,4.43,4.14],[6.62,4.7,5.9,5.55,5.41,5.05], [7.85,5.54,6.99,6.57,6.4,5.96], [8.96,6.38,8.06,7.58,7.38,6.87]]

    for phantom in (0, 1, 2, 3, 4, 5, 6):
        plt.errorbar(local_thickness[phantom], velocity_collect[phantom], yerr=uncert_collect[phantom], label="phantom {}".format(phantom + 1), fmt='o')
        plt.legend()
        plt.grid(True)
    plt.xlabel('local thickness')
    plt.ylabel('phase velocity')
    plt.title("colorcoded_42points_{}_local thickness_vs_velocity".format( algo.__class__.__name__))

    figure = plt.gcf()  # get current figure
    figure.set_size_inches(8, 6)
    outFile = outdir + "\colorcoded_42points_{}_local thickness_vs_velocity".format( algo.__class__.__name__)
    plt.savefig(outFile)
    plt.show()
    plt.close()
    # uncert_collect = []
    # velocity_collect = []

def local_thickness_vs_velocity_color_number_coded(algo, analysis_freq):
    '''min thickness at a measurement point vs velocity'''
    outdir = r"D:\ETH\Work\Latest Download\Work from home\Radial"

    fileclass = file_listing(algo)
    velocity_collect = [[],[],[],[],[],[],[]]
    uncert_collect = [[],[],[],[],[],[],[]]
    for phantom in (0, 1, 2, 3, 4, 5, 6):
        point = 0
        for file in fileclass[phantom]:
            point = point + 1
            print("Analyzing file: {}".format(file))
            result = pickle.load(open(file, "rb")).T
            frequency = result.index[bsmdMeasurement.bsmdMeasurement.findNearestIndex(result.index, analysis_freq)]
            # print(result)
            uncert = result.ix[frequency][3]
            velocity = result.ix[frequency][0]
            velocity_collect[phantom].append(velocity)
            uncert_collect[phantom].append(uncert)
            # print(frequency, velocity, uncert)
    print(velocity_collect,uncert_collect)

    local_thickness = [[1.72,1.34,1.58,1.52,1.49,1.41], [2.95,2.18,2.66,2.54,2.48,2.32], [4.18,3.02,3.74,3.53,3.45,3.23],[5.4,3.86,4.82,4.54,4.43,4.14],[6.62,4.7,5.9,5.55,5.41,5.05], [7.85,5.54,6.99,6.57,6.4,5.96], [8.96,6.38,8.06,7.58,7.38,6.87]]
    numberbullets = [1,2,3,4,5,6]

    fig, ax = plt.subplots()
    for phantom in (0, 1, 2, 3, 4, 5, 6):
        plt.errorbar(local_thickness[phantom], velocity_collect[phantom], yerr=uncert_collect[phantom], label="phantom {}".format(phantom + 1), fmt='o')
        # ax.scatter(local_thickness[phantom], velocity_collect[phantom])
        for txt in numberbullets:
            ax.annotate(txt, (local_thickness[phantom][txt - 1], velocity_collect[phantom][txt - 1]))
        plt.legend()
        plt.grid(True)
    plt.xlabel('local thickness')
    plt.ylabel('phase velocity')
    plt.title("colorcoded_42points_{}_local thickness_vs_velocity".format( algo.__class__.__name__))
    #
    # figure = plt.gcf()  # get current figure
    # figure.set_size_inches(8, 6)
    outFile = outdir + "\colorcoded_42points_{}_local thickness_vs_velocity".format( algo.__class__.__name__)
    # plt.savefig(outFile)
    plt.show()
    plt.close()
    # uncert_collect = []
    # velocity_collect = []

def local_thickness_vs_velocity_color_number_coded_with_doublepeaks(algo, analysis_freq):
    '''min thickness at a measurement point vs velocity'''
    outdir = r"D:\ETH\Work\Latest Download\Work from home\Radial"

    fileclass = file_listing(algo)
    velocity_collect = [[],[],[],[],[],[],[]]
    uncert_collect = [[],[],[],[],[],[],[]]
    for phantom in (0, 1, 2, 3, 4, 5, 6):
        point = 0
        for file in fileclass[phantom]:
            point = point + 1
            print("Analyzing file: {}".format(file))
            result = pickle.load(open(file, "rb")).T
            frequency = result.index[bsmdMeasurement.bsmdMeasurement.findNearestIndex(result.index, analysis_freq)]
            # print(result)
            uncert = result.ix[frequency][3]
            velocity = result.ix[frequency][0]
            velocity_collect[phantom].append(velocity)
            uncert_collect[phantom].append(uncert)


            # print(frequency, velocity, uncert)
    print(velocity_collect,uncert_collect)

    local_thickness = [[1.72,1.34,1.58], [2.95,2.18,2.66,2.32], [4.18,3.02,3.74,3.53,3.45,3.23],[5.4,3.86,4.82,4.54,4.43,4.14],[6.62,4.7,5.9,5.55,5.41,5.05], [7.85,5.54,6.99,6.57,6.4,5.96], [8.96,6.38,8.06,7.58,7.38,6.87]]
    colorlist = ['b','g','r','C0','C1','C2','C3']

    fig, ax = plt.subplots()
    plt.plot([0], [0], label='A : average of peaks \n + :upper peak \n - :lower peak', color='w')

    for phantom in (0, 1, 2, 3, 4, 5, 6):
        '''this block plots non-double peak points together and annotates the according to color and sensor point number'''
        plt.errorbar(local_thickness[phantom], velocity_collect[phantom], markersize=4 , yerr=uncert_collect[phantom], label="phantom {}".format(phantom + 1), color = '{}'.format(colorlist[phantom]), fmt='o')
        # ax.scatter(local_thickness[phantom], velocity_collect[phantom])

        if phantom == 0:
            numberbullets = [1,2,3]
            for txt in numberbullets:
                ax.annotate(txt, (local_thickness[phantom][txt - 1], velocity_collect[phantom][txt - 1]), weight='bold', size = 8, ha='right', va='bottom')

        elif phantom == 1:
            numberbullets = [1,2,3,6]
            for i, txt in enumerate(numberbullets):
                ax.annotate(txt, (local_thickness[phantom][i], velocity_collect[phantom][i]), weight='bold', size = 8, ha='right', va='top')

        else:
            numberbullets = [1, 2, 3, 4, 5, 6]
            for txt in numberbullets:
                ax.annotate(txt, (local_thickness[phantom][txt-1], velocity_collect[phantom][txt-1]), weight='bold', size = 8, ha='left', va='bottom')

        plt.legend()
        plt.grid(True)

    # local_thickness_dual_peak = [1.52, 1.49, 1.41, 2.54, 2.39]
    # numberbullets_doublepeak = [4, 5, 6, 4, 5]
    # '''following values are calculated using excel sheet with name dual peak camparisons'''
    # velocity_average = [343.927756, 339.157214, 250.7759862, 409.3519692, 410.9038323]
    # velocity_upper = [282.5039229, 280.7885204, 213.4924219, 332.3717876, 320.333089]
    # velocity_lower = [439.4829963, 428.1608058, 303.8370112, 532.7389099, 572.8793094]
    # plt.plot(local_thickness_dual_peak, velocity_average, 'k^', label = 'average of peaks')
    # plt.plot(local_thickness_dual_peak, velocity_upper, 'k+', label='upper peak')
    # plt.plot(local_thickness_dual_peak, velocity_lower, 'k*', label = 'lower peak')
    # for i, txt in enumerate(numberbullets_doublepeak):
    #     ax.annotate(txt, (local_thickness_dual_peak[i], velocity_average[i]),weight='bold', size = 6, ha='right', va='bottom')
    #     ax.annotate(txt, (local_thickness_dual_peak[i], velocity_upper[i]), weight='bold', size = 6, ha='right', va='bottom')
    #     ax.annotate(txt, (local_thickness_dual_peak[i], velocity_lower[i]), weight='bold', size = 6, ha='right', va='bottom')
    # plt.legend()

    '''next two blocks plot the double peaks in phantom 1 and 2 separately'''
    local_thickness_dual_peak_phantom1 = [1.52, 1.49, 1.41]
    numberbullets_doublepeak_phantom1_avg = ['4A','5A','6A']
    numberbullets_doublepeak_phantom1_upper = ['4+', '5+', '6+']
    numberbullets_doublepeak_phantom1_lower = ['4-', '5-', '6-']
    halignment = ['left','right','right']
    valignment = ['bottom', 'top', 'bottom']
    '''following values are calculated using excel sheet with name dual peak camparisons'''
    velocity_average_phantom1 = [343.927756, 339.157214, 250.7759862]
    velocity_upper_phantom1 = [282.5039229, 280.7885204, 213.4924219]
    velocity_lower_phantom1 = [439.4829963, 428.1608058, 303.8370112]
    uncert_average_phantom1 = [2.9020912417156643, 1.1316619997766482, 2.6857909124603343]
    uncert_upper_phantom1 = [0.6473537652761899, 0.6282769700027193, 0.5491298254609284]
    uncert_lower_phantom1 = [1.6214739628056907, 2.7788655847303465, 1.634908199415281]


    plt.errorbar(local_thickness_dual_peak_phantom1, velocity_average_phantom1, markersize=4, yerr = uncert_average_phantom1, fmt = 'bo')
    plt.errorbar(local_thickness_dual_peak_phantom1, velocity_upper_phantom1, markersize=4, yerr = uncert_upper_phantom1,fmt = 'bo')
    plt.errorbar(local_thickness_dual_peak_phantom1, velocity_lower_phantom1, markersize=4, yerr = uncert_lower_phantom1,fmt = 'bo')
    for i in [0,1,2]:
        ax.annotate(numberbullets_doublepeak_phantom1_avg[i], (local_thickness_dual_peak_phantom1[i], velocity_average_phantom1[i]),weight='bold', size = 8, ha='right', va='{}'.format(valignment[i]))
        ax.annotate(numberbullets_doublepeak_phantom1_upper[i], (local_thickness_dual_peak_phantom1[i], velocity_upper_phantom1[i]),weight='bold', size = 8, ha='{}'.format(halignment[i]), va='bottom')
        ax.annotate(numberbullets_doublepeak_phantom1_lower[i], (local_thickness_dual_peak_phantom1[i], velocity_lower_phantom1[i]),weight='bold', size = 8, ha='right', va='bottom')
    plt.legend()

    local_thickness_dual_peak_phantom2 = [2.54, 2.39]
    numberbullets_doublepeak_phantom2_avg = ['4A', '5A']
    numberbullets_doublepeak_phantom2_upper = ['4+', '5+']
    numberbullets_doublepeak_phantom2_lower = ['4-', '5-']
    '''following values are calculated using excel sheet with name dual peak camparisons'''
    velocity_average_phantom2 = [409.3519692, 410.9038323]
    velocity_upper_phantom2 = [332.3717876, 320.333089]
    velocity_lower_phantom2 = [532.7389099, 572.8793094]
    uncert_average_phantom2 = [7.221188831750675, 9.635924129915468]
    uncert_upper_phantom2 = [0.9843290170379789, 0.6534535584507304]
    uncert_lower_phantom2 = [2.3594564977690227,1.5428246870063909]
    plt.errorbar(local_thickness_dual_peak_phantom2, velocity_average_phantom2, markersize=4, yerr = uncert_average_phantom2, fmt = 'go')
    plt.errorbar(local_thickness_dual_peak_phantom2, velocity_upper_phantom2, markersize=4, yerr = uncert_upper_phantom2, fmt = 'go')
    plt.errorbar(local_thickness_dual_peak_phantom2, velocity_lower_phantom2, markersize=4, yerr = uncert_lower_phantom2, fmt = 'go')
    for i in [0,1]:
        ax.annotate(numberbullets_doublepeak_phantom2_avg[i], (local_thickness_dual_peak_phantom2[i], velocity_average_phantom2[i]),weight='bold', size = 8, ha='{}'.format(halignment[i]), va='bottom')
        ax.annotate(numberbullets_doublepeak_phantom2_upper[i], (local_thickness_dual_peak_phantom2[i], velocity_upper_phantom2[i]),weight='bold', size = 8, ha='right', va='bottom')
        ax.annotate(numberbullets_doublepeak_phantom2_lower[i], (local_thickness_dual_peak_phantom2[i], velocity_lower_phantom2[i]),weight='bold', size = 8, ha='right', va='bottom')
    plt.legend()

    plt.xlabel('Local thickness (mm)')
    plt.ylabel('Phase velocity (m/s)')
    # plt.title("colorcoded_42points_{}_local thickness_vs_velocity_with_double_peaks_azimuthal_uncert".format( algo.__class__.__name__))
    #

    plt.xlim(1,9.2)
    plt.ylim(180,600)
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(6.5, 4)
    outFile = outdir + "\colorcoded_42points_{}_local thickness_vs_velocity_number_coded_with_doublepeaks_Azimuthal_uncert".format( algo.__class__.__name__)
    # plt.savefig(outFile)
    plt.savefig(outdir + r"\results azimuthal legends transparent.eps", dpi=300,  edgecolor='w',
                orientation='portrait', transparent=True)
    plt.savefig(outdir + r"\results azimuthal legends transparent.png", dpi=300,  edgecolor='w',
                orientation='portrait', transparent=False)
    plt.show()
    plt.close()
    # uncert_collect = []
    # velocity_collect = []

def theory_plot_final():

    outdir = r"D:\ETH\Work\Latest Download\Work from home\Radial"
    csv_file = open(r"D:\ETH\Work\Latest Download\Work from home\Radial\Results\cts_exact_tube_results_0.csv", mode="r")
    reader = csv.reader(csv_file)
    x = []
    for line in reader:
        t = float(line[4])
        x.append(t)
    x = np.array(x)

    y = [[],[],[],[],[],[],[]]
    j = 0
    i = 5
    for k in range(len(x)):
        y[j].append(x[k])
        if k == i:
            j = j+1
            i = i+6
    print(x)
    print(y)

    # local_thickness = [1.72,1.34,1.58,1.52,1.49,1.41, 2.95,2.18,2.66,2.54,2.48,2.32, 4.18,3.02,3.74,3.53,3.45,3.23,5.4,3.86,4.82,4.54,4.43,4.14,6.62,4.7,5.9,5.55,5.41,5.05, 7.85,5.54,6.99,6.57,6.4,5.96, 8.96,6.38,8.06,7.58,7.38,6.87]
    # plt.plot(local_thickness, x, 'ro')

    fig, ax = plt.subplots()
    colorlist = ['b', 'g', 'r', 'C0', 'C1', 'C2', 'C3']
    local_thickness = [[1.72,1.34,1.58,1.52,1.49,1.41], [2.95,2.18,2.66,2.54,2.48,2.32], [4.18,3.02,3.74,3.53,3.45,3.23],[5.4,3.86,4.82,4.54,4.43,4.14],[6.62,4.7,5.9,5.55,5.41,5.05],[7.85,5.54,6.99,6.57,6.4,5.96], [8.96,6.38,8.06,7.58,7.38,6.87]]
    for phantom in (0, 1, 2, 3, 4, 5, 6):
        plt.plot(local_thickness[phantom], y[phantom], 'o', label="phantom {}".format(phantom + 1), color = '{}'.format(colorlist[phantom]))

        numberbullets = [1, 2, 3, 4, 5, 6]
        for txt in numberbullets:
            ax.annotate(txt, (local_thickness[phantom][txt - 1], y[phantom][txt - 1]), weight='bold', size=8,
                        ha='left', va='bottom')

    plt.ylim(350, 520)
    plt.xlim(1, 9.5)
    # plt.legend()
    plt.grid(True)
    plt.xlabel('Local thickness (mm)')
    plt.ylabel('Phase velocity (m/s)')
    # plt.title("Exact theory plot")
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(6, 4)
    # outFile = outdir + "\exact theory final plot.jpg"
    # plt.savefig(outFile)
    # plt.savefig(outdir + r"\Exact theory plot.eps", dpi=300, facecolor='w', edgecolor='w',
    #             orientation='portrait', )
    plt.savefig(outdir + r"\Exact theory plot only legend.eps", dpi=300,  edgecolor='w',
                orientation='portrait', transparent=False)
    plt.savefig(outdir + r"\Exact theory only legend.png", dpi=300,  edgecolor='w',
                orientation='portrait', transparent=False)
    plt.show()
    plt.close()

def scatterplot_final_functions(freq_range):

    outdir = r"D:\ETH\Work\Latest Download\Work from home\Radial"
    scatterplot_dataframe = pd.DataFrame()

    algo_list = (bsmdAlgorithm.ampConst_sigmafixed(), bsmdAlgorithm.ampConst_sigmafree(),
                 bsmdAlgorithm.ampgaussian_sigmafixed(), bsmdAlgorithm.ampgaussian_sigmafree(),
                 bsmdAlgorithm.ampsemigaussian_sigmafixed(), bsmdAlgorithm.ampsemigaussian_sigmafree())
    for algo in algo_list:
        fileclass = file_listing(algo)
        velocity_collect = []
        uncert_collect = []
        for phantom in (0, 1, 2, 3, 4, 5, 6):
            point = 0
            for file in fileclass[phantom]:
                point = point + 1
                print("Analyzing file: {}".format(file))
                result = pickle.load(open(file, "rb")).T
                frequency0 = result.index[bsmdMeasurement.bsmdMeasurement.findNearestIndex(result.index, freq_range[0])]
                frequency1 = result.index[bsmdMeasurement.bsmdMeasurement.findNearestIndex(result.index, freq_range[1])]
                velocity = (result["velocity"][frequency0:frequency1].values)
                velocity_collect.append(velocity)
        flat_list = [item for sublist in velocity_collect for item in sublist]
        scatterplot_dataframe["{}".format(algo.__class__.__name__)] = flat_list
    corr_coeff_mat = pd.DataFrame.corr(scatterplot_dataframe, method='pearson')
    print(corr_coeff_mat)
    scatterplot_matrix = scatter_matrix(scatterplot_dataframe, alpha=1, figsize=(6, 6), diagonal='kde')  #
    outFile = outdir + r"\scatterplot_final_functions.png"
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(32, 24)
    matplotlib.rc('xtick', labelsize=8)
    matplotlib.rc('ytick', labelsize=8)
    # plt.savefig(outFile)
    # plt.show()
    plt.close()
    # print(scatterplot_matrix)

    '''correlatin coefficient matrix plot'''
    sns.set(font_scale=1.7)
    sns.heatmap(corr_coeff_mat, annot=True, annot_kws={"size": 15})
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(6.5, 6.5)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    # plt.savefig(outdir + r"\correlation coefficient matrix _final_functions.png")
    # savefig(fname, dpi=None, facecolor='w', edgecolor='w',
    #         orientation='portrait', papertype=None, format=None,
    #         transparent=False, bbox_inches=None, pad_inches=0.1,
    #         frameon=None, metadata=None)
    plt.savefig(outdir + r"\correlation coefficient matrix _final_functions.eps", dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait')
    plt.show()
    plt.close()

def radial_axial_correlation(analysis_freq):

    outdir = r"D:\ETH\Work\Latest Download\Work from home\Radial"

    scatterplot_dataframe = pd.DataFrame()

    topDir_radial = r"D:\ETH\Work\Latest Download\Work from home\Radial\Pickles 42 final"
    topDir_axial = r"D:\ETH\Work\Latest Download\Work from home\Axial\Pickles 42 final"

    filelist_radial = mopa_data_review.find_files("*.p", topDir_radial)
    filelist_axial = mopa_data_review.find_files("*.p", topDir_axial)
    fileclass = [filelist_radial, filelist_axial]
    filenames = ['Radial','Axial']
    i = 0
    for filelist in fileclass:
        velocity_collect = []
        for file in filelist:
            print("Analyzing file: {}".format(file))
            result = pickle.load(open(file, "rb")).T
            frequency = result.index[bsmdMeasurement.bsmdMeasurement.findNearestIndex(result.index, analysis_freq)]
            velocity = result.ix[frequency][0]
            velocity_collect.append(velocity)
        # flat_list = [item for sublist in velocity_collect for item in sublist]
        scatterplot_dataframe["{}".format(filenames[i])] = velocity_collect
        i = i + 1
    corr_coeff_mat = pd.DataFrame.corr(scatterplot_dataframe, method='pearson')
    print(corr_coeff_mat)
    scatterplot_matrix = scatter_matrix(scatterplot_dataframe, alpha= 1, figsize=(6, 6), diagonal='kde') #
    outFile = outdir + r"\scatterplot_radial and axial.png"
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(6, 6)
    matplotlib.rc('xtick', labelsize=8)
    matplotlib.rc('ytick', labelsize=8)
    # plt.savefig(outFile)
    plt.savefig(outdir + r"\scatterplot radial axial.eps", dpi=300, facecolor='w', edgecolor='w',
                orientation='portrait')
    plt.show()
    plt.close()
    print(scatterplot_matrix)

    '''correlatin coefficient matrix plot'''
    sns.set(font_scale=1.5)
    sns.heatmap(corr_coeff_mat, annot=True, annot_kws={"size": 15})
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(7, 7)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    # plt.savefig(outdir + r"\correlation coefficient matrix _final_functions.png")
    # savefig(fname, dpi=None, facecolor='w', edgecolor='w',
    #         orientation='portrait', papertype=None, format=None,
    #         transparent=False, bbox_inches=None, pad_inches=0.1,
    #         frameon=None, metadata=None)
    plt.savefig(outdir + r"\correlation radial axial.eps", dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait')
    plt.show()
    plt.close()


if __name__ == "__main__":
    '''list of all algorithms'''
    # algo = bsmdAlgorithm.PhaseVelocity_2DFFTGauss()
    # algo = bsmdAlgorithm.PhaseVelocity_2DFFTSinc()
    # algo = bsmdAlgorithm.PhaseVelocity_2DGaussianfit()
    # algo = bsmdAlgorithm.PhaseVelocity_2DFFTMax()
    # algo = bsmdAlgorithm.PhaseVelocity_1D_functions()
    # algo = bsmdAlgorithm.PhaseVelocity_1Dlinear()
    # algo = bsmdAlgorithm.ampConst_sigmafixed()
    # algo = bsmdAlgorithm.ampConst_sigmafree()
    # algo = bsmdAlgorithm.ampgaussian_sigmafixed()
    algo = bsmdAlgorithm.ampgaussian_sigmafree()
    # algo = bsmdAlgorithm.ampsemigaussian_sigmafixed()
    # algo = bsmdAlgorithm.ampsemigaussian_sigmafree()

    # algo_list = (bsmdAlgorithm.ampgaussian_sigmafree(), bsmdAlgorithm.ampsemigaussian_sigmafixed(), bsmdAlgorithm.PhaseVelocity_2DGaussianfit())




    # for algo in algo_list:

    '''saving the result dataframe to a pickle for future use. Also contains plotting function saving dispersion relation overlap with 2Dfft'''
    # saving_dataframes()

    '''for plotting freq vs velocity with errorbars'''
    # frequency_vs_velocity(algo)

    '''velocity vs sonsor point plots for each phantom'''
    # sensorpoint_vs_velocity(algo,3000)
    # sensorpoint_vs_velocity_all(3000)

    '''avg velocity vs. avg thickness of phantom (one velocity per phantom)'''
    # avg_velocity_vs_avg_thickness(algo,3000)
    # avg_velocity_vs_avg_thickness_algoComparison(3000)
    # mode_velocity_vs_avg_thickness_algoComparison(0,3290)

    '''average_ss values listing'''
    # average_ss_listings(algo)
    # average_ss_check()

    '''local_thickness_vs_velocity for analysis frequency'''
    # local_thickness_vs_velocity(algo,3000)
    # local_thickness_42_points(algo, 3000)
    # local_thickness_42_points(algo, 3290)
    # local_thickness_42_points(algo, 3150)
    # selective_thickness_42_points(algo,3290)
    # each_phantom_separate(algo,3000,2)
    # radius_of_curvature_42_points(algo,3000)
    # local_thickness_42_points_exact_theory(algo, 3000)
    # local_thickness_42_points_dualpeak_corrected(algo, 3000)



    '''scatter matrix plot algorithm comparison'''
    # scatterplot_algo_comparison((2900,3000))

    '''scatter matrix plot algorithm comparison'''
    # scatterplot_algo_comparison((2900,3000))

    '''cross area of each phantom vs avg velocity of each phantom for analysis frequency and all algorithms'''
    # cross_area_vs_velocity(algo,3000)
    # Algo_comparison_cross_area_vs_velocity(3000)


    '''scatter matrix plot algorithm comparison'''
    # scatterplot_algo_comparison((2900,3000))

    '''bland-altman plot for the algorithm pairs'''
    # bland_altman_plot((2900,3000))

    '''velocity vs thickness-lamda plot'''
    # velocity_vs_thickness_lambda(algo, 3000)

    '''dual peak rerun files upper lower middle'''
    # one_file_velocity_run(algo, 3000)

    # local_thickness_vs_velocity_colorcoded(algo, 3000)
    # local_thickness_vs_velocity_color_number_coded(algo, 3000)
    # local_thickness_vs_velocity_color_number_coded_with_doublepeaks(algo, 3000)
    theory_plot_final()

    # scatterplot_final_functions((2900,3100))
    # radial_axial_correlation(3000)