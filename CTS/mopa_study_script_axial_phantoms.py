import sys
sys.path.append("D:\ETH\Work\Latest Download\ETH\python")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import bsmd.algorithm.bsmdAlgorithmRestructured as bsmdAlgorithm
# import bsmd.algorithm.bsmdAlgorithm as bsmdAlgorithm
import bsmd.measurement.bsmdMeasurement as bsmdMeasurement
import matplotlib.pyplot as plt
import CTS.mopa_data_review as mopa_data_review
import os
import numpy as np
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
import csv
import pandas.core.indexes
import sys
sys.modules['pandas.indexes'] = pandas.core.indexes
sys.path.append("D:\ETH\Work\Latest Download\ETH\python")
####################################
##      functions with verified input values for Axial Phantom study
####################################
def saving_dataframes():
    '''function finds the .xml data files from topdir and runs analysis. saves the dataframe of results as pickles to outdir.
    dispersion_plot=True and outfile=file saves the plot of discpersion relation plotted over 2Dfft plot. check protocol.info and parameter record.xlsx for details'''
    logging.basicConfig(level=logging.DEBUG)
    meas = bsmdMeasurement.bsmdMeasurement(xFFTNr=25000, yFFTNr=3500)
    topDir = r"D:\ETH\Work\Latest Download\ETH\project data and results\Cortical Thickness Mohit\XML data\Phantom_axial_1.5-7.5mm"
    outdir = r"D:\ETH\Work\Latest Download\Work from home\Axial\Uncert Rerun"

    filelist = mopa_data_review.find_files("point1_a25mm.xml", topDir)
    parameter_list = pd.read_excel(r'D:\ETH\Work\Latest Download\Work from home\Axial\Uncert Rerun\parameter record uncert rerun.xlsx')
    parameter_list.index = parameter_list['filename']



    for file in filelist:
        print("Analyzing file: {}".format(file))
        meas.readStepPhantomXMLData(file)
        freq1 = parameter_list['freq1'][os.path.basename(file)]
        freq2 = parameter_list['freq2'][os.path.basename(file)]
        iW1 = parameter_list['iW1'][os.path.basename(file)]
        iW2 = parameter_list['iW2'][os.path.basename(file)]
        print('parameters for {}- frquency range = ({},{}), wavenumber range = ({},{})'.format(os.path.basename(file),freq1, freq2, iW1, iW2))


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


        '''ampConst_sigmafree'''
        # try:
        #     result = algo.phaseVelocity(meas, freq=(freq1, freq2), wavenumber=(iW1, iW2),  p0=(15000, 0.001, 1, 1), mode='linear',
        #                             dispersion_plot=True, showImages=None, window= True,
        #                             sliced_plot=None, debug_plot=None, outdir=None, sumOfSquares_threshold=1,
        #                             normalize=True,outFile = outdir + "\{}_{}_linear_dispersion overlap.png".format(os.path.basename(file),
        #                                                                      algo.__class__.__name__),
        #                             title="{}_{}_linear_dispersion overlap.png".format(os.path.basename(file),
        #                                                                                algo.__class__.__name__))
        #
        #     pickle.dump(result, open(outdir + "\{}_{}_basicData.p".format(os.path.basename(file), algo.__class__.__name__), "wb"))
        # except:
        #     pass


        '''ampsemigaussian_sigmafree'''
        # try:
        # result = algo.phaseVelocity(meas, freq=(freq1, freq2), wavenumber=(iW1, iW2), p0=(15000, 0.001, 1, 1),
        #                             mode='linear',
        #                             dispersion_plot=True, showImages=True, window=True,
        #                             sliced_plot=None, debug_plot=None, outdir=None, sumOfSquares_threshold=1,
        #                             normalize=None,outFile = outdir + "\{}_{}_linear_dispersion overlap.png".format(os.path.basename(file), algo.__class__.__name__),
        #                             title="{}_{}_linear_dispersion overlap.png".format(os.path.basename(file),
        #                                                                                algo.__class__.__name__))

        # pickle.dump(result,
        #                 open(outdir + "\{}_{}_basicData.p".format(os.path.basename(file), algo.__class__.__name__),
        #                      "wb"))
        # except:
        #     pass

        '''ampgaussian_sigmafree'''
        try:
            result = algo.phaseVelocity(meas, freq=(freq1, freq2), wavenumber=(2, 9),
                                        p0=(15000, 0.005, 3, (freq1 + freq2) / 2, 550, 5), mode='linear',
                                        dispersion_plot=True, showImages=None, window=True,
                                        sliced_plot=None, debug_plot=None, outdir=None, sumOfSquares_threshold=None,
                                        normalize=None, outFile=outdir + "\{}_{}_linear_dispersion overlap dual upper.png".format(os.path.basename(file),algo.__class__.__name__),
                                        title="{}_{}_linear_dispersion overlap dual upper.png".format(os.path.basename(file),
                                                                                           algo.__class__.__name__))

            pickle.dump(result,
                        open(outdir + "\{}_{}_dual upper_basicData.p".format(os.path.basename(file), algo.__class__.__name__),
                             "wb"))
        except:
            pass

        '''ampsemigaussian_sigmafixed'''
        # try:
        #     result = algo.phaseVelocity(meas, freq=(freq1, freq2), wavenumber=(iW1, iW2), p0=(15000, 0.001, 1),
        #                                 mode='linear',
        #                                 dispersion_plot=True, showImages=None, window=True,
        #                                 sliced_plot=None, debug_plot=None, outdir=None, sumOfSquares_threshold=None,
        #                                 normalize=None, outFile=outdir + "\{}_{}_linear_dispersion overlap.png".format(
        #             os.path.basename(file),
        #             algo.__class__.__name__),
        #                                 title="{}_{}_linear_dispersion overlap.png".format(os.path.basename(file),
        #                                                                                    algo.__class__.__name__))
        #
        #     pickle.dump(result,
        #                 open(outdir + "\{}_{}_basicData.p".format(os.path.basename(file), algo.__class__.__name__),
        #                      "wb"))
        # except:
        #     pass

def file_listing(algo):
    logging.basicConfig(level=logging.DEBUG)
    meas = bsmdMeasurement.bsmdMeasurement(xFFTNr=25000, yFFTNr=3500)
    topDir = r"D:\ETH\Work\Latest Download\Work from home\Axial\Results Uncert Pickles"


    filelist = []
    fileclass = []
    for phantom in (15,25,35,45,55,65,75):
        for point in (1,2,3,4,5,6):
            file = mopa_data_review.find_files("point{}_a{}mm.xml_{}_dual upper_basicData.p".format(point, phantom , algo.__class__.__name__), topDir)
            filelist = filelist + file
        fileclass.append(filelist)
        filelist = [] #resets the filelist so that each phantom has 6 element string in fileclass, otherwise the list goes on adding filenames for next phantoms
    return(fileclass)

def frequency_vs_velocity(algo):

    topDir = r"T:\Project data and results (svn)\Cortical Thickness Mohit\Results\Axial\Result dataframe pickles"
    outdir = r"T:\Project data and results (svn)\Cortical Thickness Mohit\Results\Axial\frequency vs velocity"

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
            plt.title("phantom {}_{}_frequency_vs_velocity.png".format(phantom+1, algo.__class__.__name__))
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(8, 6)

        outFile = outdir + "\phantom {}_{}_frequency_vs_velocity.png".format(phantom+1, algo.__class__.__name__)
        plt.savefig(outFile)
        # plt.show()
        plt.close()

def sensorpoint_vs_velocity(algo, analysis_freq):

    outdir = r"D:\downloads-D\ETH\Work\Latest Download\ETH\project data and results\Cortical Thickness Mohit\Results\Axial\Sensor point vs velocity\3000Hz"

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
        plt.title("phantom {}_{}_sensor point velocity @{}.png".format(phantom + 1, algo.__class__.__name__, analysis_freq))

        figure = plt.gcf()  # get current figure
        figure.set_size_inches(8, 6)
        outFile = outdir + "\phantom {}_{}_sensor point velocity @{}.png".format(phantom + 1, algo.__class__.__name__, analysis_freq)

        plt.savefig(outFile)
        plt.show()
        plt.close()
        uncert_collect = []
        velocity_collect = []

def average_ss_listings(algo):
    '''lists all the average_ss values for all the files and all the algorithms'''
    topDir = r"T:\Project data and results (svn)\Cortical Thickness Mohit\Results\Axial\Result dataframe pickles"
    outdir = r"T:\Project data and results (svn)\Cortical Thickness Mohit\Results\Axial\avg_ss plots"

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

    topDir = r"T:\Project data and results (svn)\Cortical Thickness Mohit\Results\Axial\Result Dataframe Pickles"
    outdir = r"T:\Project data and results (svn)\Cortical Thickness Mohit\Results\Radial\temp destination"

    algo_list = (bsmdAlgorithm.PhaseVelocity_2DFFTGauss(), bsmdAlgorithm.PhaseVelocity_2DGaussianfit(),
                  bsmdAlgorithm.ampConst_sigmafixed(),
                 bsmdAlgorithm.ampConst_sigmafree(),
                 bsmdAlgorithm.ampgaussian_sigmafixed(), bsmdAlgorithm.ampgaussian_sigmafree(),
                 bsmdAlgorithm.ampsemigaussian_sigmafixed(), bsmdAlgorithm.ampsemigaussian_sigmafree())

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
    outdir = r"T:\Project data and results (svn)\Cortical Thickness Mohit\Results\Axial\local thickness vs velocity"

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
        plt.title("phantom {}_{}_local thickness_vs_velocity.png".format(phantom + 1, algo.__class__.__name__))



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
    outdir = r"D:\downloads-D\ETH\Work\Latest Download\ETH\project data and results\Cortical Thickness Mohit\Results\Axial\local thickness 42 points vs velocity\3000Hz"

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

    local_thickness = [1.72,1.34,1.58,1.52,1.49,1.41, 2.95,2.18,2.66,2.54,2.48,2.32, 4.18,3.02,3.74,3.53,3.45,3.23,5.4,3.86,4.82,4.54,4.43,4.14,6.62,4.7,5.9,5.55,5.41,5.05, 7.85,5.54,6.99,6.57,6.4,5.96, 8.96,6.38,8.06,7.58,7.38,6.87]

    # print(np.shape(local_thickness))
    # print(np.shape(velocity_collect))
    # print(np.shape(uncert_collect))

    plt.errorbar(local_thickness, velocity_collect, yerr=uncert_collect, label="{}".format(algo.__class__.__name__), fmt='o')
    plt.legend()
    plt.grid(True)
    plt.xlabel('local thickness')
    plt.ylabel('phase velocity')
    plt.title("{}_local thickness_42_points @{}.png".format(algo.__class__.__name__, analysis_freq))

    # csv_file = open("T:\mathematica-svn\phantomResults\id_tubetheory_mohit", "r")
    # reader = csv.reader(csv_file)
    # x = []
    # y = []
    # for line in reader:
    #     t = int(line[0])
    #     p = line[1]
    #     x.append(t)
    #     y.append(p)
    # x = np.array(x) / 1500
    # y = np.array(y)
    # plt.plot(x, y)
    #
    # csv_file = open("T:\mathematica-svn\phantomResults\id_tubetheory_mohit_perfectfit", "r")
    # reader = csv.reader(csv_file)
    # x = []
    # y = []
    # for line in reader:
    #     t = int(line[0])
    #     p = line[1]
    #     x.append(t)
    #     y.append(p)
    # x = np.array(x) / 1500
    # y = np.array(y)
    # plt.plot(x, y)
    #
    # csv_file = open("T:\mathematica-svn\phantomResults\id_lambtheory_mohit", "r")
    # reader = csv.reader(csv_file)
    # x_1 = []
    # y_1 = []
    # for line in reader:
    #     t = int(line[0])
    #     p = line[1]
    #
    #     if int(float(p)) < 1000:
    #         y_1.append(p)
    #         x_1.append(t)
    # x_1 = np.array(x_1) / 1500
    # # print(x, y)
    # # print(np.shape(x))
    # # print(np.shape(y))
    # plt.plot(x_1, y_1, '.')

    plt.ylim(0,1300)
    # plt.xlim(1,9)


    figure = plt.gcf()  # get current figure
    figure.set_size_inches(8, 6)
    outFile = outdir + "\{}_local thickness_42_points @{}.png".format(algo.__class__.__name__, analysis_freq)

    plt.savefig(outFile)
    plt.show()
    plt.close()
    uncert_collect = []
    velocity_collect = []

def cross_area_vs_velocity(algo, analysis_freq):
    '''avg velocity of each phantom vs cross sectional area of each phantom'''
    outdir = r"D:\downloads-D\ETH\Work\Latest Download\Work from home\Axial"

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

def avg_velocity_vs_avg_thickness(algo, analysis_freq):

    outdir = r"T:\Project data and results (svn)\Cortical Thickness Mohit\Results\Axial\avg velocity vs avg thickness"

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
    plt.title('avg velocity vs avg thickness')

    figure = plt.gcf()  # get current figure
    figure.set_size_inches(8, 6)
    outFile = outdir + "\{}_avg_velocity_vs_avg_thickness_all_phantoms_together.png".format(algo.__class__.__name__)
    plt.savefig(outFile)
    # plt.show()
    plt.close()

def avg_velocity_vs_avg_thickness_algoComparison(analysis_freq):

    outdir = r"T:\Project data and results (svn)\Cortical Thickness Mohit\Results\Axial\temp destination"

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

def sensorpoint_vs_velocity_all(analysis_freq):

    outdir = r"D:\downloads-D\ETH\Work\Latest Download\ETH\project data and results\Cortical Thickness Mohit\Results\Axial\Sensor point vs velocity\3000Hz"

    algo_list = (bsmdAlgorithm.ampsemigaussian_sigmafixed(),
                 bsmdAlgorithm.PhaseVelocity_2DGaussianfit(), bsmdAlgorithm.ampgaussian_sigmafree())

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

def mode_velocity_vs_avg_thickness_algoComparison(mode, analysis_freq):

    outdir = r"T:\Project data and results (svn)\Cortical Thickness Mohit\Results\Axial\temp destination"

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
        plt.title('mode velocity vs local thickness {}'.format(mode))

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
    outFile = outdir + r"\mode_velocity_vs_avg_thickness_all_phantoms_together {}.png".format(mode)
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(8, 6)
    plt.savefig(outFile)
    plt.show()
    plt.close()

def scatterplot_algo_comparison(freq_range):

    outdir = r"T:\Project data and results (svn)\Cortical Thickness Mohit\Results\Axial\scatterplot_algo_comparison"

    scatterplot_dataframe = pd.DataFrame()

    algo_list = (bsmdAlgorithm.PhaseVelocity_2DGaussianfit(), bsmdAlgorithm.ampConst_sigmafree(),
                 bsmdAlgorithm.ampsemigaussian_sigmafree())
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
    plt.savefig(outFile)
    plt.show()
    plt.close()
    # print(scatterplot_matrix)

    '''correlatin coefficient matrix plot'''
    sns.set(font_scale=0.5)
    sns.heatmap(corr_coeff_mat, annot=True, annot_kws={"size":12})
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(16, 12)
    plt.savefig(outdir + r"\correlation coefficient matrix _algo_comparison.png")
    plt.show()
    plt.close()

def bland_altman_plot(freq_range):

    outdir = r"T:\Project data and results (svn)\Cortical Thickness Mohit\Results\Axial\bland-altman algo comparison"

    scatterplot_dataframe = pd.DataFrame()

    algo_list = (bsmdAlgorithm.PhaseVelocity_2DGaussianfit(), bsmdAlgorithm.ampConst_sigmafree(),
                 bsmdAlgorithm.ampsemigaussian_sigmafree())

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

def axial_cross_area_42_points(algo, analysis_freq):
    '''min thickness at a measurement point vs velocity'''
    outdir = r"P:\Student\Mohit\project data and results\Cortical Thickness Mohit\Results\Axial\temp destination"

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
            # print(frequency, velocity, uncert)

    local_thickness = [0.000955207,0.001175254,0.001009723,0.001041071,0.001058918,0.001114755,0.000952569,0.00089014,
                       0.000927692,0.000905126,0.000894593,0.000890176,0.00118765,0.000979659,0.001136759,0.001059377,
                       0.001015036,0.00097466,0.001490688,0.001128101,0.001348814,0.001282162,0.001256403,0.001189902,
                       0.001799341,0.001320073,0.001615909,0.00152802,0.00149317,0.001404541,0.002117917,0.001525524,
                       0.001894621,0.001786511,0.001742981,0.001631072,0.00240869,0.00173787,0.002172744,0.002047582,
                       0.001995608,0.001863658]

    # print(np.shape(local_thickness))
    # print(np.shape(velocity_collect))
    # print(np.shape(uncert_collect))

    plt.errorbar(local_thickness, velocity_collect, yerr=uncert_collect, label="{}".format(algo.__class__.__name__),
                 fmt='o')
    plt.legend()
    plt.grid(True)
    plt.xlabel('axial cross section')
    plt.ylabel('phase velocity')
    plt.title("{}_axial_cross_area_42_points @{}.png".format(algo.__class__.__name__, analysis_freq))

    # csv_file = open("T:\mathematica-svn\phantomResults\id_tubetheory_mohit", "r")
    # reader = csv.reader(csv_file)
    # x = []
    # y = []
    # for line in reader:
    #     t = int(line[0])
    #     p = line[1]
    #     x.append(t)
    #     y.append(p)
    # x = np.array(x) / 1500
    # y = np.array(y)
    # plt.plot(x, y)
    #
    # csv_file = open("T:\mathematica-svn\phantomResults\id_tubetheory_mohit_perfectfit", "r")
    # reader = csv.reader(csv_file)
    # x = []
    # y = []
    # for line in reader:
    #     t = int(line[0])
    #     p = line[1]
    #     x.append(t)
    #     y.append(p)
    # x = np.array(x) / 1500
    # y = np.array(y)
    # plt.plot(x, y)
    #
    # csv_file = open("T:\mathematica-svn\phantomResults\id_lambtheory_mohit", "r")
    # reader = csv.reader(csv_file)
    # x_1 = []
    # y_1 = []
    # for line in reader:
    #     t = int(line[0])
    #     p = line[1]
    #
    #     if int(float(p)) < 1000:
    #         y_1.append(p)
    #         x_1.append(t)
    # x_1 = np.array(x_1) / 1500
    # # print(x, y)
    # # print(np.shape(x))
    # # print(np.shape(y))
    # plt.plot(x_1, y_1, '.')
    plt.ylim(5, 1000)
    # plt.xlim(1, 9)

    figure = plt.gcf()  # get current figure
    figure.set_size_inches(8, 6)
    outFile = outdir + "\{}_axial_cross_area_42_points @{}.png".format(algo.__class__.__name__, analysis_freq)

    plt.savefig(outFile)
    plt.show()
    plt.close()
    uncert_collect = []
    velocity_collect = []

#work from home

def one_file_velocity_run(algo,analysis_freq):
    logging.basicConfig(level=logging.DEBUG)
    topDir = r"D:\ETH\Work\Latest Download\Work from home\Axial\Results Rework double peak"
    filelist = mopa_data_review.find_files("*_{}_*.p".format(algo.__class__.__name__), topDir)
    # print(filelist)

    for file in filelist:
        # print("Analyzing file: {}".format(file))
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
    outdir = r"D:\downloads-D\ETH\Work\Latest Download\Work from home\Axial"

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
    local_thickness = [1.72,1.34,1.58, 2.95,2.18,2.66, 2.32, 4.18,3.02, 3.53,3.23, 3.86,4.82,4.54,4.43, 6.62,4.7,5.9,5.55,5.41,5.05, 7.85,5.54,6.99,6.57,6.4,5.96, 8.96,6.38,8.06,7.58,7.38,6.87]

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

    local_thickness_dual_peak = [1.52,1.49, 1.41, 2.54, 2.48, 3.74, 3.45, 5.4, 4.14]
    '''following values are calculated using excel sheet with name dual peak camparisons'''
    velocity_average = [343.6472543, 339.276819, 249.2359355, 431.6025207, 389.919515, 572.3692149, 559.4428504, 646.2099583, 474.6166848]
    velocity_upper = [282.2945095, 281.5393773, 214.6834374, 357.5915918, 311.307649, 423.7848624, 401.7409756, 440.9059169, 381.8102977]
    velocity_lower = [439.0737677, 426.8049964, 297.0440699, 544.2452628, 521.646443, 881.3988358, 920.9635495, 1209.318871, 627.0279779]
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
    plt.xlim(1, 9.5)

    figure = plt.gcf()  # get current figure
    figure.set_size_inches(8, 6)
    outFile = outdir + "\{}_local thickness_42_points dual peaks corrected axial only upper.png".format(algo.__class__.__name__, analysis_freq)

    plt.savefig(outFile)
    plt.show()
    plt.close()
    uncert_collect = []
    velocity_collect = []

def local_thickness_42_points_exact_theory(algo, analysis_freq):
    '''min thickness at a measurement point vs velocity'''
    outdir = r"D:\downloads-D\ETH\Work\Latest Download\Work from home\Axial"

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
    plt.xlabel('max local thickness of axial phantom')
    plt.ylabel('phase velocity')
    plt.title("{}_local thickness_42_points_axial with exact theory.png".format(algo.__class__.__name__, analysis_freq))

    csv_file = open("D:\downloads-D\ETH\Work\Latest Download\Work from home\Axial\cts_exact_tube_results_0.csv", mode="r")
    reader = csv.reader(csv_file)
    x = []
    for line in reader:
        t = float(line[4])
        x.append(t)
    x = np.array(x)
    plt.plot(local_thickness, x, 'ro',label ='Exact tube theory')
    plt.legend()

    plt.ylim(100, 600)
    plt.xlim(1, 9)

    figure = plt.gcf()  # get current figure
    figure.set_size_inches(8, 6)
    outFile = outdir + "\{}_local thickness_42_points @{} with exact theory.png".format(algo.__class__.__name__, analysis_freq)

    plt.savefig(outFile)
    plt.show()
    plt.close()
    uncert_collect = []
    velocity_collect = []

def local_thickness_vs_velocity_color_number_coded_with_doublepeaks(algo, analysis_freq):
    '''min thickness at a measurement point vs velocity'''
    outdir = r"D:\ETH\Work\Latest Download\Work from home\Axial"

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

    local_thickness = [[1.72,1.34,1.58], [2.95,2.18,2.66,2.32], [4.18,3.02,3.53,3.23], [3.86,4.82,4.54,4.43 ], [6.62,4.7,5.9,5.55,5.41,5.05], [7.85,5.54,6.99,6.57,6.4,5.96], [8.96,6.38,8.06,7.58,7.38,6.87]]
    colorlist = ['b', 'g', 'r', 'C0', 'C1', 'C2', 'C3']

    fig, ax = plt.subplots()
    plt.plot([0], [0], label='A : average of peaks \n + :upper peak \n - :lower peak', color='w')

    for phantom in (0, 1, 2, 3, 4, 5, 6):
        '''this block plots non-double peak points together and annotates the according to color and sensor point number'''
        plt.errorbar(local_thickness[phantom], velocity_collect[phantom], markersize=4, yerr=uncert_collect[phantom], color = '{}'.format(colorlist[phantom]),  label="phantom {}".format(phantom + 1), fmt='o')
        # ax.scatter(local_thickness[phantom], velocity_collect[phantom])

        if phantom == 0:
            numberbullets = [1,2,3]
            for txt in numberbullets:
                ax.annotate(txt, (local_thickness[phantom][txt - 1], velocity_collect[phantom][txt - 1]), weight='bold', size = 8, ha='right', va='bottom')

        elif phantom == 1:
            numberbullets = [1,2,3,6]
            for i, txt in enumerate(numberbullets):
                ax.annotate(txt, (local_thickness[phantom][i], velocity_collect[phantom][i]), weight='bold', size = 8, ha='right', va='top')

        elif phantom == 2:
            numberbullets = [1,2,4,6]
            for i, txt in enumerate(numberbullets):
                ax.annotate(txt, (local_thickness[phantom][i], velocity_collect[phantom][i]), weight='bold', size = 8, ha='left', va='bottom')

        elif phantom == 3:
            numberbullets = [2,3,4,5]
            for i, txt in enumerate(numberbullets):
                ax.annotate(txt, (local_thickness[phantom][i], velocity_collect[phantom][i]), weight='bold', size = 6, ha='right', va='bottom')

        else:
            numberbullets = [1, 2, 3, 4, 5, 6]
            for txt in numberbullets:
                ax.annotate(txt, (local_thickness[phantom][txt-1], velocity_collect[phantom][txt-1]), weight='bold', size = 6, ha='right', va='bottom')

        plt.legend()
        plt.grid(True)

    # local_thickness_dual_peak = [1.52,1.49, 1.41, 2.54, 2.48, 3.74, 3.45, 5.4, 4.14]


    '''next two blocks plot the double peaks in phantom 1 and 2 separately'''
    local_thickness_dual_peak_phantom1 = [1.52, 1.49, 1.41]
    numberbullets_doublepeak_phantom1_avg = ['4A','5A','6A']
    numberbullets_doublepeak_phantom1_upper = ['4+', '5+', '6+']
    numberbullets_doublepeak_phantom1_lower = ['4-', '5-', '6-']
    halignment = ['left', 'right', 'right']
    valignment = ['bottom', 'top', 'bottom']
    '''following values are calculated using excel sheet with name dual peak camparisons'''
    velocity_average_phantom1 = [343.6472543, 339.276819, 249.2359355]
    velocity_upper_phantom1 = [282.2945095, 281.5393773, 214.6834374]
    velocity_lower_phantom1 = [439.0737677, 426.8049964, 297.0440699]
    uncert_average_phantom1 = [3.156024350243835, 2.5835459257255757, 2.009493059860092]
    uncert_upper_phantom1 = [0.6806064385896275, 0.5481700930027668, 0.6659371985571859]
    uncert_lower_phantom1 = [1.5697666760941382, 1.4488452505037077, 2.632466778286837]


    plt.errorbar(local_thickness_dual_peak_phantom1, velocity_average_phantom1, markersize=4, yerr = uncert_average_phantom1, fmt = 'bo')
    plt.errorbar(local_thickness_dual_peak_phantom1, velocity_upper_phantom1, markersize=4, yerr = uncert_upper_phantom1,fmt = 'bo')
    plt.errorbar(local_thickness_dual_peak_phantom1, velocity_lower_phantom1, markersize=4, yerr = uncert_lower_phantom1,fmt = 'bo')
    for i in [0,1,2]:
        ax.annotate(numberbullets_doublepeak_phantom1_avg[i], (local_thickness_dual_peak_phantom1[i], velocity_average_phantom1[i]),weight='bold', size = 8, ha='right', va='{}'.format(valignment[i]))
        ax.annotate(numberbullets_doublepeak_phantom1_upper[i], (local_thickness_dual_peak_phantom1[i], velocity_upper_phantom1[i]),weight='bold', size = 8, ha='{}'.format(halignment[i]), va='bottom')
        ax.annotate(numberbullets_doublepeak_phantom1_lower[i], (local_thickness_dual_peak_phantom1[i], velocity_lower_phantom1[i]),weight='bold', size = 8, ha='right', va='bottom')
    plt.legend()

    local_thickness_dual_peak_phantom2 = [2.54, 2.48]
    numberbullets_doublepeak_phantom2_avg = ['4A', '5A']
    numberbullets_doublepeak_phantom2_upper = ['4+', '5+']
    numberbullets_doublepeak_phantom2_lower = ['4-', '5-']
    '''following values are calculated using excel sheet with name dual peak camparisons'''
    velocity_average_phantom2 = [431.6025207, 389.919515]
    velocity_upper_phantom2 = [357.5915918, 311.307649]
    velocity_lower_phantom2 = [544.2452628, 521.646443]
    uncert_average_phantom2 = [2.3415709263002555, 25.770834150326305]
    uncert_upper_phantom2 = [3.0126800796051536, 0.9477614839376391]
    uncert_lower_phantom2 = [8.960771796048503, 2.255157379562995]
    plt.errorbar(local_thickness_dual_peak_phantom2, velocity_average_phantom2, markersize=4, yerr=uncert_average_phantom2, fmt='go')
    plt.errorbar(local_thickness_dual_peak_phantom2, velocity_upper_phantom2, markersize=4, yerr=uncert_upper_phantom2, fmt='go')
    plt.errorbar(local_thickness_dual_peak_phantom2, velocity_lower_phantom2, markersize=4, yerr=uncert_lower_phantom2, fmt='go')
    for i in [0,1]:
        ax.annotate(numberbullets_doublepeak_phantom2_avg[i], (local_thickness_dual_peak_phantom2[i], velocity_average_phantom2[i]),weight='bold', size = 8, ha='{}'.format(halignment[i]), va='bottom')
        ax.annotate(numberbullets_doublepeak_phantom2_upper[i], (local_thickness_dual_peak_phantom2[i], velocity_upper_phantom2[i]),weight='bold', size = 8, ha='right', va='bottom')
        ax.annotate(numberbullets_doublepeak_phantom2_lower[i], (local_thickness_dual_peak_phantom2[i], velocity_lower_phantom2[i]),weight='bold', size = 8, ha='right', va='bottom')
    plt.legend()

    local_thickness_dual_peak_phantom3 = [3.74, 3.45]
    numberbullets_doublepeak_phantom3_avg = ['3A', '5A']
    numberbullets_doublepeak_phantom3_upper = ['3+', '5+']
    numberbullets_doublepeak_phantom3_lower = ['3- 881.39 m/s', '5- 920.96 m/s']
    '''following values are calculated using excel sheet with name dual peak camparisons'''
    velocity_average_phantom3 = [572.3692149, 559.4428504]
    velocity_upper_phantom3 = [423.7848624, 401.7409756]
    velocity_lower_phantom3 = [700, 700] #here, 700 is range value for the outlier point with actual velocity 881m/s and 920m/s which is put as annotation text
    uncert_average_phantom3 = [6.48597475749972, 12.275327396238684]
    uncert_upper_phantom3 = [1.0721523039101852,0.7354990532886283 ]
    uncert_lower_phantom3 = [11.168185954615774,17.13731551115852 ]
    plt.errorbar(local_thickness_dual_peak_phantom3, velocity_average_phantom3, markersize=4, yerr=uncert_average_phantom3, fmt='ro')
    plt.errorbar(local_thickness_dual_peak_phantom3, velocity_upper_phantom3, markersize=4, yerr=uncert_upper_phantom3, fmt='ro')
    plt.errorbar(local_thickness_dual_peak_phantom3, velocity_lower_phantom3, markersize=4, yerr=uncert_lower_phantom3, fmt='ro')
    alignment_va = ['top', 'bottom', 'top']
    for i in [0, 1]:
        ax.annotate(numberbullets_doublepeak_phantom3_avg[i],
                    (local_thickness_dual_peak_phantom3[i], velocity_average_phantom3[i]), weight='bold', size=8,
                    ha='right', va='bottom')
        ax.annotate(numberbullets_doublepeak_phantom3_upper[i],
                    (local_thickness_dual_peak_phantom3[i], velocity_upper_phantom3[i]), weight='bold', size=8,
                    ha='right', va='bottom')
        ax.annotate(numberbullets_doublepeak_phantom3_lower[i],
                    (local_thickness_dual_peak_phantom3[i], velocity_lower_phantom3[i]), weight='bold', size=8,
                    ha='{}'.format(halignment[i]), va='top')
    plt.legend()

    local_thickness_dual_peak_phantom4 = [5.4, 4.14]
    numberbullets_doublepeak_phantom4_avg = ['1A', '6A']
    numberbullets_doublepeak_phantom4_upper = ['1+', '6+']
    numberbullets_doublepeak_phantom4_lower = ['1- 1209.31 m/s', '6-']
    '''following values are calculated using excel sheet with name dual peak camparisons'''
    velocity_average_phantom4 = [646.2099583, 474.6166848]
    velocity_upper_phantom4 = [440.9059169, 381.8102977]
    velocity_lower_phantom4 = [700, 627.0279779] #here, 650 is range value for the outlier point with actual velocity 1209m/s which is put as annotation text
    uncert_average_phantom4 = [7.978852532951758, 3.658275597588124]
    uncert_upper_phantom4 = [1.1848964748544926, 1.4754194796609996]
    uncert_lower_phantom4 = [19.47909961900277, 12.986045525853319]
    plt.errorbar(local_thickness_dual_peak_phantom4, velocity_average_phantom4, markersize=4, yerr=uncert_average_phantom4, fmt='C0o')
    plt.errorbar(local_thickness_dual_peak_phantom4, velocity_upper_phantom4, markersize=4, yerr=uncert_upper_phantom4, fmt='C0o')
    plt.errorbar(local_thickness_dual_peak_phantom4, velocity_lower_phantom4, markersize=4, yerr=uncert_lower_phantom4, fmt='C0o')
    for i in [0, 1]:
        ax.annotate(numberbullets_doublepeak_phantom4_avg[i],
                    (local_thickness_dual_peak_phantom4[i], velocity_average_phantom4[i]), weight='bold', size=8,
                    ha='right', va='bottom')
        ax.annotate(numberbullets_doublepeak_phantom4_upper[i],
                    (local_thickness_dual_peak_phantom4[i], velocity_upper_phantom4[i]), weight='bold', size=8,
                    ha='right', va='bottom')
        ax.annotate(numberbullets_doublepeak_phantom4_lower[i],
                    (local_thickness_dual_peak_phantom4[i], velocity_lower_phantom4[i]), weight='bold', size=8,
                    ha='left', va='top')
    plt.legend()

    plt.ylim(180, 702)
    plt.xlim(1,9)
    plt.xlabel('Local thickness (mm)')
    plt.ylabel('Phase velocity (m/s)')
    # plt.title("colorcoded_42points_{}_local thickness_vs_velocity_with_double_peaks_Axial".format( algo.__class__.__name__))
    #
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(6.5, 4)
    outFile = outdir + "\colorcoded_42points_{}_local thickness_vs_velocity_number_coded_with_doublepeaks_axial_uncert".format( algo.__class__.__name__)
    plt.savefig(outdir + r"\results axial legends transparent.eps", dpi=300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=False)
    plt.savefig(outdir + r"\results axial legends transparent.png", dpi=300, facecolor='w', edgecolor='w',
                orientation='portrait', transparent=False)
    plt.show()
    plt.close()
    # uncert_collect = []
    # velocity_collect = []

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

    # algo_list = (bsmdAlgorithm.PhaseVelocity_2DGaussianfit(), bsmdAlgorithm.ampgaussian_sigmafree(), bsmdAlgorithm.ampsemigaussian_sigmafixed())

    # for algo in algo_list:
    '''saving the result dataframe to a pickle for future use. Also contains plotting function saving dispersion relation overlap with 2Dfft'''
    # saving_dataframes()

    '''for plotting freq vs velocity with errorbars'''
    # frequency_vs_velocity(algo)

    '''velocity vs sonsor point plots for each phantom'''
    # sensorpoint_vs_velocity(algo, 3000)

    '''avg velocity vs. avg thickness of phantom (one velocity per phantom)'''
    # avg_velocity_vs_avg_thickness(algo,3000)
    # avg_velocity_vs_avg_thickness_algoComparison(3000)

    '''average_ss values listing'''
    # average_ss_listings(algo)
    # average_ss_check()

    '''local_thickness_vs_velocity for analysis frequency'''
    # local_thickness_vs_velocity(algo,3000)
    # local_thickness_42_points(algo,3000)
    # axial_cross_area_42_points(algo,3000)
    # mode_velocity_vs_avg_thickness_algoComparison(5,3000)

    '''cross area of each phantom vs avg velocity of each phantom for analysis frequency and all algorithms'''
    # cross_area_vs_velocity(algo, 3000)

    '''scatter matrix plot algorithm comparison'''
    # scatterplot_algo_comparison((2900,3100))

    '''bland-altman plot for the algorithm pairs'''
    # bland_altman_plot((2900,3100))

    '''sensor point velocity'''
    # sensorpoint_vs_velocity(3000)

    '''Rework with florian'''
    # one_file_velocity_run(algo, 3000)

    # local_thickness_42_points_dualpeak_corrected(algo, 3000)

    # local_thickness_42_points_exact_theory(algo, 3000)

    local_thickness_vs_velocity_color_number_coded_with_doublepeaks(algo,3000)