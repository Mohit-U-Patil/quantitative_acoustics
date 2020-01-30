import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import bsmd.algorithm.bsmdAlgorithmRestructured as bsmdAlgorithm
# import bsmd.algorithm.bsmdAlgorithm as bsmdAlgorithm
import bsmd.measurement.bsmdMeasurement as bsmdMeasurement
import matplotlib.pyplot as plt
import mopa_data_review
import os
import numpy as np
import pandas as pd
import logging
from itertools import cycle
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from mopa_2dgauss import twoD_gaussian as gauss
import argparse
import pylab
import pickle
import scipy.signal as sig
import scipy
import csv


'''Temporary tests to see the fitting of data with 2d gaussian and 3D plot to get better visualizations'''

logging.basicConfig(level=logging.DEBUG)
topDir = r"D:\downloads-D\ETH\Work\Latest Download\ETH\project data and results\Cortical Thickness Mohit\XML data\Phantom_radial_1.5-7.5mm"
outdir = r"D:\downloads-D\ETH\Work\Latest Download\Work from home"
# meas = bsmdMeasurement.bsmdMeasurement(xFFTNr=25000, yFFTNr=3500)
filelist = mopa_data_review.find_files("point4_r15mm.xml", topDir)
parameter_list = pd.read_excel(r'D:\downloads-D\ETH\Work\Latest Download\ETH\project data and results\Cortical Thickness Mohit\Results\Radial\All algorithms\Result Dataframe Pickles\parameter record.xlsx')
parameter_list.index = parameter_list['filename']


# algo = bsmdAlgorithm.PhaseVelocity_2DFFTGauss()
# algo = bsmdAlgorithm.PhaseVelocity_2DFFTSinc()
# algo = bsmdAlgorithm.PhaseVelocity_2DGaussianfit()
# algo = bsmdAlgorithm.PhaseVelocity_2DFFTMax()
# algo = bsmdAlgorithm.ampConst_sigmafixed()
# algo = bsmdAlgorithm.ampConst_sigmafree()
# algo = bsmdAlgorithm.ampgaussian_sigmafixed()
algo = bsmdAlgorithm.ampgaussian_sigmafree()
# algo = bsmdAlgorithm.ampsemigaussian_sigmafixed()
# algo = bsmdAlgorithm.ampsemigaussian_sigmafree()
for file in filelist:


    print("Analyzing file: {}".format(file))
    # meas.readStepPhantomXMLData(file)
    freq1 = parameter_list['freq1'][os.path.basename(file)]
    freq2 = parameter_list['freq2'][os.path.basename(file)]
    iW1 = parameter_list['iW1'][os.path.basename(file)]
    iW2 = parameter_list['iW2'][os.path.basename(file)]
    print('parameters for {}- frquency range = ({},{}), wavenumber range = ({},{})'.format(os.path.basename(file), freq1,
                                                                                         freq2, iW1, iW2))

    '''plotting fft data with wavenumber data superimposed'''
    # meas.plot2D(abs(meas.dataFFT), "Frequency (Hz)", "Wavenumber (1/m)", title=os.path.basename(file) + "_basic_fft",
    #             clickEvent=None,
    #             showImages=True, colorbar=True,
    #             xRange=(2000, 4000), yRange=(-20, 20))
    # result = algo.phaseVelocity(meas, freq=(2600, 3600), wavenumber=(4, 16), p0=(15000, 0.001, 1), mode='linear',
    #                                                         sliced_plot=None, debug_plot=None, outdir=None, sumOfSquares_threshold=0.05,
    #                                                         normalize=True, oldschool=None,conditional_gauss=True, file=file)

    '''checking printing functions for study script'''
    # meas.plot2D_all("Raw measurement", clickEvent= True, showImages = True, colorbar = True)
    # meas.plot2D(meas.data, xLabel = "Time (s)", yLabel = "Pos (m)", title = os.path.basename(file), clickEvent = True, showImages = True, colorbar = True, xRange = (0.01,0.014), yRange=None)
    # meas.plot1D(meas.data.ix[0.06], "Time (s)", "Amplitude (g)", title = os.path.basename(file), showImages = True)


    # '''windowing the data before fft, plotting with debug_plot=true to get the fitted one-D gauss plot'''
    # # # window = np.hamming(36)
    # # # copy = meas.data.copy()
    # # # copy = np.transpose(copy)
    # # # copy = copy * window
    # # # copy = np.transpose(copy)
    # # # meas.data = copy
    # # print(np.shape(meas.dataFFT))
    # result = algo.phaseVelocity(meas, freq = (2600,3300), wavenumber = (4,12), debug_plot = None)
    # b = result.iloc[:, 0:].values
    # print(result)
    # print(b)

    '''restructured bsmdAlgorithm testing, choose bsmdAlgorithmRestructured in import'''
    # algo = bsmdAlgorithm.PhaseVelocity_2DFFTGauss()
    # result = algo.phaseVelocity(meas, freq=(2999,3001), wavenumber=(0,20), outdir=True, sumOfSquares_threshold=0.6)
    # print(result)
    # outFile = "T:\Data\Cortical Thickness Mohit\testing directory\testfile.txt"
    # results = result.T
    # text_file = open(outFile, "w")
    # text_file.write("filename; phase_velocity; lower; upper; avgSS; frequency\n")  # write header
    # text_file.write("{};{};{}\n".format(os.path.basename(file), results.loc["velocity", frequency], frequency))
    # T_result = results.T
    # T_result["filename"] = os.path.basename(file)
    # T_result["frequency"] = T_result.index
    # T_result = T_result.reset_index(drop=True)
    # # add uncertainty according to elliptical relations
    # # DataFrame_result = pd.concat([DataFrame_result, T_result])

    # text_file.close()

    '''penultimate run trials'''
    # result = algo.phaseVelocity(meas, freq=(2800, 3800), wavenumber=(4, 16), p0=(15000, 0.001, 1), mode = 'linear',dispersion_plot=True,
    #                             sliced_plot=True, debug_plot=True, outdir=None, sumOfSquares_threshold=0.05,
    #                             normalize=True, oldschool=None,conditional_gauss=True)
    # pickle.dump(result, open("result.p", "wb"))
    # print(result)
    # pickle_rick = pickle.load(open("result.p", "rb"))
    # print(pickle_rick)

    '''penultimate run 2D gaussian fit trials'''
    # result= algo.phaseVelocity(meas , freq=(2300, 3200),wavenumber=(4,14), p0=(15000, (freq1+freq2)/2 , 11, 1, 1, 0.5)  ,sliced_plot = True, debug_plot = True, sumOfSquares_threshold = None, normalize = True)
    # print(result)

    '''penulimate run 2D fft fit - 2DFFTGauss'''
    # try:
    # result = algo.phaseVelocity(meas,  freq=(2600, 3400), wavenumber=(4, 10), p0=(10000., 8., 0.7, 1000), mode='linear',
    #                             dispersion_plot=None, showImages=None, window=True,
    #                             sliced_plot=None, debug_plot=None, outdir=None, sumOfSquares_threshold=0.5,
    #                             normalize=True, oldschool=None, conditional_gauss=True,
    #
    #                             title="{}_{}_linear_dispersion overlap.png".format(os.path.basename(file),
    #                                                                                algo.__class__.__name__))
    # outFile = outdir + "\{}_{}_linear_dispersion overlap.png".format(os.path.basename(file),
    #                                                                  algo.__class__.__name__),
    # print(result)

    # except:
    #     pass

    '''penultimate run 2D fft max'''
    # try:
    #     result = algo.phaseVelocity(meas,  freq=(freq1, freq2), wavenumber=(iW1, iW2), mode='linear',
    #                                 dispersion_plot=None, showImages=None, window=True,
    #                                 sliced_plot=None, debug_plot=None, outdir=None, sumOfSquares_threshold=0.1,
    #                                 normalize=True, oldschool=None, conditional_gauss=True,
    #                                 title="{}_{}_linear_dispersion overlap.png".format(os.path.basename(file),
    #                                                                                    algo.__class__.__name__))
    #     # outFile = outdir + "\{}_{}_linear_dispersion overlap.png".format(os.path.basename(file),
    #     #                                                                  algo.__class__.__name__),
    #     print(result)
    #
    # except:
    #     pass

    '''penultimate run ampconst_sigmafree'''
    # try:
    # result = algo.phaseVelocity(meas, freq=(freq1, freq2), wavenumber=(iW1, iW2),  p0=(15000, 0.001, 1, 1), mode='linear',
    #                         dispersion_plot=True, showImages=True, window= True,
    #                         sliced_plot=None, debug_plot=None, outdir=None, sumOfSquares_threshold=1,
    #                         normalize=True,
    #                         title="{}_{}_linear_dispersion overlap.png".format(os.path.basename(file),
    #                                                                            algo.__class__.__name__))
    # pickle.dump(result, open(outdir + "\{}_{}_basicData.p".format(os.path.basename(file), algo.__class__.__name__),
    #                                  "wb"))
    # # outFile = outdir + "\{}_{}_linear_dispersion overlap.png".format(os.path.basename(file),
    # #                                                                  algo.__class__.__name__),
    # print(result)
    # except:
    #     pass

    '''penultimate run ampgaussian_sigmafixed'''
    # try:
    #     result = algo.phaseVelocity(meas, freq=(freq1, freq2), wavenumber=(iW1, iW2),  p0=(15000, 0.001, 1,  3000, 1), mode='linear',
    #                             dispersion_plot=True, showImages=True, window= True,
    #                             sliced_plot=True, debug_plot=True, outdir=None, sumOfSquares_threshold=0.1,
    #                             normalize=None, outFile = outdir + "\{}_{}_linear_dispersion overlap.png".format(os.path.basename(file),
    #                                                                      algo.__class__.__name__),
    #                             title="{}_{}_linear_dispersion overlap.png".format(os.path.basename(file),
    #                                                                                algo.__class__.__name__))
    #
    #     print(result)
    # except:
    #     pass

    '''penultimate run ampgaussian_sigmafree'''
    # # try:
    #     result = algo.phaseVelocity(meas, freq=(freq1, freq2), wavenumber=(iW1, iW2),  p0=(15000, 0.001, 1, (freq1+freq2)/2 , 1 , 1), mode='linear',
    #                             dispersion_plot=True, showImages=True, window= True,
    #                             sliced_plot=None, debug_plot=None, outdir=None, sumOfSquares_threshold=0.1,
    #                             normalize=None,outFile = outdir + "\{}_{}_linear_dispersion overlap.png".format(os.path.basename(file),
    #                                                                        algo.__class__.__name__),
    #                             title="{}_{}_linear_dispersion overlap.png".format(os.path.basename(file),
    #                                                                                algo.__class__.__name__))
    #
    #     print(result)
    # # except:
    # #     pass

    '''penultimate run ampsemigaussian_sigmafixed'''
    # try:
    #     result = algo.phaseVelocity(meas, freq=(freq1, freq2), wavenumber=(iW1, iW2),  p0=(15000, 0.001, 1), mode='linear',
    #                             dispersion_plot=True, showImages=None, window= True,
    #                             sliced_plot=None, debug_plot=None, outdir=None, sumOfSquares_threshold=0.1,
    #                             normalize=None,outFile = outdir + "\{}_{}_linear_dispersion overlap.png".format(os.path.basename(file),
    #                                                                        algo.__class__.__name__),
    #                             title="{}_{}_linear_dispersion overlap.png".format(os.path.basename(file),
    #                                                                                algo.__class__.__name__))
    #
    #     print(result)
    # except:
    #     pass

    '''penultimate run ampsemigaussian_sigmafree'''
    # try:
    # result = algo.phaseVelocity(meas, freq=(freq1, freq2), wavenumber=(iW1, iW2), p0=(15000, 0.001, 1,  1),
    #                             mode='linear',
    #                             dispersion_plot=True, showImages=True, window=True,
    #                             sliced_plot=None, debug_plot=None, outdir=None, sumOfSquares_threshold=0.1,
    #                             normalize=None, outFile=outdir + "\{}_{}_linear_dispersion overlap.png".format(
    #         os.path.basename(file),
    #         algo.__class__.__name__),
    #                             title="{}_{}_linear_dispersion overlap.png".format(os.path.basename(file),
    #                                                                                algo.__class__.__name__))
    #
    # print(result)
    # except:
    #     pass

    '''with the hardcoded fakedata function'''
    # result= algo.testData(meas , freq = (2700, 3500), wavenumber = (4,12), p0=(0.7, 3300, 8, 1, 1, 0.5) ,sliced_plot = True, debug_plot = True, outdir = None, sumOfSquares_threshold = 0.03, normalize = True, oldschool=True,
    #                       conditional_gauss=None)
    # print(result)

    '''slicing the data to get the important window based on frequency and wavenumber'''
    # print(meas.dataFFT)
    # fft = meas.dataFFT.ix[10:4, 2700:3400] # .ix working fine with the actual value ranges of wavenumber and frequency even with non integer values, w1,w2,f1,f2 gives out error.

    # fft = abs(fft)
    # X = fft.columns #frequency
    # Y = fft.index #wavenumber
    # # t = fft.values.T.tolist()
    # # Z = t.np.asarray()
    # Z = fft
    # Z = fft.values
    # print(type(Z))
    # a = X.shape
    # b = a[0]
    # print(b)
    # c = Y.shape
    # d = c[0]
    # print(d)
    # Z = Z.reshape(b*d,)
    # # print(X)
    # # print(Y)
    # # print(Z)
    #
    # Xcopy = X
    # Ycopy = Y
    # base = gauss.data_meshgrid(X, Y)
    # X,Y = base
    #
    # # print(type(base), type(X), type(Z))
    # print(X.shape, Y.shape,Z.shape, fft.shape)
    #
    # '''Temporary tests to see the 3D plot to get better visualizations'''
    #
    # '''Plot the surface.'''
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # surf = ax.plot_surface(X,Y , fft, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    # # # ax.set_xlim(1000,7000)
    # # # ax.set_ylim(0, 20)
    # ax.set_xlabel('X axis-frequency')
    # ax.set_ylabel('Y axis-wavenumber ')
    # ax.set_zlabel('Z axis- magnitude')
    #
    # plt.show()
    #
    # '''temporary test to see the 2D gaussian fit'''
    # initial_guess= (15000, 2700, 6, 1, 1, 0.5)
    # data_fitted, popt, covariance_matrix, eigenval, eigenvec = gauss.multivariate_Gaussian_fit(base, Z,initial_guess)
    #
    # '''plotting fitted 2D gaussian curve'''
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # surf = ax.plot_surface(X,Y , data_fitted.reshape(X.shape), cmap=cm.coolwarm,linewidth=0, antialiased=False)
    # ax.set_xlabel('X axis-frequency')
    # ax.set_ylabel('Y axis-wavenumber ')
    # ax.set_zlabel('Z axis- magnitude')
    # plt.show()
    #
    # '''finding cordinates (frequency,wavenumber) of the maximum'''
    # print(data_fitted.shape)
    # max = np.argmax(data_fitted)
    # print(max)
    # print(data_fitted[max])
    # # index = np.unravel_index(max,2)
    # # print(index)
    # data_fitted = data_fitted.reshape(X.shape)
    # print(data_fitted.shape)
    # i, j = np.unravel_index(data_fitted.argmax(), data_fitted.shape)
    # print(data_fitted[i,j], i, j)
    # print(Xcopy[j])
    # print(Ycopy[i])
    # # maxcol = X.ix[data_fitted[max]]
    # # maxrow = y.ix[data_fitted[max]]
    # # print(max,maxcol,maxrow)
    #
    # '''establishing a relationship between k and f for evaluating the phase velocity in the whole range of frequencies'''
    # # we need maxima cordinates, the right slope from eigenvectors and the frequency range, add filter for negative slopes
    # freq_range = fft.columns
    # print(eigenvec[1, 0])
    # print(eigenvec[0, 0])
    # q = np.absolute(Xcopy[j] * (eigenvec[1, 0] / eigenvec[0, 0]))
    # c = Ycopy[i] - q
    # print(q)
    # print(c)
    #
    # fitted_params = []
    # for frequency in freq_range:
    #     k = (frequency * np.absolute(eigenvec[1, 0] / eigenvec[0, 0])) + c
    #     phase_velocity = frequency /k
    #     # print(phase_velocity, frequency)
    #     fitted_params.append({"iWavelength": (k), "Velocity": (phase_velocity)})
    #
    # results = pd.DataFrame(index=fft.columns)
    # frequencies = fft.columns.values
    # iWavelengths = np.array([s["iWavelength"] for s in fitted_params])
    # results["velocity"] = np.divide(freq uencies, iWavelengths)
    # # return results.T
    # print(results)
    # results = results.T
    # text_file = open(outdir + "\{}_velocity_data_{}.txt".format(os.path.basename(file), 3000), "w")
    # text_file.write("filename; phase_velocity; lower; upper; avgSS; frequency\n")  # write header
    # text_file.write("{};{};{}\n".format(os.path.basename(file), results.loc["velocity", frequency], frequency))
    # T_result = results.T
    # T_result["filename"] = os.path.basename(file)
    # T_result["frequency"] = T_result.index
    # T_result = T_result.reset_index(drop=True)
    # # add uncertainty according to elliptical relations
    # # DataFrame_result = pd.concat([DataFrame_result, T_result])
    #
    # text_file.close()

    '''test dummy data , same as test cases'''
    # xValid = np.linspace(0, 1000, 781)
    # yValid = np.linspace(0, 1000, 1068)
    #
    # initial_guess = (3, 100, 100, 20, 40)
    # base_data = gauss.data_meshgrid(xValid, yValid)
    #
    # data = gauss.twoD_Gaussian(base_data, 3, 100, 100, 20, 40)
    # np.random.seed(2)
    # data_to_fit = data + 0.1 * np.random.normal(size=data.shape)
    #
    # print(data_to_fit)
    # print(xValid)
    # print(yValid)
    #
    # data_fitted, popt, pcov = gauss.twoD_Gaussian_fit(base_data, data_to_fit, initial_guess)
    # print(type(base_data))
    # print(type(data_to_fit))
    # print(xValid.shape, yValid.shape, data_to_fit.shape)
    # print(data_fitted,popt,pcov)

    '''writing a text file of results and saving it. this saves fftgauss fit. here results are being calculated independently of the above fitting using algo.phasevelocity(). '''
    # text_file = open(outdir + "\{}_velocity_data_{}.txt".format(os.path.basename(file), 3000), "w")
    # text_file.write("filename; phase_velocity; lower; upper; avgSS; frequency\n")  # write header
    # DataFrame_result = pd.DataFrame()
    # # for file in measFiles:
    #
    #
    # result = algo.phaseVelocity(meas, freq=(2950, 3050), wavenumber= (5,9) ,debug_plot=False,OfSquares_threshold=1)  # make sure to analyse all datafiles, even if they have crappy fittings
    # print(result)
    # frequency = meas.dataFFT.columns[meas.findNearestIndex(meas.dataFFT.columns, 3000)]
    # text_file.write("{};{};{};{};{};{}\n".format(os.path.basename(file), result.loc["velocity", frequency],
    #                                                  result.loc["lower", frequency], result.loc["upper", frequency],
    #                                                  result.loc["averageSS", frequency], frequency))
    # T_result = result.T
    # T_result["filename"] = os.path.basename(file)
    # T_result["frequency"] = T_result.index
    # T_result = T_result.reset_index(drop=True)
    # DataFrame_result = pd.concat([DataFrame_result, T_result])
    #
    # text_file.close()


    # DataFrame_result["specimen"] = DataFrame_result["filename"].apply(lambda a: a[7:])
    # DataFrame_result["point"] = DataFrame_result["filename"].apply(lambda b: b[5:6])
    # DataFrame_result["series"] = DataFrame_result["filename"].apply(lambda b: b[5:6])
    # DataFrame_result["specimen_value"] = DataFrame_result["filename"].apply(lambda c: c[8:10])
    # DataFrame_result["sensor_point"] = DataFrame_result["filename"].apply(lambda c: c[6:7])  # reliability
    #    print(DataFrame_result["point"])

    '''writing results analogous to phasevelocity fftfit method from bsmdAlgorithm'''
    # results = pd.DataFrame(index=fft.columns)
    # # just aliases for better legibility
    # frequencies = fft.columns.values
    # iWavelengths = np.array([s["iWavelength"][0] for s in
    #                          fitted_params])  # the inverse wavelength corresponding to the maximum of the gauss, for each frequency
    # sigmas = np.array([s["sigma"][0] for s in fitted_params])  # the sqrt(variance) of the gauss, for each frequency
    # amplitudes = np.array([s["Amplitude"][0] for s in fitted_params])
    #
    # uncert_iWs = np.array([s["iWavelength"][1] for s in fitted_params])
    # assert np.all(uncert_iWs >= 0)
    #
    # results["velocity"] = np.divide(frequencies, iWavelengths)
    # uncert_vel = np.divide(frequencies * uncert_iWs,
    #                        iWavelengths ** 2)  # follows from error propagation of the velocity, when inverse wavelength has an uncertainty
    # results["lower"] = results["velocity"] - uncert_vel
    # results["upper"] = results["velocity"] + uncert_vel
    # results["averageSS"] = [s["averageSS"][0] for s in fitted_params]
    # return results.T

    '''checking fft behavior'''
    # N = 128
    # x = np.arange(0, 9 * np.pi, 9 * np.pi/N)
    # # y = 1 * np.exp(-(x - 5) ** 2 / (2. * 1 ** 2))
    # y = np.sin(x)
    # for i in np.arange(len(y)):
    #     if y[i] < 0:
    #         y[i] = 0
    # y_fft = np.fft.fftshift(np.abs(np.fft.fft(y))) / np.sqrt(2 * 2*N)
    # plt.plot(x,y)
    # plt.plot(x,y_fft)
    # plt.show()

    """plotting csv results from mathematica"""
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
    plt.xlim(1, 8)

    csv_file = open(
        "D:\downloads-D\ETH\Work\Latest Download\ETH\Mathematica svn-20180816T165737Z-001\Mathematica svn\",
        "r")
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
    plt.xlim(1, 8)

    figure = plt.gcf()  # get current figure
    figure.set_size_inches(8, 6)
    outFile = outdir + "\mathematic plot"

    plt.savefig(outFile)
    plt.show()
    plt.close()