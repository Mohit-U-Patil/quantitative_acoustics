import bsmdMeasurement
import bsmdAlgorithm
import matplotlib.pyplot as plt

import os
import numpy as np
import pandas as pd
import logging
from itertools import cycle
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd





#==============================================================================
#     Measurement Data of phantoms_cortical-thickness
#==============================================================================
#XML-files for the radial_and_axial_changed phantoms_wall_thickness: Contain measurement-data from measured-point 1-6 of the phantoms with wall_thickness 1.5-7.5mm


dataRoot =  r"Z:\Student\Danun\measurements-svn\cortical_thickness-ashraf"
errorRoot= r"Z:\Student\Danun\measurements-svn\cortical_thickness-ashraf\Repeatability_measurement_raw_data"
error_dataRoot= os.path.join(errorRoot, "Reliability_test_Radial_4.5_corrected")
errorFiles=[os.path.join(error_dataRoot, "point{}{}_reliability_tester_RT_45mm.xml".format(series,point)) for series in range(1,7) for point in range(1,7)]
radial_dataRoot = os.path.join(dataRoot, "Phantom_radial_1.5-7.5mm")
rPhantomsFiles = [os.path.join(radial_dataRoot, "point{}_r{}mm.xml".format(point, thickness)) for point in range(1,7) for thickness in [15,25,35,45,55,65,75]]
axial_dataRoot =  os.path.join(dataRoot, "Phantom_axial_1.5-7.5mm")
aPhantomsFiles = [os.path.join(axial_dataRoot, "point{}_a{}mm.xml".format(point, thickness)) for point in range(1,7) for thickness in [15,25,35,45,55,65,75]]



#==============================================================================
#     Summarized procedures, which haven been done
#==============================================================================
  
#The time-(x-axis)-plots shows for the first 3 measured-positions a high deviation of the amplitude compare to the other measured positions.This plots looked colorless.    
def first_viewing():       
       

    
    meas = bsmdMeasurement.bsmdMeasurement(xFFTNr = 20000, yFFTNr = 350)
    for file in scaleFiles:
        meas.readStepPhantomXMLData(file)
        meas.plot2D_all("Raw measurement", clickEvent= True, showImages = True, colorbar = True)
        meas.plot2D(meas.data, xLabel = "Time (s)", yLabel = "Pos (m)", title = os.path.basename(file), clickEvent = True, showImages = True, colorbar = True, xRange = (0.01,0.014), yRange=None)
        meas.plot1D(meas.data.ix[0.06], "Time (s)", "Amplitude (g)", title = os.path.basename(file), showImages = True)


#Solving the problem from first_viewing()    
def assess_influence_of_standardization():
    """
        Take one meausurement with exceptionally high amplitude in the first and last 3-6 measurement positions
        Assess the influence on the 2D FFT, when these 3 sensor locations have their amplitude manually reduced to the normal level (of all the other measurements)
    """
    outdir = "1Dplot_FFT_y714285."
    meas = bsmdMeasurement.bsmdMeasurement(xFFTNr = 20000, yFFTNr = 350)
#    for file in errorFiles:
    meas.readStepPhantomXMLData(rPhantomsFiles[40])
    print(rPhantomsFiles[40])
    meas.data = meas.movingTimeFilter(np.hamming(400), starttime = 0.011198275, speed = 479.98464049150425, showImages = None)
#    meas.plot2D_all("Circumferential stepped femur phantom ID122_circumferential_7.5 mm filtered by hamming", clickEvent=True, showImages = True)   
    meas.plot2D(meas.data, "Time [s]", "Sensor Position [m]", title = "Circumferential stepped femur phantom ID122_circumferential_7.5 mm (sensor point nr. 6) filtered by Hamming function", clickEvent= True, showImages = True, colorbar = True,  xRange = (0.01 , 0.014), yRange=None)  
    meas.plot2D(abs(meas.dataFFT), "Frequency [Hz]", "Wavenumber [1/m]", "Circumferential stepped femur phantom ID122_circumferential_7.5 mm (sensor point nr. 6) filtered by Hamming function", clickEvent= True, showImages = True, colorbar = True, xRange = (1919, 4800), yRange = (-21,21))
    for sensor_pos in [0.04,0.05,0.06,0.07,0.08,0.09,0.34,0.35,0.36,0.37,0.38,0.39]:
        meas.data.ix[sensor_pos] = meas.data.ix[sensor_pos]/3
        meas.plot2D(meas.data, "Time (s)", "Pos (m)", title = os.path.basename(file), clickEvent= True, showImages = True, outFile = outdir + os.path.basename(file) + ".png" ,colorbar = True,  xRange = (0.01 , 0.014), yRange = None)
        meas.plot2D(abs(meas.dataFFT), "Frequency (Hz)", "Wavenumber (1/m)", title = os.path.basename(file), clickEvent= True, showImages = None, outFile = outdir + os.path.basename(file) + ".png" ,colorbar = True, xRange = (2000, 4000), yRange = (-20,20))
        meas.plot1D(meas.data.ix[0.20], "Time [s]", "Amplitude [g]", title ="Circumferential stepped femur phantom ID122_circumferential_7.5 mm (e.g. sensor position 0.20 m) filtered by Hamming function", showImages = True)
        meas.plot1D(abs(meas.dataFFT).ix[7.1428571429999996], "Frequency [Hz]", "Amplitude [g]", title = "Circumferential stepped femur phantom ID122_circumferential_7.5 mm (e.g. wavenumber 7 1/m) filtered by Hamming function", showImages = True, outFile = None, xRange = (1500, 4500))
        meas.calcStep(meas.dataFFT.columns)
        meas.findNearestIndex(meas.dataFFT.index, 7.428)
        meas.dataFFT.index[meas.findNearestIndex(meas.dataFFT.index,7)]
        plt.close()
    freq = meas.findNearestIndex(meas.dataFFT.columns, 3000)
    meas.plot1D(abs(meas.dataFFT)[meas.dataFFT.columns[freq]], "Wavenumber [1/m]", "Amplitude [g]", title = "Circumferential stepped femur phantom ID122_circumferential_7.5 mm (e.g. frequency 3000 Hz) filtered by Hamming function", showImages = True, outFile = True, xRange = (0,15))

def assess_invers_wave_length(file, wn):
    
        """
        Assessing the axial and radial Phantoms each time with the 2DFFTSinc(), defining windows for the wavenumber and for the frequency
        """   
    
   
        outdir = "1Dplot_Freq3000_range_invers_wavelength."
        meas = bsmdMeasurement.bsmdMeasurement(xFFTNr = 20000, yFFTNr = 350)
#  for file in rPhantomsFiles:
        print(file)
        meas.readStepPhantomXMLData(file)
       
        algo = bsmdAlgorithm.PhaseVelocity_2DFFTGauss()
        result = algo.phaseVelocity(meas, freq = (2950,3050), wavenumber = wn, debug_plot = True, sumOfSquares_threshold = 1)

        freq = meas.findNearestIndex(meas.dataFFT.columns, 3000)
        meas.plot1D(abs(meas.dataFFT)[meas.dataFFT.columns[freq]], "Inverse Wavelength (1/m)", "Amplitude (g)", title = None, showImages = None, outFile = True, xRange = (0,20))           
        

               
def safe_window_to_phantom(fileList, excDic, std):
    """
        fileList -> list of strings; containing the absolute paths to the files to be analysed
        std -> tuple of numbers; standard value for the window
        excDic - dictionary with exceptions,
    """
    for file in fileList:
        f = os.path.basename(file)
        if f not in excDic.keys():  #then it is not exception (for which a value already exists), so set the value to standard
            excDic[f] = std
    return excDic
    
#    stdDic = {file:std for file in fileList}    #full of standards for each of the files
#    for excFile, excVal in excDic.items():
#        stdDic[excFile] =excVal 
#        
#    return excDic
   

def save3000HzAnalysisResultsToFile(algo, outFile, measFiles, exceptionfiles, standardWindow, analysis_freq = 3000, window = True):

    """
    Creating a text-file .txt for the measerument data at a frequency of 3000Hz    
    Creating a Dataframe (DataFrame_result), which contains the measurement data: velocity, averageSS, upper and lower velocity for a certain frequency area 
    """
    
    Windows = safe_window_to_phantom(measFiles, exceptionfiles, standardWindow)

    meas = bsmdMeasurement.bsmdMeasurement(xFFTNr = 20000, yFFTNr = 350)
    text_file =open(outFile, "w")     
    text_file.write("filename; phase_velocity; lower; upper; avgSS; frequency\n") #write header
    DataFrame_result= pd.DataFrame()   
    for file in measFiles:
        print("Analyzing file: {}".format(file))
        meas.readStepPhantomXMLData(file)
        if window is True:
            w = np.hamming(400)
            meas.data = meas.movingTimeFilter(w, starttime = 0.011198275, speed = 479.98464049150425, showImages = False)
            meas.plot2D_all("Filtered measurement", clickEvent=True, showImages = False)
        plt.close()
        result = algo.phaseVelocity(meas, freq = (2950,3050), wavenumber = Windows[os.path.basename(file)], debug_plot = False, sumOfSquares_threshold = 1)    #make sure to analyse all datafiles, even if they have crappy fittings
        frequency = meas.dataFFT.columns[meas.findNearestIndex(meas.dataFFT.columns, analysis_freq)]
        text_file.write("{};{};{};{};{};{}\n".format(os.path.basename(file), result.loc["velocity", frequency], result.loc["lower", frequency], result.loc["upper", frequency], result.loc["averageSS", frequency], frequency))
        T_result= result.T
        T_result["filename"] = os.path.basename(file)
        T_result["frequency"]= T_result.index
        T_result= T_result.reset_index(drop=True)
        DataFrame_result= pd.concat([DataFrame_result, T_result])
        
    text_file.close()
    DataFrame_result["specimen"] = DataFrame_result["filename"].apply(lambda a: a[7:])
    DataFrame_result["point"]= DataFrame_result["filename"].apply(lambda b: b[5:6])
    DataFrame_result["series"]= DataFrame_result["filename"].apply(lambda b: b[5:6])
    DataFrame_result["specimen_value"]= DataFrame_result["filename"].apply(lambda c: c[8:10])
    DataFrame_result["sensor_point"]= DataFrame_result["filename"].apply(lambda c: c[6:7]) #reliability
#    print(DataFrame_result["point"])

    return DataFrame_result

    
def plotErrorbar(fullData, specimen_kind):
    
    """Plotting the result of the velocity against the frequency, to get first view about the measurement data
    """
    
    specimen = set(fullData["specimen"])
    femur_phantoms_radial =["ID120_circumferential_5.5","ID116_circumferential_1.5","ID122_circumferential_7.5", "ID121_circumferential_6.5","ID119_circumferential_4.5","ID118_circumferential_3.5" ,"ID117_circumferential_2.5"]
    femur_phantoms_axial=["ID123_longitudinal_2.5","ID127_longitudinal_6.5","ID126_longitudinal_5.5","ID116_longitudinal_1.5","ID125_longitudinal_4.5","ID124_longitudinal_3.5","ID128_longitudinal_7.5"]
    for (spec), femur in zip(specimen, cycle(femur_phantoms_axial)): #plotting results for all 6 points for specimen
        print("Plotting results for all points for specimen: {}".format(spec))
        fig, axes = plt.subplots()
        plt.xlabel(r"Frequency $[Hz]$", fontsize=22)
        plt.xlim(2950,3050)
        plt.tick_params(axis="both", which="major", labelsize=20)
        plt.tick_params(axis="both", which="minor", labelsize=20)

        plt.ylabel(r"Phase velocity $[\frac{m}{s}]$", fontsize=22)
        plt.grid()
        plt.title("Femur phantom: {} mm \n\n Phase velocity of the 6 measured sensor points \n".format(femur), fontsize=20, fontweight="bold")
        linestyles = ["-", "--"]
        col=["black","white"]
        marker=["s","o","d"]
        for (point_i),marks, line, cols in zip(range(1,7), cycle(marker),cycle(linestyles), cycle(col)): #for-loop for plotting a specimen for all points
            sample = fullData[fullData["filename"]=="point{}_{}".format(point_i, spec)]
            plt.plot(sample["frequency"], sample["velocity"], linestyle=line, linewidth=2.0, marker= marks, color="black" ,markerfacecolor=cols, markeredgecolor="black", markeredgewidth=1.0, markersize=8.0 ,label = "Sensor point {}".format(str(point_i)))
        plt.legend(loc="center left", bbox_to_anchor=(0.95,0.6), fancybox=True, shadow=True, ncol=1, fontsize=20)
        fig.set_size_inches(24,13, forward=True)
        plt.savefig("Velocity_vs_frequency_2950-3050_all_7_points {}".format(spec) + ".png", dpi=300)
        plt.close()

def plotGroups(fullData, analysis_freq, specimen):
    
    """Plotting the result of the velocity against the points of the specimens at 3000 Hz
    """
    freq_set = fullData["frequency"].unique() #all unique values of frequency
    actual_analysis_freq = freq_set[bsmdMeasurement.bsmdMeasurement.findNearestIndex(freq_set, analysis_freq)] #we would like to analyse at "analysis_freq" but that frequency might not be available, so pick the nearest available frequency value
    logging.info("Analysis could not be performed at {} Hz, and was performed at {} instead.".format(analysis_freq, actual_analysis_freq))
    fullData_analysis_freq = fullData[fullData["frequency"]== actual_analysis_freq]
    wb=pd.read_excel("Z:\Student\Danun\Python-svnwc\Manual_measured_point_of_radial_femur_phantoms.xlsx")
    c= pd.DataFrame(wb, columns=["Ix", "Local_thickness"])
    c.index=c["Ix"]
    c.drop(["Ix"], inplace=True, axis=1)
    all_result=pd.concat([fullData_analysis_freq,c], axis=1, join_axes=[fullData_analysis_freq.index])
#    femur_phantoms_radial =["ID116_circumferential_1.5","ID117_circumferential_2.5","ID118_circumferential_3.5","ID119_circumferential_4.5","ID120_circumferential_5.5","ID121_circumferential_6.5","ID122_circumferential_7.5"]
    femur_phantoms_axial=["ID116_longitudinal_1.5","ID123_longitudinal_2.5","ID124_longitudinal_3.5","ID125_longitudinal_4.5","ID126_longitudinal_5.5","ID127_longitudinal_6.5","ID128_longitudinal_7.5"]    
    fig, axes =plt.subplots()
    plt.tick_params(axis="both", which="major", labelsize=20)
    plt.tick_params(axis="both", which="minor", labelsize=20)    
    linestyles = ["-", "--"]
    col=["white","black"]
    marker=["s","o","d"]
    plt.grid()
    plt.xlim(0.8,7.0)
    plt.ylim(270,550)
    plt.ylabel(r"Phase velocity $[\frac{m}{s}]$", fontsize=22)
    plt.xlabel(r"Sensor point $[-]$", fontsize=22)
    plt.title("Phase velocitiy of the {} femur phantoms at a frequency of {} Hz: \n\n Compared by the sensor points \n".format(specimen, analysis_freq), fontsize=20, fontweight="bold")
    for (i, group),femur, marks, line, cols in zip(all_result.groupby("specimen"),cycle(femur_phantoms_axial), cycle(marker),cycle(linestyles), cycle(col)):
#    for i, group in fullData_3000.groupby("specimen"):
        print(i)
        axes.plot(group.point, group.velocity,linestyle=line, marker= marks, color= "black", linewidth=2.0, ms=8.0, markerfacecolor=cols, markeredgecolor="black", label=str(femur))  
    axes.legend(loc="center left", bbox_to_anchor=(0.89,0.6), fancybox=True, shadow=True, ncol=1, fontsize=18)
    fig.set_size_inches(24,15, forward=True)
    plt.savefig("Velocity_vs_points_{}_specimen_{}".format(specimen, analysis_freq) + ".png", dpi=300)

def plotGroups2(fullData, analysis_freq, specimen):
    freq_set = fullData["frequency"].unique() #all unique values of frequency
    actual_analysis_freq = freq_set[bsmdMeasurement.bsmdMeasurement.findNearestIndex(freq_set, analysis_freq)] #we would like to analyse at "analysis_freq" but that frequency might not be available, so pick the nearest available frequency value
    logging.info("Analysis could not be performed at {} Hz, and was performed at {} instead.".format(analysis_freq, actual_analysis_freq))
    fullData_analysis_freq = fullData[fullData["frequency"]== actual_analysis_freq]
    wb=pd.read_excel("Z:\Student\Danun\Python-svnwc\Manual_measured_point_of_radial_femur_phantoms.xlsx")
    c= pd.DataFrame(wb, columns=["Ix", "Local_thickness"])
    c.index=c["Ix"]
    c.drop(["Ix"], inplace=True, axis=1)
    all_result=pd.concat([fullData_analysis_freq,c], axis=1, join_axes=[fullData_analysis_freq.index])
    
   
    fig, axes =plt.subplots()
    plt.tick_params(axis="both", which="major", labelsize=20)
    plt.tick_params(axis="both", which="minor", labelsize=20)    
    linestyles = ["-", "--"]
    col=["white","black"]
    marker=["s","o","d"]
    plt.grid()


    plt.ylabel(r"Phase velocity $[\frac{m}{s}]$", fontsize=22)
    plt.xlabel(r"Thickness $[mm]$", fontsize=22)
    plt.title("Phase velocitiy of the {} femur phantoms at a frequency of {} Hz:\n\n Compared by the local thickness of all 6 sensor points \n".format(specimen, analysis_freq), fontsize=20, fontweight="bold")
    for (i, group),marks, line, cols in zip(all_result.groupby("point"), cycle(marker),cycle(linestyles), cycle(col)):       
#    for i, group in fullData_3000.groupby("point"):
        axes.plot(group.Local_thickness, group.velocity,linestyle=line, marker= marks, color= "black", linewidth=2.0, ms=8.0, markerfacecolor=cols, markeredgecolor="black", label= "Sensor point {}".format(str(i)))
    axes.legend(loc="center left", bbox_to_anchor=(0.95,0.6), fancybox=True, shadow=True, ncol=1, fontsize=18)
    fig.set_size_inches(24,15, forward=True)
    plt.savefig("velocity_vs_specimen_local_thickness_{}_specimen_{}".format(specimen, analysis_freq) + ".png", dpi=300)
    
    
    
    
    
def plot_boxplot(fullData,analysis_freq, specimen):

    freq_set = fullData["frequency"].unique() #all unique values of frequency
    actual_analysis_freq = freq_set[bsmdMeasurement.bsmdMeasurement.findNearestIndex(freq_set, analysis_freq)] #we would like to analyse at "analysis_freq" but that frequency might not be available, so pick the nearest available frequency value
    logging.info("Analysis could not be performed at {} Hz, and was performed at {} instead.".format(analysis_freq, actual_analysis_freq))
    fullData_analysis_freq = fullData[fullData["frequency"]== actual_analysis_freq]
    wb=pd.read_excel("Z:\Student\Danun\Python-svnwc\Manual_measured_point_of_radial_femur_phantoms.xlsx")
    c= pd.DataFrame(wb, columns=["Ix", "Local_thickness"])
    c.index=c["Ix"]
    c.drop(["Ix"], inplace=True, axis=1)
    all_result=pd.concat([fullData_analysis_freq,c], axis=1, join_axes=[fullData_analysis_freq.index])

    fig, axes =plt.subplots()
    xerr= all_result.groupby("specimen_value").mean()

    axes.errorbar(xerr["Local_thickness"], xerr["velocity"],yerr["velocity"], ecolor="black", fmt="--o", linewidth=2, capsize=12, mfc="white", markeredgecolor="black", ms=8.0, color="black")
    plt.title("Mean phase velocitiy of the {} femur phantoms at a frequency of {} Hz:\n\n Compared by the mean thickness across the local thickness of all 6 sensor points \n".format(specimen, analysis_freq), fontsize=20, fontweight="bold")    
    plt.grid(True, which="major", axis="both", color="black", linewidth=1.5)
    plt.tick_params(axis="both", which="major", labelsize=20)
    plt.tick_params(axis="both", which="minor", labelsize=20)    
    plt.ylabel(r"Phase velocity $[\frac{m}{s}]$", fontsize=22)
    plt.xlabel(r"Thickness $[mm]$", fontsize=22)
    fig.set_size_inches(24,15, forward=True)        
#    plt.savefig("Boxplot_specimen velocity vs. {} at Frequency of {} Hz".format(specimen,analysis_freq) + ".png", dpi=300)
 

def ANOVA_specimen(outFile, ax_fullData, rad_fullData, analysis_freq):
    freq_set = ax_fullData["frequency"].unique()
    actual_analysis_freq = freq_set[bsmdMeasurement.bsmdMeasurement.findNearestIndex(freq_set, analysis_freq)] #we would like to analyse at "analysis_freq" but that frequency might not be available, so pick the nearest available frequency value
    logging.info("Analysis could not be performed at {} Hz, and was performed at {} instead.".format(analysis_freq, actual_analysis_freq))
    rad_fullData_analysis_freq = ax_fullData[ax_fullData["frequency"]== actual_analysis_freq]
    wb=pd.read_excel("Z:\Student\Danun\Python-svnwc\Manual_measured_point_of_radial_femur_phantoms.xlsx")
    c= pd.DataFrame(wb, columns=["Ix", "Local_thickness"])
    c.index=c["Ix"]
    c.drop(["Ix"], inplace=True, axis=1)
    all_result=pd.concat([rad_fullData_analysis_freq,c], axis=1, join_axes=[rad_fullData_analysis_freq.index])
    print(all_result)
    ANOVA_samples=[]    
    for spec in rad_fullData_analysis_freq["specimen"].unique():
        ANOVA_result_samples = rad_fullData_analysis_freq[rad_fullData_analysis_freq["specimen"]== spec]["velocity"]
        ANOVA_samples.append(ANOVA_result_samples)
    
    F_val, p_val = stats.f_oneway(*ANOVA_samples)#homogenität, normalverteilt, unabhängige samples
    print("\n\n F_val=", F_val, "\n\n p_val=", p_val)
    z_val, p_val = stats.kruskal(*ANOVA_samples) #keine normalverteilung notwendig-testen
    print("\n\n Z_val= ", z_val, "\n\n p_val= ", p_val)
    tukey = pairwise_tukeyhsd(endog= rad_fullData_analysis_freq["velocity"], groups=rad_fullData_analysis_freq["specimen"], alpha=0.05) #pairity-test for undependet samples
    tukey.plot_simultaneous()  
    print("\n\n ", tukey.summary())
    
def calculat_my_Error(fullData,analysis_freq, specimen):
    
    freq_set = fullData["frequency"].unique() #all unique values of frequency
    actual_analysis_freq = freq_set[bsmdMeasurement.bsmdMeasurement.findNearestIndex(freq_set, analysis_freq)] #we would like to analyse at "analysis_freq" but that frequency might not be available, so pick the nearest available frequency value
    logging.info("Analysis could not be performed at {} Hz, and was performed at {} instead.".format(analysis_freq, actual_analysis_freq))
    fullData_analysis_freq = fullData[fullData["frequency"]== actual_analysis_freq]

    print(fullData_analysis_freq)

    fig, axes =plt.subplots()
    xerr= fullData_analysis_freq.groupby("sensor_point").mean()
    xerr.reset_index(inplace = True)
    yerr= fullData_analysis_freq.groupby("sensor_point").std()
    print(xerr,yerr)
#
    axes.errorbar(xerr["sensor_point"], xerr["velocity"],yerr["velocity"], ecolor="black", fmt="--o", linewidth=2, capsize=12, mfc="white", markeredgecolor="black", ms=8.0, color="black")
    plt.title("Repeatability test of the BSMD setup:\n\n Measuring femur phantom ID119_circumferential_4.5 mm by 6 \n", fontsize=20, fontweight="bold")    
    plt.grid(True, which="major", axis="both", color="black", linewidth=1.5)
    plt.tick_params(axis="both", which="major", labelsize=20)
    plt.tick_params(axis="both", which="minor", labelsize=20)    
    plt.ylabel(r"Phase velocity $[\frac{m}{s}]$", fontsize=22)
    plt.xlabel(r"Sensor point $[-]$", fontsize=22)
    plt.xlim(0,7)
    plt.ylim(400,450,5)
    fig.set_size_inches(24,15, forward=True)        
    plt.savefig("Reliability_specimen velocity vs. {} at Frequency of {} Hz".format(specimen,analysis_freq) + ".png", dpi=300)

  
    

    

       
if __name__ == "__main__":
###    radialExceptionDic = {"point1_r15mm.xml":(6.0, 8.0),"point1_r25mm.xml":(5.0, 8.0), "point2_r15mm.xml":(5.0, 7.5),"point3_r65mm.xml":(5.0, 9.0),"point2_r25mm.xml":(4.0, 11.5),"point3_r25mm.xml":(5.0, 8.0),"point3_r35mm.xml":(6.0, 8.5),"point4_r15mm.xml":(9.0, 11.0),"point4_r25mm.xml":(7.0, 9.0),"point5_r15mm.xml":(5.0, 7.5),"point5_r25mm.xml":(4.0, 6.0),"point6_r15mm.xml":(8.0, 11.0),"point6_r25mm.xml":(6.0, 8.0),"point6_r45mm.xml":(6.0, 8.5)}
###    axialExceptionDic= {"point1_a15mm.xml":(6.0, 8.0),"point1_a25mm.xml":(4.0,7.0),"point2_a15mm.xml":(5.0, 7.5),"point3_a25mm.xml":(5.0,8.0),"point4_a25mm.xml":(7.0,10.0),"point4_a15mm.xml":(9.0, 11.0),"point5_a25mm.xml":(4.0,7.0),"point3_a35mm.xml":(6.0,7.5),"point5_a15mm.xml":(5.0, 7.5),"point5_a35mm.xml":(6.5,9.0),"point6_a15mm.xml":(8.0, 11.0),"point3_a45mm.xml":(5.0,8.0), "point4_a45mm.xml":(7.0,9.0), "point5_a45mm.xml":(6.0,8.5), "point6_a55mm.xml":(6.0,8.5),"point6_a25mm.xml":(6.0,8.0),"point6_a35mm.xml":(7.0,10.0)}
    radialExceptionDicGauss = {"point1_r15mm.xml":(6.0, 8.0),"point1_r25mm.xml":(9.0, 11.0), "point2_r15mm.xml":(5.0, 7.5),"point3_r65mm.xml":(5.0, 9.0),"point2_r25mm.xml":(4.0, 11.5),"point5_r25mm.xml":(7.0, 10.0),"point3_r25mm.xml":(9.0, 11.0),"point3_r35mm.xml":(6.0, 8.5),"point4_r15mm.xml":(5.0, 7.0),"point4_r25mm.xml":(7.0, 9.0),"point5_r15mm.xml":(5.0, 7.5),"point6_r15mm.xml":(9.0, 11.0),"point6_r25mm.xml":(6.0, 9.0),"point6_r45mm.xml":(6.0, 8.5)}
#    axialExceptionDicGauss= {"point1_a15mm.xml":(6.0, 8.0),"point1_a25mm.xml":(9.0,11.0),"point2_a15mm.xml":(5.0, 7.5),"point2_a25mm.xml":(5.0, 8.0),"point3_a25mm.xml":(9.0,11.0),"point4_a25mm.xml":(7.0,10.0),"point4_a15mm.xml":(5.0, 7.0),"point5_a25mm.xml":(9.0,11.0),"point3_a35mm.xml":(6.0,7.5),"point5_a15mm.xml":(5.0, 7.5),"point5_a35mm.xml":(6.5,9.0),"point6_a15mm.xml":(8.0, 11.0),"point3_a45mm.xml":(5.0,8.0), "point4_a45mm.xml":(7.0,9.0), "point5_a45mm.xml":(6.0,8.5), "point6_a55mm.xml":(6.0,8.5),"point6_a25mm.xml":(6.0,8.0),"point6_a35mm.xml":(7.0,10.0)}
#    errorExcemptionDicGauss= {}    
    ANALYSIS_FREQ = 3000
    algo = bsmdAlgorithm.PhaseVelocity_2DFFTGauss()  #gauss
    radial_all_results = save3000HzAnalysisResultsToFile(algo, "radial_phantom.txt", rPhantomsFiles, radialExceptionDicGauss, (6.0, 9.0), analysis_freq = ANALYSIS_FREQ)
#   axial_all_results =  save3000HzAnalysisResultsToFile(algo, "axial_phantoms.txt", aPhantomsFiles, axialExceptionDicGauss, (6.0,9.0), analysis_freq = ANALYSIS_FREQ)
#   error_all_results =  save3000HzAnalysisResultsToFile(algo, "measure_error.txt", errorFiles, errorExcemptionDicGauss, (6.0,9.0), analysis_freq = ANALYSIS_FREQ)
#    plotErrorbar(axial_all_results, specimen_kind="Longitudinal stepped")
#    plotGroups(axial_all_results, ANALYSIS_FREQ, specimen="longitudinal stepped")    
#    plotGroups2(axial_all_results, ANALYSIS_FREQ, specimen="longitudinal stepped")
    plot_boxplot(radial_all_results, ANALYSIS_FREQ, specimen="longitudinal stepped")

#    ANOVA_specimen("statsitcs.txt", axial_all_results, radial_all_results, ANALYSIS_FREQ)
#    calculat_my_Error(error_all_results, ANALYSIS_FREQ, specimen="Circumferential stepped")
    

    
#    first_viewing()
#    assess_influence_of_standardization()
#    assess_invers_wave_length("Z:\Student\Danun\measurements-svn\cortical_thickness-ashraf\Phantom_radial_1.5-7.5mm\point6_r15mm.xml", (9.0,11.0))



    
