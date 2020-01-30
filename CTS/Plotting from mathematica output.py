"""
Scatterplot Matrix
==================

_thumb: .5, .43
"""
import seaborn as sns
import csv
import matplotlib.pyplot as plt
import numpy as np
import operator
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import bsmd.algorithm.bsmdAlgorithmRestructured as bsmdAlgorithm
# import bsmd.algorithm.bsmdAlgorithm as bsmdAlgorithm
import bsmd.measurement.bsmdMeasurement as bsmdMeasurement

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
import argparse
import pickle
import fnmatch
from pandas.tools.plotting import scatter_matrix
from  itertools import combinations
from pandas.tools.plotting import table
import seaborn as sns
from matplotlib import pylab
import csv


'theoretical plots'
# csv_file = open("T:\mathematica-svn\phantomResults\id_tubetheory_mohit", "r")
# reader = csv.reader(csv_file)
# x = []
# y = []
# for line in reader:
#     t = int(line[0])
#     p = line[1]
#     x.append(t)
#     y.append(p)
# x = np.array(x) / 2000
# y = np.array(y)
# plt.plot(x, y)
#
# csv_file = open("T:\mathematica-svn\phantomResults\id_tubetheory_mohit_perfectfit_temp", "r")
# reader = csv.reader(csv_file)
# x = []
# y = []
# for line in reader:
#     t = int(line[0])
#     p = line[1]
#     x.append(t)
#     y.append(p)
# x = np.array(x) / 2000
# y = np.array(y)
# plt.plot(x, y)
#
# csv_file = open("T:\mathematica-svn\phantomResults\id_tubetheory_mohit_bone", "r")
# reader = csv.reader(csv_file)
# x = []
# y = []
# for line in reader:
#     t = int(line[0])
#     p = line[1]
#     x.append(t)
#     y.append(p)
# x = np.array(x) / 2000
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

# csv_file = open("T:\mathematica-svn\phantomResults\id_tubetheory_mohit_bone_fullrange", "r")
# reader = csv.reader(csv_file)
# x = []
# y = []
# for line in reader:
#     t = int(line[0])
#     p = line[1]
#     x.append(t)
#     y.append(p)
# x = np.array(x)
# y = np.array(y)
# plt.plot(x, y)
#
# # plt.ylim(2, 800)
# # plt.xlim(0,8)
# plt.show()



# points = [[1,4,5]]
# next = [1,4,5]
# for i in [1,2,3,4,5,6]:
#     next = [x+6 for x in next]
#     points.append(next)
# flat_list = [item for sublist in points for item in sublist]
# print(flat_list)

'String output for mathematica input of thickness-radius pair combination of 42 points'
# parameter_list = pd.read_excel(r'P:\Student\Mohit\Mathematica svn\a-h values for tube plot.xlsx')
# parameter_list.index = parameter_list['index']
# string_collect = []
# for index in np.arange(1,43):
#
#     a = parameter_list['a'][index]/1000
#     h = parameter_list['h'][index]/1000
#     string = "{"+"a->{}, h->{}".format(a,h)+"},"
#     # string_collect.append(string)
#     print(string)
# print(string_collect)

