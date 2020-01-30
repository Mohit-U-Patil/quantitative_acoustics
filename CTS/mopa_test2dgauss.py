# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 16:04:32 2017

@author: patilm
"""
import scipy.optimize as opt
import numpy as np
import pylab as plot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt


#define model function and pass independant variables x and y as a list"


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

for mu, sig in [(1.5, 1.5)]:
    plt.plot(gaussian(np.linspace(-4, 7, num=500), mu, sig))
outdir = r"D:\ETH\Work\Latest Download\Work from home\Radial"
# figure = plt.gcf()
plt.savefig(outdir + r"\1D-Gauss.eps", dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait')
plt.show()


def twoD_Gauss(base_data, amplitude, xo, yo, sigma_x, sigma_y):

    (x, y) = base_data
    a = 2/(2*sigma_x**2)
    c = 2/(2*sigma_y**2)
    g = amplitude*np.exp( - (a*((x-xo)**2) + c*((y-yo)**2)))
    return g.ravel()


x = np.linspace(0, 200, 201)
y = np.linspace(0, 200, 201)
x, y = np.meshgrid(x, y)


data = twoD_Gauss((x, y), 3, 100, 100, 20, 40)
plt.figure()
plt.imshow(data.reshape(201,201))
plt.colorbar()

data_noisy = data + 0.2*np.random.normal(size=data.shape)

initial_guess = (3,100,100,20,40)
popt, pcov = opt.curve_fit(twoD_Gauss, (x, y), data_noisy, p0=initial_guess)
print(pcov)
data_fitted = twoD_Gauss((x, y), *popt)

fig, ax = plt.subplots(1, 1)
ax.hold(True)
ax.imshow(data_noisy.reshape(201, 201), cmap=plt.cm.jet, origin='bottom',
    extent=(x.min(), x.max(), y.min(), y.max()))
ax.contour(x, y, data_fitted.reshape(201, 201), 8, colors='w')
plt.show()


def multivariate_Gaussian(base_data, amplitude, xo, yo, sigma_x, sigma_y, sigma_xy):
    '''give base data as a meshgrid tuple or simply use data_meshgrid(x,y) function with x and y arrays as arguments.
         see the example at the end of the scrip for tuple input'''
    try:
        (x, y) = base_data
        a = -1 / (2 * (((sigma_x * sigma_y) ** 2) - (sigma_xy ** 2)))
        b = (sigma_y * (x - xo)) ** 2
        c = 2 * sigma_xy * (x - xo) * (y - yo)
        d = (sigma_x * (y - yo)) ** 2
        g = amplitude * np.exp(a * (b - c + d))
        return g.ravel()

    except ZeroDivisionError as error:
        print(
            "Reconsider sigma_x, sigma_y and sigma_xy values. check that 'a' in the function doesn't have 0 in the denominator")
        raise error

x = np.arange(0, 5, 201)
y = np.arange(0, 5, 201)
x, y = np.meshgrid(x, y)

data = multivariate_Gaussian((x, y), 3, 2, 2, 1,1,0.5)

fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y = x,y
Z = data

ax.plot_surface(X, Y, Z, alpha=0.3)

ax.set_zlim(-4, 2)

cset = ax.contourf(X, Y, Z, zdir='z', offset=-3, cmap=cm.coolwarm)
# cset = ax.contourf(X, Y, Z, zdir='x', offset=-400, cmap=cm.coolwarm)
# cset = ax.contourf(X, Y, Z, zdir='y', offset=100, cmap=cm.coolwarm)
plt.show()
plt.close()

"""fig, ax = plt.subplots(1, 1)
ax.hold(True)
ax.imshow(data.reshape(201, 201), cmap=plt.cm.jet, origin='bottom',
    extent=(x.min(), x.max(), y.min(), y.max()))
#ax.contour(x, y, data_fitted.reshape(201, 201), 8, colors='w')
"""


data_noisy = data + 0.2*np.random.normal(size=data.shape)

initial_guess = (6, 5, 5, 2, 4,5)
popt, pcov = opt.curve_fit(multivariate_Gaussian, (x, y), data_noisy, p0=initial_guess)
data_fitted = multivariate_Gaussian((x, y), *popt)
amplitude, xo, yo, sigma_x, sigma_y, sigma_xy = popt
covariance = [[sigma_x**2,sigma_xy],[sigma_xy, sigma_y**2]]
eigenval, eigenvec = np.linalg.eig(covariance)
print("covariance matrix: {}".format(covariance))
print("Eigenvalues: {}".format(eigenval))
print("covariance matrix: {}".format(eigenvec))
fig, ax = plt.subplots(1, 1)
ax.hold(True)
ax.imshow(data_noisy.reshape(201, 201), cmap=plt.cm.jet, origin='bottom',
    extent=(x.min(), x.max(), y.min(), y.max()))
ax.contour(x, y, data_fitted.reshape(201, 201), 8, colors='w')
plt.show()
