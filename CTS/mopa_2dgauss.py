# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 16:04:32 2017

@author: patilm
"""
import scipy.optimize as opt
import numpy as np
import pylab as plt

class twoD_gaussian():
    @staticmethod
    def data_meshgrid(x, y):
        """takes in x and y arrays and returns tuple(x,y)"""
        if len(x) <= 0 or len(y) <= 0:
            raise TypeError
        x,y = np.meshgrid(x, y)
        base_data= (x,y)
        return(base_data)

    @staticmethod
# defining model function
    def twoD_Gaussian(base_data, amplitude, xo, yo, sigma_x, sigma_y):
        '''give base data as a meshgrid tuple or simply use data_meshgrid(x,y) function with x and y arrays as arguments.
         see the example at the end of the scrip for tuple input'''
        (x,y) = base_data
        a = 2 / (2 * sigma_x ** 2)
        c = 2 / (2 * sigma_y ** 2)
        g = amplitude * np.exp(- (a * ((x - xo) ** 2) + c * ((y - yo) ** 2)))
        return g.ravel()

    @classmethod
    def twoD_Gaussian_fit(cls, base_data,data_to_fit,initial_guess):
        '''base_data can be given as tuple or data_meshgrid(x,y) function with x and y arrays as arguments can be used
           data_to_fit is 1D data to which you want to find best fitting gaussian curve
           initial_guess= list(amplitude, xo, yo, sigma_x, sigma_y)
           This function finds best fitting gaussian curve and returns values for each(x,y)'''
        (x, y) = base_data
        popt, pcov = opt.curve_fit(cls.twoD_Gaussian, base_data, data_to_fit, p0=initial_guess)
        data_fitted = cls.twoD_Gaussian((x, y), *popt)
        fig, ax = plt.subplots(1, 1)
        ax.hold(True)
        ax.imshow(data_to_fit.reshape(x.shape), cmap=plt.cm.jet, origin='bottom',
                  extent=(x.min(), x.max(), y.min(), y.max()))
        ax.contour(x, y, data_fitted.reshape(x.shape), 8, colors='w')
        # plt.show()
        return(data_fitted,popt,pcov)



    @staticmethod
# defining second model function
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
            print( "Reconsider sigma_x, sigma_y and sigma_xy values. check that 'a' in the function doesn't have 0 in the denominator")
            raise error


    @classmethod
    def multivariate_Gaussian_fit(cls, base_data,data_to_fit,initial_guess):
        '''base_data can be given as tuple or data_meshgrid(x,y) function with x and y arrays as arguments can be used
           data_to_fit is 1D data to which you want to find best fitting gaussian curve
           initial_guess= list(amplitude, xo, yo, sigma_x, sigma_y, sigma_xy)
           This function finds best fitting gaussian curve and returns values for each(x,y)'''
        (x, y) = base_data
        popt, pcov = opt.curve_fit(cls.multivariate_Gaussian, base_data, data_to_fit, p0=initial_guess)
        data_fitted = cls.multivariate_Gaussian((x, y), *popt)
        fig, ax = plt.subplots(1, 1)
        ax.hold(True)
        ax.imshow(data_to_fit.reshape(x.shape), cmap=plt.cm.jet, origin='bottom',
              extent=(x.min(), x.max(), y.min(), y.max()))
        ax.contour(x, y, data_fitted.reshape(x.shape), 8, colors='w')
        plt.show()
        amplitude, xo, yo, sigma_x, sigma_y, sigma_xy = popt
        covariance_matrix = [[sigma_x ** 2, sigma_xy], [sigma_xy, sigma_y ** 2]]
        eigenval, eigenvec = np.linalg.eig(covariance_matrix)
        print("covariance matrix: {}".format(covariance_matrix))
        print("Eigenvalues: {}".format(eigenval))
        print("eigenvec: {}".format(eigenvec))
        return(data_fitted,popt,covariance_matrix,eigenval, eigenvec)

#--------------------------------------------------------------------------------------------

##for tilted gauss
'''#define model function and pass independant variables x and y as a list
def twoD_Gaussian((x,y), amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    return offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))

# Create x and y indices
x = np.linspace(0, 200, 201)
y = np.linspace(0, 200, 201)
x,y = np.meshgrid(x, y)

#create data
data = twoD_Gaussian((x, y), 3, 100, 100, 20, 40, 0, 10)

# plot twoD_Gaussian data generated above
plt.figure()
plt.imshow(data)
plt.colorbar()

# add some noise to the data and try to fit the data generated beforehand
initial_guess = (3,100,100,20,40,0,10)

data_noisy = data + 0.2*np.random.normal(size=len(x))

popt, pcov = opt.curve_fit(twoD_Gaussian, (x,y), data_noisy, p0 = initial_guess)
'''