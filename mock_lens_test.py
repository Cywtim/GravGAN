# import standard python libraries
import numpy as np
import scipy
import os
import imageio
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from lenstronomy.LightModel.Profiles.shapelets import ShapeletSet
import lenstronomy.Util.util as util
import lenstronomy.Util.image_util as image_util

# define the lens model of the main deflector
main_halo_type = 'SIE'  # You have many other possibilities available. Check out the SinglePlane class!
kwargs_lens_main = {'theta_E': 1., 'e1': 0.1, 'e2': 0, 'center_x': 0, 'center_y': 0}
kwargs_shear = {'gamma1': 0.05, 'gamma2': 0}
lens_model_list = [main_halo_type, 'SHEAR']
kwargs_lens_list = [kwargs_lens_main, kwargs_shear]

################################################################
# now we define a LensModel class of all the lens models combined
from lenstronomy.LensModel.lens_model import LensModel
lensModel = LensModel(lens_model_list)
# we set up a grid in coordinates and evaluate basic lensing quantities on it
x_grid, y_grid = util.make_grid(numPix=100, deltapix=0.05)
kappa = lensModel.kappa(x_grid, y_grid, kwargs_lens_list)
# we make a 2d array out of the 1d grid points
kappa = util.array2image(kappa)
# and plot the convergence of the lens model
plt.matshow(np.log10(kappa), origin='lower')
plt.show()
###################################################
filename = os.path.join("img_shrink_256/image_000006.png")
data = imageio.imread(filename, as_gray=True, pilmode=None)
# we now degrate the pixel resoluton by a factor.
# This reduces the data volume and increases the spead of the Shapelet decomposition
factor = 25  # lower resolution of image with a given factor
numPix_large = 256
x, y = util.make_grid(numPix=numPix_large, deltapix=1)  # make a coordinate grid
# we turn the image in a single 1d array
image_1d = util.image2array(data)  # map 2d image in 1d data array
#-------------------------------------------------------#
# we define the shapelet basis set we want the image to decompose in
n_max = 150  # choice of number of shapelet basis functions, 150 is a high resolution number, but takes long
beta = 10  # shapelet scale parameter (in units of resized pixels)

shapeletSet = ShapeletSet()
# decompose image and return the shapelet coefficients
coeff_data = shapeletSet.decomposition(image_1d, x, y, n_max, beta, 1., center_x=0, center_y=0)

############################################################
# we define a very high resolution grid for the ray-tracing (needs to be checked to be accurate enough!)
numPix = 100  # number of pixels (low res of data)
deltaPix = 0.05  # pixel size (low res of data)
high_res_factor = 3  # higher resolution factor (per axis)
# make the high resolution grid
theta_x_high_res, theta_y_high_res = util.make_grid(numPix=numPix*high_res_factor, deltapix=deltaPix/high_res_factor)
# ray-shoot the image plane coordinates (angles) to the source plane (angles)
beta_x_high_res, beta_y_high_res = lensModel.ray_shooting(theta_x_high_res, theta_y_high_res, kwargs=kwargs_lens_list)

# now we do the same as in Section 2, we just evaluate the shapelet functions in the new coordinate system of the source plane
# Attention, now the units are not pixels but angles! So we have to define the size and position.
# This is simply by chosing a beta (Gaussian width of the Shapelets) and a new center

source_lensed = shapeletSet.function(beta_x_high_res, beta_y_high_res, coeff_data, n_max, beta=.05, center_x=0.5, center_y=0)
# and turn the 1d vector back into a 2d array
source_lensed = util.array2image(source_lensed)  # map 1d data vector in 2d image

plt.imshow(source_lensed)
plt.show()