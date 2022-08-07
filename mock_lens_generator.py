"""

lens the galaxy by lenstronomy
This file defining a class that receives a source of galaxy
then output a distorted image

CHENGYI

"""
import cv2
import scipy
import imageio
import numpy as np
import scipy.signal as signal
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt

import lenstronomy.Util.util as util
import lenstronomy.Util.image_util as image_util
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.LightModel.Profiles.shapelets import ShapeletSet
from lenstronomy.ImSim.image_model import ImageModel
import lenstronomy.Util.kernel_util as kernel_util
from lenstronomy.LensModel.lens_model import LensModel


class Mock_Lens_Generator:

    def __init__(self, main_halo_type, kwargs_lens_main, kwargs_shear, shear="SHEAR",
                 subhalo_type=None, kwargs_subhalo=None):


        #define the main lens model
        if shear == "SHEAR":
            self.lens_model_list = [main_halo_type, shear]
            self.kwargs_lens_list = [kwargs_lens_main, kwargs_shear]
        else:
            self.lens_model_list = [main_halo_type]
            self.kwargs_lens_list = [kwargs_lens_main]

        #if there is subhalo beside the main lens
        if (subhalo_type != None) and (kwargs_subhalo != None):

            if len(subhalo_type) != len(kwargs_subhalo):
                #Check the index of subhalo and kwargs of subhalo
                print("Error! the size of subhalo_type didn't coincide with kwargs_subhalo.")

            self.subhalo_type = subhalo_type # number of subhalos to be rendered
            for num in range(len(subhalo_type)):
                #add the subhalo into the lens model
                self.lens_model_list.append(subhalo_type[num])
                self.kwargs_lens_list.append(kwargs_subhalo[num])

        #build up the whole lens model
        self.lens_model = LensModel(self.lens_model_list)


    def shapelet_lens_image(self, file_path,
                            n_max=150,  # choice of number of shapelet basis functions, 150 is a high resolution number, but takes long
                            beta=10,  # shapelet scale parameter (in units of resized pixels)
                            numPix=100,  # number of pixels (low res of data)
                            deltaPix = 0.05, # pixel size (low res of data)
                            high_res_factor = 3,  # higher resolution factor (per axis)
                            plot_lens=False):

        # define the lens model of the main deflector in self

        ################################################################
        # now we define a LensModel class of all the lens models combined

        lensModel = LensModel(self.lens_model_list)
        # we set up a grid in coordinates and evaluate basic lensing quantities on it
        if plot_lens:
            x_grid, y_grid = util.make_grid(numPix=100, deltapix=0.05)
            kappa = lensModel.kappa(x_grid, y_grid, self.kwargs_lens_list)
            # we make a 2d array out of the 1d grid points
            kappa = util.array2image(kappa)
            # and plot the convergence of the lens model
            plt.matshow(np.log10(kappa), origin='lower')
            plt.show()
        ###################################################
        data = imageio.imread(file_path, as_gray=True, pilmode=None)
        # we now degrate the pixel resoluton by a factor.
        # This reduces the data volume and increases the spead of the Shapelet decomposition

        # numPix_large lower resolution of image with a given factor
        numPix_large = data.shape[0]
        x, y = util.make_grid(numPix=numPix_large, deltapix=1)  # make a coordinate grid

        # we turn the image in a single 1d array
        image_1d = util.image2array(data)  # map 2d image in 1d data array
        # -------------------------------------------------------#
        # we define the shapelet basis set we want the image to decompose in
        shapeletSet = ShapeletSet()
        # decompose image and return the shapelet coefficients
        coeff_data = shapeletSet.decomposition(image_1d, x, y, n_max, beta, 1., center_x=0, center_y=0)

        ###########################################################
        # we define a very high resolution grid for the ray-tracing (needs to be checked to be accurate enough!)
        # make the high resolution grid
        theta_x_high_res, theta_y_high_res = util.make_grid(numPix=numPix * high_res_factor,
                                                            deltapix=deltaPix / high_res_factor)
        # ray-shoot the image plane coordinates (angles) to the source plane (angles)
        beta_x_high_res, beta_y_high_res = lensModel.ray_shooting(theta_x_high_res, theta_y_high_res,
                                                                  kwargs=self.kwargs_lens_list)

        # now we do the same as in Section 2, we just evaluate the shapelet functions in the new coordinate system of the source plane
        # Attention, now the units are not pixels but angles! So we have to define the size and position.
        # This is simply by chosing a beta (Gaussian width of the Shapelets) and a new center

        source_lensed = shapeletSet.function(beta_x_high_res, beta_y_high_res, coeff_data, n_max, beta=.05,
                                             center_x=0.0, center_y=0.0)
        # and turn the 1d vector back into a 2d array
        source_lensed = util.array2image(source_lensed)  # map 1d data vector in 2d image

        return source_lensed

    def PSF_conv(self,source_lensed,PSF_path,high_res_factor=3):

        # import PSF file
        kernel = pyfits.getdata(PSF_path)

        # subsample PSF to the high_res_factor scaling
        kernel_subsampled = kernel_util.subgrid_kernel(kernel,
                                                       high_res_factor,
                                                       odd=True, num_iter=5)
        # we cut the PSF a tiny bit on the edges
        kernel_subsampled = kernel_util.cut_psf(psf_data=kernel_subsampled,
                                                psf_size=len(kernel_subsampled)
                                                         - 30 * high_res_factor)

        image_conv = signal.fftconvolve(source_lensed,
                                             kernel_subsampled, mode='same')

        return image_conv

    def down_sample(self,image_conv,high_res_factor=3):

        image_conv_resized = image_util.re_size(image_conv,
                                                high_res_factor)

        return image_conv_resized

    def add_poisson_noisy(self,image_conv_resized):

        exposure_time = 1  # the units are photons and therefore the exposure time is unity
        poisson = image_util.add_poisson(image_conv_resized, exp_time=exposure_time)

        # and we add here another Gaussian component uniform over the entire image (just because it's easy)
        background_rms = 10  # background rms
        bkg = image_util.add_background(image_conv_resized, sigma_bkd=background_rms)

        # and we add them all together
        image_noisy = image_conv_resized + poisson + bkg

        return image_noisy


def resize_fig(image, dsize, interpolation=cv2.INTER_LINEAR, method=cv2.IMREAD_GRAYSCALE, if_save=None):
    """
    This function is to resize the image into size of dsize
    :param image:
    :param dsize:
    :param interpolation:
    :param method:
    :param if_save:
    :return:
    """
    #image = cv2.imread(image_path, method)

    image = cv2.resize(image, dsize, interpolation=interpolation)

    if type(if_save) is not str:
        return image
    else:
        cv2.imwrite(if_save, image)


def nearest_random_pad(image_path, iter, edge=1, decay_rate=(0.9, 1.0), kennel=np.ones((3, 3))/9,
                        cv_method=cv2.IMREAD_GRAYSCALE, padding_method="edge"):
    """
    This function provide a method to add a smooth boundary for a image at a path of image_path
    :param image_path:
    :param iter:
    :param edge:
    :param decay_rate:
    :param kennel:
    :param cv_method:
    :param padding_method:
    :return:
    """
    image = cv2.imread(image_path, cv_method)

    padding = np.pad(image, ((edge, edge), (edge, edge)), padding_method)
    for i in range(iter):

        padding = cv2.filter2D(padding, -1, kennel)
        padding = padding * np.random.uniform(decay_rate[0], decay_rate[1], padding.shape) + \
                            np.random.randint(1, size=padding.shape)
        padding[edge:-edge, edge:-edge] = image
        image = padding
        padding = np.pad(padding, ((edge, edge), (edge, edge)), padding_method)

    return padding


def pad_boundary(image_path, edge_matrix=((10, 10),(10, 10)),
                 cv_method=cv2.IMREAD_GRAYSCALE, padding_method="minimum"):

    image = cv2.imread(image_path, cv_method)

    image = np.pad(image, edge_matrix, padding_method)

    return image
