"""

lens the galaxy by lenstronomy
This file defining a class that receives a source of galaxy
then output a distorted image

CHENGYI

"""
import scipy
import imageio
import scipy.signal as signal
import astropy.io.fits as pyfits

import lenstronomy.Util.util as util
import lenstronomy.Util.image_util as image_util
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.LightModel.Profiles.shapelets import ShapeletSet
import lenstronomy.Util.kernel_util as kernel_util


class Mock_Lens_Generator:

    def __init__(self,main_halo_type,kwargs_lens_main,kwargs_shear,shear="SHEAR",
                 subhalo_type=None,kwargs_subhalo=None):


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


    def shapelet_lens_image(self, source_image_path, factor=10,
                            gaussion_sigma=5, gaussion_mode='nearest', gaussion_truncate=6,
                            basis_n_max=100, scale_beta=10,coeff_center_x=0, coeff_center_y=0,
                            recon_center_x=0, recon_center_y=0,
                            numPix=100, deltaPix=0.05, high_res_factor=3,
                            lens_beta=0.05, lens_center_x=0.2, lens_center_y=0,
                            light_model_list=[], kwargs_lens_light=[]):

        source_data = imageio.imread(source_image_path, as_gray=True, pilmode=None)

        # we slightly convolve the image with a Gaussian convolution kernel of a few pixels (optional)
        source_gaussion = scipy.ndimage.filters.gaussian_filter(source_data, gaussion_sigma,
                                                                mode=gaussion_mode,
                                                                truncate=gaussion_truncate)

        # we now degrade the pixel resolution by a factor.
        # This reduces the data volume and increases the spead of the Shapelet decomposition
        #factor: lower resolution of image with a given factor
        numPix_large = int(len(source_gaussion) / factor)
        n_new = int((numPix_large - 1) * factor)
        source_cut = source_gaussion[0:n_new, 0:n_new]
        x, y = util.make_grid(numPix=numPix_large - 1, deltapix=1)  # make a coordinate grid
        source_resized = image_util.re_size(source_cut, factor)  # re-size image to lower resolution

        # we turn the image in a single 1d array
        image_1d = util.image2array(source_resized)  # map 2d image in 1d data array

        # we define the shapelet basis set we want the image to decompose in
        shapeletSet = ShapeletSet() #define the shapeletset function

        # basis_n_max choice of number of shapelet basis functions, 150 is a high resolution number, but takes long
        # scale_beta shapelet scale parameter (in units of resized pixels)
        # decompose image and return the shapelet coefficients
        coeff_source = shapeletSet.decomposition(image_1d, x, y, basis_n_max, scale_beta,
                                                 1., center_x=coeff_center_x,
                                                 center_y=coeff_center_y)

        # reconstruct NGC1300 with the shapelet coefficients
        image_reconstructed = shapeletSet.function(x, y, coeff_source, basis_n_max,
                                                   scale_beta, center_x=recon_center_x,
                                                   center_y=recon_center_y)
        # turn 1d array back into 2d image
        image_reconstructed_2d = util.array2image(image_reconstructed)  # map 1d data vector in 2d image

        # we define a very high resolution
        # grid for the ray-tracing (needs to be checked to be accurate enough!)

        #numPix number of pixels (low res of data)
        #deltaPix pixel size (low res of data)
        #high_res_factor higher resolution factor (per axis)
        # make the high resolution grid
        theta_x_high_res, theta_y_high_res = util.make_grid(numPix=numPix * high_res_factor,
                                                            deltapix=deltaPix / high_res_factor)
        # ray-shoot the image plane coordinates (angles) to the source plane (angles)
        beta_x_high_res, beta_y_high_res = self.lens_model.ray_shooting(theta_x_high_res,
                                                                        theta_y_high_res,
                                                                        kwargs=self.kwargs_lens_list)

        source_lensed = shapeletSet.function(beta_x_high_res, beta_y_high_res,
                                             coeff_source, basis_n_max, beta=lens_beta,
                                             center_x=lens_center_x, center_y=lens_center_y)
        source_lensed = util.array2image(source_lensed)

        if len(light_model_list) and len(kwargs_lens_light):

            #evaluate the surface brightness of the unlensed coordinates
            flux_lens = self.lens_model.surface_brightness(theta_x_high_res,theta_y_high_res,
                                                           self.kwargs_lens_list)
            flux_lens = util.array2image(flux_lens)
            add_lens_image = source_lensed + flux_lens

            return add_lens_image

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