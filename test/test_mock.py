"""

This file is an example to lens a source,
you can directly run it and get a pair of image of
original and lensed figures.

CHENGYI

"""

import sys
sys.path.append("..")
import imageio
import numpy as np
import matplotlib.pyplot as plt
import mock_lens_generator as mock_lens

num = np.random.randint(400)
str_num = str(num)
str_num = str_num.zfill(6)
file_path = r"..\img\image_" + str_num +r".jpg"
psf_path = "..\img\psf_example.fits"
test_image = imageio.imread(file_path,as_gray=True, pilmode=None)
main_halo_type = 'SIE'  # You have many other possibilities available. Check out the SinglePlane class!
kwargs_lens_main = {'theta_E': 1, 'e1': 0.1, 'e2': 0.0, 'center_x': 0.1, 'center_y': 0.1}
kwargs_shear = {'gamma1': 0.05, 'gamma2': 0}
lens_model_list = main_halo_type
lens = mock_lens.Mock_Lens_Generator(lens_model_list, kwargs_lens_main, kwargs_shear, shear="")
source_lensed = lens.shapelet_lens_image(file_path, factor=1,
                                         lens_center_x= 0.3, lens_center_y=-0.1)

#PSF file aslo in img floder
source_psf = lens.PSF_conv(source_lensed,psf_path)
#down sample
#source_downsample = lens.down_sample(source_psf)
#noisy
source_noisy = lens.add_poisson_noisy(source_psf)

imageio.imsave(r"..\lensed\image_" + str_num +r".jpg",source_noisy)

plt.imshow(test_image);plt.show()
plt.imshow(source_noisy);plt.show()