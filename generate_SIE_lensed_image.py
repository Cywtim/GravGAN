"""

This file use mock lens generator to produce a figures in img fold
and save the pictures in lens fold by Singularity Isotropic Elliptical model
(SIE model) with shears.

CHENGYI

"""


import os
import sys
sys.path.append("..")
import pandas as pd
import imageio
import numpy as np
import mock_lens_generator as mock_lens

fold_path = r"\img"
main_d = os.getcwd()
fold = os.listdir(main_d + fold_path)
num_fig = len(fold) - 1

theta_E = (np.random.random(num_fig) + 1 ) * 0.5
e1 = np.random.uniform(-1,1,num_fig) * 0.8
e2 = np.random.uniform(-1,1,num_fig) * 0.8
center_x = np.random.uniform(-1,1,num_fig) * 0.3
center_y = np.random.uniform(-1,1,num_fig) * 0.3
gamma1 = np.random.uniform(0,1,num_fig) * 0.1
gamma2 = np.random.uniform(0,1,num_fig) * 0.1
lens_center_x =  np.random.uniform(-1,1,num_fig) * 0.5
lens_center_y = np.random.uniform(-1,1,num_fig) * 0.5


psf_path = "\img\psf_example.fits"
for num in range(num_fig):

    str_num = str(num)
    str_num = str_num.zfill(6)
    file_path = r"\img\image_" + str_num +r".jpg"

    test_image = imageio.imread(main_d + file_path,as_gray=True, pilmode=None)
    main_halo_type = 'SIE'  # You have many other possibilities available. Check out the SinglePlane class!
    kwargs_lens_main = {'theta_E': theta_E[num], 'e1': e1[num],'e2': e2[num],
                        'center_x': center_x[num], 'center_y': center_y[num]}
    kwargs_shear = {'gamma1': gamma1[num], 'gamma2': gamma2[num]}
    lens_model_list = main_halo_type
    lens = mock_lens.Mock_Lens_Generator(lens_model_list, kwargs_lens_main, kwargs_shear)
    source_lensed = lens.shapelet_lens_image(main_d+file_path,factor=1,
                                             lens_center_x=lens_center_x[num],
                                             lens_center_y=lens_center_y[num])

    #PSF file aslo in img floder
    source_psf = lens.PSF_conv(source_lensed,main_d + psf_path)
    #noisy
    source_noisy = lens.add_poisson_noisy(source_psf)

    imageio.imsave(main_d + r"\SIElensed\image_" + str_num +r".jpg",source_noisy)

    para_file = main_d + r"\SIEpara\image_" + str_num + r".csv"
    with open(para_file, "w") as pfile:
        paras = [theta_E[num], e1[num], e2[num], center_x[num], center_y[num], gamma1[num], gamma2[num]]
        pfile.write(",".join(str(i) for i in paras))
