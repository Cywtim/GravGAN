"""

This file is to translate the img into gray figures

"""

import os
import sys
sys.path.append("..")
import imageio
import numpy as np

fold_path = r"\img"
main_d = os.getcwd()
fold = os.listdir(main_d + fold_path)
num_fig = len(fold) - 1

for num in range(num_fig):

    str_num = str(num)
    str_num = str_num.zfill(6)
    file_path = r"\img\image_" + str_num + r".jpg"

    source_image = imageio.imread(main_d + file_path, as_gray=True, pilmode=None)

    imageio.imsave(main_d + r"\gray\image_" + str_num +r".jpg",source_image)

