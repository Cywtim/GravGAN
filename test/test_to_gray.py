"""

This test file is to translate the img into gray figures

"""

import os
import sys
sys.path.append("..")
import imageio
import numpy as np

num = np.random.randint(400)
str_num = str(num)
str_num = str_num.zfill(6)
file_path = r"..\img\image_" + str_num +r".jpg"
source_image = imageio.imread(file_path, as_gray=True, pilmode=None)

imageio.imsave(r"..\gray\image_" + str_num +r".jpg",source_image)

