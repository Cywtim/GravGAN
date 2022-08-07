
import os
import imageio
import matplotlib.pyplot as plt
from mock_lens_generator import nearest_random_pad, resize_fig


fold_path = r"\img"
main_d = os.getcwd()
fold = os.listdir(main_d + fold_path)
num_fig = len(fold) - 1

for num in range(num_fig):

    str_num = str(num)
    str_num = str_num.zfill(6)
    image_path = r"img\image_" + str_num +r".jpg"
    padding = nearest_random_pad(image_path, 256)
    padding = resize_fig(padding, (64,64))

    imageio.imsave(main_d + r"\img_shrink_64\image_" + str_num + r".png", padding)
