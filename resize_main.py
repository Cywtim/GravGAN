
import numpy as np
import cv2
from mock_lens_generator import resize_fig
import matplotlib.pyplot as plt

file_path = "img/image_000014.jpg"

image = resize_fig(file_path, 300/256, 300/256)
plt.imshow(image)
plt.show()