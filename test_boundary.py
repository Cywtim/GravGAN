
import numpy as np
import matplotlib.pyplot as plt
from mock_lens_generator import nearest_random_pad, resize_fig

image_path = r"img/image_000123.jpg"
p = nearest_random_pad(image_path, 256)
plt.imshow(p)
plt.show()

p1 = resize_fig(p, (256, 256))
plt.imshow(p1)
plt.show()