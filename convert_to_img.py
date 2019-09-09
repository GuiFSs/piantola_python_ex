from PIL import Image
import numpy as np


def convert_array_to_img(array, img_name):
    img = Image.fromarray(array)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.save(img_name)
    print('done saving image')
