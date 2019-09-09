import cv2
import numpy as np
from matplotlib import pyplot as plt
from convert_to_img import convert_array_to_img
import math

IMAGE_PATH = './imgs/'


def read_img_grey_scale(img_name):
    return cv2.imread('./imgs/'+img_name, cv2.IMREAD_GRAYSCALE)


def get_img_name(img_name, aditional_name):
    img_name_splited = img_name.split('.')
    aditional_img_name = f'{img_name_splited[0]}_{aditional_name}.{img_name_splited[1]}'
    return aditional_img_name


def equalize_img_and_save(img_name=None, equalized_img_name=None):
    img = cv2.imread('./imgs/'+img_name, cv2.IMREAD_GRAYSCALE)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    if equalized_img_name == None:
        equalized_img_name = get_img_name(img_name, 'equalized')

    N = img.shape[0] * img.shape[1]
    L = 256
    K = np.array(range(0, L))
    Hk = hist
    Hak = []

    prev = Hk[0][0]
    Hak.append(prev)

    for i, val in enumerate(Hk):
        if i != 0:
            Hak.append(prev + val[0])
            prev = prev + val[0]

    Hak = np.array(Hak)
    Pk = []
    K_line = []
    for num in Hak:
        curr_pk = num / N
        Pk.append(curr_pk)
        pk_rounded = round((L - 1) * curr_pk)
        K_line.append(pk_rounded)

    new_img = np.zeros(img.shape)
    for idx_line, line in enumerate(img):
        for idx_col, col in enumerate(line):
            new_img[idx_line][idx_col] = K_line[col]

    convert_array_to_img(new_img, './equalized_imgs/'+equalized_img_name)


def negative(img_name=None, negatived_img_name=None):
    L = 256
    img = cv2.imread('./imgs/'+img_name, cv2.IMREAD_GRAYSCALE)
    return ((L-1) - img, get_img_name(img_name, 'negative'))


def logarithmic(img_name=None, negatived_img_name=None):
    img = read_img_grey_scale(img_name)
    C = 255 / math.log(256)
    res = C * np.log(img + [1])  # + 1?
    res = res.astype(int)
    return (res, get_img_name(img_name, 'logari'))


def pow_img(img_name=None, negatived_img_name=None):
    img = read_img_grey_scale(img_name)
    C = 255 / math.log(256)
    B = 0.5
    res = np.power(C * img, B)
    res = res.astype(int)
    return (res, get_img_name(img_name, 'pow'))


def main():
    transformed_img, transformed_img_name = logarithmic('nix.jpg')
    print(transformed_img)
    convert_array_to_img(
        transformed_img, './logari_imgs/'+transformed_img_name)


if __name__ == "__main__":
    main()
