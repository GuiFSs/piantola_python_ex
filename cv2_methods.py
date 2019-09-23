import cv2
import numpy as np
import math

img_base_path = './imgs/'


def read_img(img_name):
    # return cv2.imread(img_base_path + img_name, cv2.IMREAD_GRAYSCALE)
    return cv2.imread(img_base_path + img_name, 0)


def show_img(img):
    cv2.imshow('equalized hist', img)
    cv2.waitKey(0)


def equalize():
    img = read_img('hulk.jpg')
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    equalized_img = cv2.equalizeHist(img)
    show_img(equalized_img)


def negative():
    img = read_img('hulk.jpg')
    show_img(255 - img)


def logarithmic():
    img = read_img('nix.jpg')
    C = 255 / math.log(256)
    res = C * np.uint8(np.log(img + 1))
    res = cv2.threshold(res, 1, 255, cv2.THRESH_BINARY)[1]
    print(res)
    show_img(res)


def pow_img():
    img = read_img('nix.jpg')
    C = 255 / math.log(256)
    B = 2
    res = np.uint8(np.power(img, B))
    # res = cv2.threshold(res, 1, 255, cv2.THRESH_BINARY)[1]
    show_img(res)


pow_img()
# logarithmic()
