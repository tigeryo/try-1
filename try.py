# @author: tiger
# time: 2022/10/17 21:17


import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_Several_MinMax_Array(np_arr, several):
    """get max or min nums in array
    positive-min
    negative-max
    """
    if several > 0:
        several_min_or_max = np_arr[np.argpartition(np_arr,several)[:several]]
    else:
        several_min_or_max = np_arr[np.argpartition(np_arr, several)[several:]]
    return several_min_or_max


def contours_in(img_gray, contours):
    """get points in contours"""
    p = np.zeros(shape=img_gray.shape)
    cv2.drawContours(p, contours, -1, 255, -1)
    a = np.where(p==255)[0].reshape(-1,1)
    b = np.where(p==255)[1].reshape(-1,1)
    coordinate = np.concatenate([a,b], axis=1).tolist()
    inside = [list(x) for x in coordinate]
    return np.array(inside)


img_path = input()

img_ori = cv2.imread(img_path, 1)
# img_gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)

img_double = cv2.bilateralFilter(img_ori, 50, 100, 100)

img_gray = cv2.cvtColor(img_double, cv2.COLOR_BGR2GRAY)

t, img_t = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

k = np.ones((5, 5), np.uint8)
img_m = cv2.morphologyEx(img_t, cv2.MORPH_CLOSE, k, iterations=3)
img_m = 255 - img_m

contours, hierarchy = cv2.findContours(img_m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

contours_points = contours_in(img_gray, contours)

x_mean = np.mean(contours_points[:, 1])
d_max = np.max(contours_points[:, 1]) - np.min(contours_points[:, 1])

y_mins = get_Several_MinMax_Array(contours_points[:, 0], 10)
y_min = int(np.mean(y_mins))

# cv2.drawContours(img_ori, contours, -1, (0, 0, 0), 2)
img_mask = cv2.line(img_ori, (int(x_mean - 0.5*d_max), y_min), (int(x_mean + 0.5*d_max), y_min), (255, 125, 125), 2)


# cv2.namedWindow('tiger', 0)
# cv2.resizeWindow('tiger', 500, 500)
cv2.imshow('img_m', img_mask)
# cv2.imshow('img_c', img_m)
# cv2.imshow('img_t', img_t)
cv2.waitKey()
cv2.destroyAllWindows()

