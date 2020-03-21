import numpy as np

def bilateral(image, sigma_d, sigma_r):
    n_image = np.zeros(image.shape)
    w = image.shape[0]
    h = image.shape[1]
    for i in range(1, w-1):
        for j in range(1, h-1):
            n_image[i, j] = bil_pixel(image, i, j, sigma_d, sigma_r)
    return n_image


def bil_pixel(image, i, j, sigma_d, sigma_r):
    #mask_image = image[i-1:i+2, j-1:j+2]
    c = 0
    s = 0
    for k in range(i-1, i+2):
        for l in range(j-1, j+2):
            g = np.exp(-((k - i) ** 2 + (l - j) ** 2) / sigma_d ** 2)
            i1 = image[k, l]/255
            i2 = image[i, j]/255
            r = np.exp(-((i1 - i2)*255) ** 2 / sigma_r ** 2)
            c += g*r
            s += g*r*image[k, l]
    result = s / c
    return result
