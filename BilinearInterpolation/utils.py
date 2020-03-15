import numpy as np

def prepare_image(image):
    uint32_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint32)
    w = uint32_image.shape[0]
    h = uint32_image.shape[1]
    for x in range(w):
        for y in range(h):
            for ch in range(image.shape[2]):
                uint32_image[x, y] += image[x, y, ch] << (8 * (image.shape[2] - ch - 1))
    return uint32_image

def normalize_image(image, channels):
    rgba_image = np.zeros((image.shape[0], image.shape[1], channels), dtype=np.uint32)
    for x in range(rgba_image.shape[0]):
        for y in range(rgba_image.shape[1]):
            output_x_y = image[x, y]
            for ch in range(rgba_image.shape[2]):
                rgba_image[x, y, rgba_image.shape[2] - ch - 1] = output_x_y % 256
                output_x_y >>= 8
    return rgba_image

def bilinear(image):
    preres = np.zeros((image.shape[0]*2, image.shape[1]*2, image.shape[2]), dtype=np.uint32)
    w = preres.shape[0]
    h = preres.shape[1]
    for i in range(w):
        for j in range(h):
            preres[i, j] = bi_pixel(image, i, j)
    return preres

def bi_pixel(image, w, h):
    scale = 2
    out = []
    i = int(w/scale)
    j = int(h/scale)
    k = 0 if i + 1 >= image.shape[0] else i + 1
    l = 0 if j + 1 >= image.shape[1] else j + 1
    for ch in range(image.shape[2]):
        a = image[i, j][ch]/255
        b = image[k,j][ch]/255
        c = image[i,l][ch]/255
        d = image[k,l][ch]/255
        out.append((a + b + c + d)*255*0.25)
    return out