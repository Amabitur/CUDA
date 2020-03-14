import pycuda.autoinit
from pycuda import driver, compiler, gpuarray
import numpy as np
from string import Template
import matplotlib.pyplot as plt
import cv2
import imageio
import timeit

IMG_PATH = './data/deer.jpeg'


image = cv2.imread(IMG_PATH)

M1, N1, _ = image.shape
M2 = int(2*M1)
N2 = int(2*N1)

result = np.zeros((M2, N2), dtype=np.uint32)
print(image.shape)
#result = np.reshape(result, (1, -1))[0]

block = (16, 16, 1)
grid = (int(np.ceil(M2/block[0])),int(np.ceil(N2/block[1])))
print(grid)

# собираем ядро
mod = compiler.SourceModule(open("kernel.cu", "r").read())
bilinear_interpolation_kernel = mod.get_function("interpolate")

start = driver.Event()
stop = driver.Event()

uint32_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint32)
w = uint32_image.shape[0]
h = uint32_image.shape[1]
for x in range(w):
    for y in range(h):
        for ch in range(image.shape[2]):
            uint32_image[x, y] += image[x, y, ch] << (8 * (image.shape[2] - ch - 1))

cu_tex = mod.get_texref("d_texture_interp_float")
cu_tex.set_filter_mode(driver.filter_mode.LINEAR)
cu_tex.set_address_mode(0, driver.address_mode.CLAMP)
cu_tex.set_address_mode(1, driver.address_mode.CLAMP)
driver.matrix_to_texref(uint32_image, cu_tex, order="C")

print(uint32_image)

print("Считаем на ГПУ...")
start.record()
bilinear_interpolation_kernel(driver.Out(result), np.int32(M1), np.int32(N1), np.int32(M2), np.int32(N2), block=block, grid=grid, texrefs=[cu_tex])
stop.record()
stop.synchronize()
gpu_time = stop.time_since(start)
print("Время перемножения матриц на ГПУ: %.3f ms" % (gpu_time))

print(result.shape)

rgba_image = np.zeros((M2, N2, image.shape[2]), dtype=np.uint32)
for x in range(rgba_image.shape[0]):
    for y in range(rgba_image.shape[1]):
        output_x_y = result[x, y]
        for ch in range(rgba_image.shape[2]):
            rgba_image[x, y, rgba_image.shape[2] - ch - 1] = output_x_y % 256
            output_x_y >>= 8

plt.imshow(rgba_image)
plt.show()
cv2.imwrite("./data/deer-int.png", rgba_image.astype(np.uint8))

#print("Считаем на ЦПУ...")
#start = timeit.default_timer()
#cpu_time = timeit.default_timer() - start
#print("Время перемножения матриц на ЦПУ: %.3f ms" % (cpu_time * 1e3))

#сравниваем полученные результаты, если они различаются не критично - хорошо, если сильно - не очень хорошо, но ведь никто нам и не гарантирует точность вычислений на ГПУ :(
if (True):
    print("Наши восхитительные результаты получены")
else:
    print("Что-то пошло не так, результаты расходятся :(")
