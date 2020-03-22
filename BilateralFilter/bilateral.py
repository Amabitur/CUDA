import pycuda.autoinit
from pycuda import driver, compiler, gpuarray
import numpy as np
import cv2
import timeit
from utils import bilateral

IMG_PATH = './data/seal-gray.bmp'

image = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)

M, N = image.shape

sigma_d = 200
sigma_r = 20
gpu_result = np.zeros((M, N), dtype=np.uint32)
block = (16, 16, 1)
grid = (int(np.ceil(M/block[0])),int(np.ceil(N/block[1])))

# собираем ядро
mod = compiler.SourceModule(open("kernel.cu", "r").read())
bilinear_interpolation_kernel = mod.get_function("interpolate")

start = driver.Event()
stop = driver.Event()

#подготовка текстуры
print("Считаем на ГПУ...")
start.record()

tex = mod.get_texref("tex")
tex.set_filter_mode(driver.filter_mode.LINEAR)
tex.set_address_mode(0, driver.address_mode.MIRROR)
tex.set_address_mode(1, driver.address_mode.MIRROR)
driver.matrix_to_texref(image.astype(np.uint32), tex, order="C")

bilinear_interpolation_kernel(driver.Out(gpu_result), np.int32(M), np.int32(N), np.float32(sigma_d), np.float32(sigma_r), block=block, grid=grid, texrefs=[tex])
stop.record()
stop.synchronize()
gpu_time = stop.time_since(start)
print("Время фильтрации на ГПУ: %.3f ms" % (gpu_time))

cv2.imwrite("./data/gpu-seal.bmp", gpu_result.astype(np.uint8))


print("Считаем на ЦПУ...")
start = timeit.default_timer()
cpu_result = bilateral(image, sigma_d, sigma_r)
cpu_time = timeit.default_timer() - start
print("Время фильтрации на ЦПУ: %.3f ms" % (cpu_time * 1e3))

cv2.imwrite("./data/cpu-seal.bmp", cpu_result)
