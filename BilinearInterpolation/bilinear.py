import pycuda.autoinit
from pycuda import driver, compiler, gpuarray
import numpy as np
import cv2
import timeit
from utils import prepare_image, normalize_image, bilinear

IMG_PATH = './data/seal.png'
image = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
M1, N1 = image.shape
M2 = int(2*M1)
N2 = int(2*N1)

result = np.zeros((M2, N2), dtype=np.uint32)
block = (16, 16, 1)
grid = (int(np.ceil(M2/block[0])),int(np.ceil(N2/block[1])))

# собираем ядро
mod = compiler.SourceModule(open("kernel.cu", "r").read())
bilinear_interpolation_kernel = mod.get_function("interpolate")

start = driver.Event()
stop = driver.Event()

#подготовка текстуры
print("Считаем на ГПУ...")
start.record()

#prep_image = prepare_image(image)
tex = mod.get_texref("tex")
tex.set_filter_mode(driver.filter_mode.LINEAR)
tex.set_address_mode(0, driver.address_mode.CLAMP)
tex.set_address_mode(1, driver.address_mode.CLAMP)
driver.matrix_to_texref(image, tex, order="C")

bilinear_interpolation_kernel(driver.Out(result), np.int32(M1), np.int32(N1), np.int32(M2), np.int32(N2), block=block, grid=grid, texrefs=[tex])
#big_image = normalize_image(result, image.shape[2])
stop.record()
stop.synchronize()
gpu_time = stop.time_since(start)
print("Время интерполяции на ГПУ: %.3f ms" % (gpu_time))

cv2.imwrite("data/big-gpu-seal.png", result.astype(np.uint8))

#p_image = prepare_image(image)

print("Считаем на ЦПУ...")
start = timeit.default_timer()
cpu_result = bilinear(image)
cpu_time = timeit.default_timer() - start
print("Время интерполяции на ЦПУ: %.3f ms" % (cpu_time * 1e3))

#big_cpu_image = normalize_image(cpu_result, image.shape[2])

cv2.imwrite("./data/big-cpu-seal.png", cpu_result.astype(np.uint8))

