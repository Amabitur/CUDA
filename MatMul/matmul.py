import pycuda.autoinit
from pycuda import driver, compiler, gpuarray
import numpy as np
from string import Template

import timeit

N = 2048

kernel_code_template = Template("""
__global__ void matmul_kernel(float *d_C, float *d_A, float *d_B)
{
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    float sum = 0.f;
    for (int e = 0; e < ${MATRIX_SIZE}; e++)
        sum += d_A[idx_y * ${MATRIX_SIZE} + e] * d_B[e * ${MATRIX_SIZE} + idx_x];
    d_C[idx_y * ${MATRIX_SIZE} + idx_x] = sum;
}
""")

def matmul(A, B):
    C = np.zeros(N*N)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i * N + j] += A[i * N + k] * B[k * N + j]
    return C


np.random.seed(42)
A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)
C = np.zeros((N, N), dtype=np.float32)
A = np.reshape(A, (1, -1))[0]
B = np.reshape(B, (1, -1))[0]
C = np.reshape(C, (1, -1))[0]

# собираем ядро
mod = compiler.SourceModule( \
        kernel_code_template.substitute(MATRIX_SIZE=N))

matmul_kernel = mod.get_function("matmul_kernel")

dimBlock = 16
dimGrid = int((N + dimBlock - 1) / dimBlock)

start = driver.Event()
stop = driver.Event()

print("Считаем на ГПУ...")
start.record()
matmul_kernel(driver.Out(C), driver.In(A), driver.In(B), block=(dimBlock, dimBlock, 1), grid=(dimGrid, dimGrid))
stop.record()
stop.synchronize()
gpu_time = stop.time_since(start)
print("Время перемножения матриц на ГПУ: %.3f ms" % (gpu_time))

print("Считаем на ЦПУ...")
start = timeit.default_timer()
c_cpu = matmul(A, B)
cpu_time = timeit.default_timer() - start
print("Время перемножения матриц на ЦПУ: %.3f ms" % (cpu_time * 1e3))

#сравниваем полученные результаты, если они различаются не критично - хорошо, если сильно - не очень хорошо, но ведь никто нам и не гарантирует точность вычислений на ГПУ :(
if (np.allclose(c_cpu, C)):
    print("Наши восхитительные результаты получены")
else:
    print("Что-то пошло не так, результаты расходятся :(")
