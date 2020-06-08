import numpy as np
import cupy
import Pk_library as PKL
import time


dimensions = 768
threads    = 1

random_array = np.random.random((dimensions,dimensions,dimensions)).astype(np.float32)
cupy_array   = cupy.array(random_array)

print(random_array.shape, random_array.dtype)
print(cupy_array.shape, cupy_array.dtype)

start = time.time()
cupy_fft = cupy.fft.rfftn(cupy_array, s=None, axes=None, norm=None)
print('Time take for cupy fft = %.3f seconds'%(time.time() - start))

start = time.time()
Paco_fft = PKL.FFT3Dr_f(random_array, threads)
print('Time take for Paco fft = %.3f seconds'%(time.time() - start))

cupy_fft_cpu = cupy.asnumpy(cupy_fft)

cupy_modulus = np.absolute(cupy_fft_cpu)
Paco_modulus = np.absolute(Paco_fft)

ratio = cupy_modulus/Paco_modulus
#print(ratio)
print(np.min(ratio), np.max(ratio))

