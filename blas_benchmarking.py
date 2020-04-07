import numpy as np
from numpy import linalg as npl
import time

# Performs a variety of matrix operations to benchmark performance of
# numpy's core compiled math operations library (this is typically
# Intel MKL, which includes BLAS, LAPACK, ScaLAPACK, and other libraries)
#
# For square matrices whose dimensions range in size from 500-2000, this
# script performs:
# 1) Discrete Fast Fourier Transform (`np.fft`)
# 2) Inverse of Discrete Fast Fourier Transform (`np.ifft`)
# 3) Cholesky decomposition (`npl.cholesky`)
# 4) Orthogonal-triangular decomposition (`npl.qr`)
# 5) Singular Value decomposition (`npl.svd`)
# 6) Matrix inverse (`npl.inv`)
# 7) Matrix pseudoinverse (`npl.pinv`)
# 8) Matrix multiplication (`np.matmul`)

n_op = 10  # number of times to perform each operation
t0 = time.perf_counter()  # start stopwatch
m = [500, 1000, 1500, 2000]  # dimension length of square matrix
for N in m:
    t0_loop = time.perf_counter()  # start stopwatch for this iteration of loop
    print('\nN = %i, n_op = %i: ' % (N, n_op), end="", flush=True)
    np.random.seed(1)  # set random seed
    A = np.random.random_sample((N, N))  # create matrix of size (N,N)
    B = np.random.random_sample((N, N))  # create matrix of size (N,N)
    A_pd = np.matmul(A, A.T)  # make positive-definite matrix for `chol`

    t0_fft = time.perf_counter()
    for i in range(n_op):
        A_f = np.fft.fft(A)
    t_fft = time.perf_counter() - t0_fft
    print('fft ', end="", flush=True)

    t0_ifft = time.perf_counter()
    for i in range(n_op):
        np.fft.ifft(A_f)
    t_ifft = time.perf_counter() - t0_ifft
    print('ifft ', end="", flush=True)

    t0_cholesky = time.perf_counter()
    for i in range(n_op):
        npl.cholesky(A_pd)
    t_cholesky = time.perf_counter() - t0_cholesky
    print('cholesky ', end="", flush=True)

    t0_qr = time.perf_counter()
    for i in range(n_op):
        npl.qr(A)
    t_qr = time.perf_counter() - t0_qr
    print('qr ', end="", flush=True)

    t0_svd = time.perf_counter()
    for i in range(n_op):
        npl.svd(A)
    t_svd = time.perf_counter() - t0_svd
    print('svd ', end="", flush=True)

    t0_inv = time.perf_counter()
    for i in range(n_op):
        npl.inv(A)
    t_inv = time.perf_counter() - t0_inv
    print('inv ', end="", flush=True)

    t0_pinv = time.perf_counter()
    for i in range(n_op):
        npl.pinv(A)
    t_pinv = time.perf_counter() - t0_pinv
    print('pinv ', end="", flush=True)

    t0_matmul = time.perf_counter()
    for i in range(n_op):
        np.matmul(A, B)
    t_matmul = time.perf_counter() - t0_matmul
    print('matmul ')

    # Print all benchmark results for each iteration of loop
    print('\nTime in Seconds (SIZE: %i): ' % N)
    print('fft: %f' % t_fft)
    print('ifft: %f' % t_ifft)
    print('cholesky: %f' % t_cholesky)
    print('qr: %f' % t_qr)
    print('svd: %f' % t_svd)
    print('inv: %f' % t_inv)
    print('pinv: %f' % t_pinv)
    print('matmul: %f' % t_matmul)
    print('Total Time for N = %i: %f' % (N, time.perf_counter() - t0_loop))

print('\nTotal Time: %f\n' % (time.perf_counter() - t0))
