# blas_benchmarking

MATLAB and Python matrix operation benchmarking.

Benchmarking for blas + other compiled math libraries (typically within Intel MKL or openblas) used by MATLAB and Python packages.

Contents:
`blas_benchmarking.m`: a script to run benchmarking in MATLAB
`blas_benchmarking.py`: a script to run benchmarking in Python.
`Intel_MKL_Benchmarks.txt`: output of MATLAB (2019B) AND Python numpy (1.17.2) benchmarking before and after forcing Intel MKL to use the AVX2 code path for an AMD CPU.
