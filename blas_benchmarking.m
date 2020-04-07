% Performs a variety of matrix operations to benchmark performance of
% MATLAB's core compiled math operations library (this is typically
% Intel MKL, which includes BLAS, LAPACK, ScaLAPACK, and other libraries)
%
% For square matrices whose dimensions range in size from 500-2000, this
% script performs:
% 1) Discrete Fast Fourier Transform (`fft`)
% 2) Inverse of Discrete Fast Fourier Transform (`ifft`)
% 3) Cholesky factorization (`chol`)
% 4) Orthogonal-triangular decomposition (`qr`)
% 5) Singular Value decomposition (`svd`)
% 6) Matrix inverse (`inv`)
% 7) Matrix pseudoinverse (`pinv`)
% 8) Matrix multiplication (`mtimes`)

n_op = 10;  % number of times to perform each operation
t0 = tic;  % start stopwatch
for N = [500, 1000, 1500, 2000]  % dimension length of square matrix
    t0_loop = tic; % start stopwatch for this iteration of loop
    fprintf('\n\nN = %i, n_op = %i: ', N, n_op);
    rng(1);  % set random number generator
    A = rand([N, N]);  % create matrix of size (N,N)
    B = rand([N, N]);  % create matrix of size (N,N)
    A_pd = A * A';  % make positive-definite matrix for `chol`
 
    tic;
    for i = 1:n_op
        A_f = fft(A);
    end
    t_fft = toc;
    fprintf('fft ');
    
    tic;
    for i = 1:n_op
        ifft(A_f);
    end
    t_ifft = toc;
    fprintf('ifft ');

    tic;
    for i = 1:n_op
        chol(A_pd);
    end
    t_chol = toc;
    fprintf('chol ');
    
    tic;
    for i = 1:n_op
        qr(A);
    end
    t_qr = toc;
    fprintf('qr ');

    tic;
    for i = 1:n_op
        svd(A);
    end
    t_svd = toc;
    fprintf('svd ');
    
    tic;
    for i = 1:n_op
        inv(A);
    end
    t_inv = toc;
    fprintf('inv ');
 
    tic;
    for i = 1:n_op
        pinv(A);
    end
    t_pinv = toc;
    fprintf('pinv ');
 
    tic;
    for i = 1:n_op
        mtimes(A, B); %#ok<*VUNUS>
    end
    t_mtimes = toc;
    fprintf('mtimes');

    % Print all benchmark results for each iteration of loop
    fprintf('\n\nTime in Seconds (SIZE: %i):', N);
    fprintf('\nfft: %f', t_fft);
    fprintf('\nifft: %f', t_ifft);
    fprintf('\nchol: %f', t_chol);
    fprintf('\nqr: %f', t_qr);
    fprintf('\nsvd: %f', t_svd);
    fprintf('\ninv: %f', t_inv);
    fprintf('\npinv: %f', t_pinv);
    fprintf('\nmtimes: %f', t_mtimes);
    fprintf('\nTotal Time for N = %i: %f', N, toc(t0_loop));
end

fprintf('\n\nTotal Time: %f\n', toc(t0));
