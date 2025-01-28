// Includes
#include <cmath>
#include <cstdio>
#include <stdio.h>
#include <chrono>

// Jacobi iteration kernel using global memory
__global__ void jacobi(const float *A, const float *b, float *x, float *xNew, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 

    float sum = 0.0f;
    for (int j = 0; j < N; ++j) {
        if (j != i) {
            sum += A[i * N + j] * x[j];
        }
    }
    xNew[i] = (b[i] - sum) / A[i * N + i];
}

// Host code to test Jacobi iteration
int main() {
    const int N = 128; // Size of the matrix and vectors
    const int maxIterations = 10000;
    const float tolerance = 1e-5f;
    std::chrono::steady_clock::time_point start; // start timer
    std::chrono::steady_clock::time_point stop; // stop timer

    // Host arrays
    float *A, *b, *x, *xNew;
    A = (float *)malloc(N * N * sizeof(float));
    b = (float *)malloc(N * sizeof(float));
    x = (float *)malloc(N * sizeof(float));
    xNew = (float *)malloc(N * sizeof(float));

    // Initialize A, b, and x
    for (int i = 0; i < N; i++) {
        x[i] = 0.0f; // Initial guess
        b[i] = static_cast<float>(i);
        for (int j = 0; j < N; j++) {
            A[i * N + j] = rand() % 100 + 1;
        }
    }
    for (int i = 0; i < N; i++) {
        int sum = 0;
        for (int j = 0; j < N; j++) {
            sum += A[i * N + j];
        }
        A[i * N + i] = sum + (rand() % 100 + 1);
    }

    // Device arrays
    float *dA, *db, *dx, *dxNew;
    cudaMalloc((void **)&dA, N * N * sizeof(float));
    cudaMalloc((void **)&db, N * sizeof(float));
    cudaMalloc((void **)&dx, N * sizeof(float));
    cudaMalloc((void **)&dxNew, N * sizeof(float));

    // Copy data to device
    cudaMemcpy(dA, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch Jacobi iteration kernel
    start = std::chrono::steady_clock::now();

    for (int iter = 0; iter < maxIterations; iter++) {
        for (int i = 0; i < N; i++) {
            xNew[i] = x[i];
        }

        cudaMemcpy(dx, x, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dxNew, xNew, N * sizeof(float), cudaMemcpyHostToDevice);

        jacobi<<<1, N>>>(dA, db, dx, dxNew, N);

        cudaMemcpy(xNew, dxNew, N * sizeof(float), cudaMemcpyDeviceToHost);

        float maxError = 0.0f;
        for (int i = 0; i < N; i++) {
            maxError = fmaxf(maxError, fabs(xNew[i] - x[i]));
            x[i] = xNew[i];
        }

        if (maxError < tolerance) {
            printf("Converged after %d iterations with max error: %f\n", iter + 1, maxError);
            break;
        }
    }
    stop = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    printf("Solution:\n");
    for (int i = 0; i < fminf(N, 10); i++) { // Print first 10 elements
        printf("x[%d] = %f\n", i, x[i]);
    }
    printf("Elapsed time: %ld ms\n", elapsed);

    // Cleanup
    free(A);
    free(b);
    free(x);
    free(xNew);
    cudaFree(dA);
    cudaFree(db);
    cudaFree(dx);
    cudaFree(dxNew);

    return 0;
}
