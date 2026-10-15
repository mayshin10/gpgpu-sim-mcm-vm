// GEMM_REUSE_SMALL: GEMM_REUSE (CLAP MICRO'25 Fig. 20) scaled to ~1/32 of the work
// so a trace-driven simulation finishes in about an hour.
//
//   kernel 1: C* (M x N)   = A (M x K) * B (K x N)          M=1024, K=384, N=512
//   kernel 2: D  (M/2 x N) = C*q (M/2 x N/2) * E (N/2 x N)
//
// C*q is the top-left quarter of C*, read in place (leading dimension N), so the
// second kernel reuses a quarter of a data structure mapped by the first kernel
// with a different access pattern. Kernel body follows the CUDA samples
// matrixMul (block 32) with explicit leading dimensions.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32

static const int M = 1024;
static const int K = 384;
static const int N = 512;

#define CHECK(call)                                                          \
  do {                                                                       \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err),   \
              __FILE__, __LINE__);                                           \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)

// C (rows x cols, ldc) = A (rows x k, lda) * B (k x cols, ldb)
__global__ void MatrixMulLd(float *C, const float *A, const float *B, int k,
                            int lda, int ldb, int ldc) {
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int aBegin = lda * BLOCK_SIZE * by;
  int aEnd = aBegin + k - 1;
  int aStep = BLOCK_SIZE;
  int bBegin = BLOCK_SIZE * bx;
  int bStep = BLOCK_SIZE * ldb;

  float Csub = 0;
  for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    As[ty][tx] = A[a + lda * ty + tx];
    Bs[ty][tx] = B[b + ldb * ty + tx];
    __syncthreads();

#pragma unroll
    for (int i = 0; i < BLOCK_SIZE; ++i) {
      Csub += As[ty][i] * Bs[i][tx];
    }
    __syncthreads();
  }

  int c = ldc * BLOCK_SIZE * by + BLOCK_SIZE * bx;
  C[c + ldc * ty + tx] = Csub;
}

static void ConstantInit(float *data, size_t size, float val) {
  for (size_t i = 0; i < size; ++i) data[i] = val;
}

int main() {
  const int M2 = M / 2;  // rows of C*q and D
  const int K2 = N / 2;  // cols of C*q

  const size_t size_A = (size_t)M * K;
  const size_t size_B = (size_t)K * N;
  const size_t size_C = (size_t)M * N;
  const size_t size_E = (size_t)K2 * N;
  const size_t size_D = (size_t)M2 * N;

  const float valA = 1.0f, valB = 0.01f, valE = 0.01f;
  float *h_A = (float *)malloc(size_A * sizeof(float));
  float *h_B = (float *)malloc(size_B * sizeof(float));
  float *h_E = (float *)malloc(size_E * sizeof(float));
  float *h_D = (float *)malloc(size_D * sizeof(float));
  if (!h_A || !h_B || !h_E || !h_D) {
    fprintf(stderr, "host allocation failed\n");
    return EXIT_FAILURE;
  }
  ConstantInit(h_A, size_A, valA);
  ConstantInit(h_B, size_B, valB);
  ConstantInit(h_E, size_E, valE);

  float *d_A, *d_B, *d_C, *d_E, *d_D;
  CHECK(cudaMalloc(&d_A, size_A * sizeof(float)));
  CHECK(cudaMalloc(&d_B, size_B * sizeof(float)));
  CHECK(cudaMalloc(&d_C, size_C * sizeof(float)));
  CHECK(cudaMalloc(&d_E, size_E * sizeof(float)));
  CHECK(cudaMalloc(&d_D, size_D * sizeof(float)));
  CHECK(cudaMemcpy(d_A, h_A, size_A * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_B, h_B, size_B * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_E, h_E, size_E * sizeof(float), cudaMemcpyHostToDevice));

  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

  printf("kernel 1: C*(%d,%d) = A(%d,%d) * B(%d,%d)\n", M, N, M, K, K, N);
  dim3 grid1(N / BLOCK_SIZE, M / BLOCK_SIZE);
  MatrixMulLd<<<grid1, threads>>>(d_C, d_A, d_B, K, K, N, N);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  printf("kernel 2: D(%d,%d) = C*q(%d,%d, ld %d) * E(%d,%d)\n", M2, N, M2, K2, N, K2, N);
  dim3 grid2(N / BLOCK_SIZE, M2 / BLOCK_SIZE);
  MatrixMulLd<<<grid2, threads>>>(d_D, d_C, d_E, K2, N, N, N);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  CHECK(cudaMemcpy(h_D, d_D, size_D * sizeof(float), cudaMemcpyDeviceToHost));

  // every element of C* is K*valA*valB, so every element of D is K2*(K*valA*valB)*valE
  const double ref = (double)K2 * ((double)K * valA * valB) * valE;
  const double eps = 1.e-4;
  bool correct = true;
  for (size_t i = 0; i < size_D; i++) {
    double rel_err = fabs(h_D[i] - ref) / fabs(ref);
    if (rel_err > eps) {
      printf("Error! D[%zu]=%.8f, ref=%.8f\n", i, h_D[i], ref);
      correct = false;
      break;
    }
  }
  printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

  free(h_A); free(h_B); free(h_E); free(h_D);
  CHECK(cudaFree(d_A)); CHECK(cudaFree(d_B)); CHECK(cudaFree(d_C));
  CHECK(cudaFree(d_E)); CHECK(cudaFree(d_D));
  return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
