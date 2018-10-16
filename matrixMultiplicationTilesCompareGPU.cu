// system libraries
// use nvcc -o (output name) -Wno-deprecated-gpu-targets -std=c++11 -Xcompiler -fopenmp  file_name.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <chrono>

// size definition. modify as needed
#define N 2000
#define T_SIZE 32

using namespace std;

// safe call definition
static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number){
	if(err!=cudaSuccess){
		fprintf(stderr,"%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n",msg,file_name,line_number,cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

// safe call definition
#define SAFE_CALL(call,msg) _safe_cuda_call(call,msg,__FILE__,__LINE__)

// initialize major row matrix
void initializeMatrix(float *ip, const int nxy){
  srand (static_cast <unsigned> (time(0)));
  float random;
  for(int i = 0; i < nxy; i++){
    random = 1.0 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(10.0-1.0)));
    ip[i] = random;
  }
    return;
}


// utility function to check result
void checkResult(float *hostRef, float *gpuRef, const int nxy){
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < nxy; i++){
        if (abs(hostRef[i] - gpuRef[i]) > epsilon){
            match = 0;
            printf("host %f gpu %f\n", hostRef[i], gpuRef[i]);
            break;
        }
    }

    if (match)
        printf("Arrays match.\n\n");
    else
        printf("Arrays do not match.\n\n");
}

// multiply matrix on host
void multiplyMatrixOnHost(float *A, float *B, float *C, const int nx){

		for(int i = 0; i < nx; i++) {
			 for(int j = 0; j < nx; j++) {
					 for(int k = 0; k < nx; k++) {
							 C[i * nx + j] += A[i * nx + k] * B[j + k * nx];
					 }
			 }
	 }

    return;
}

// function to multiply matrix on host with threads
void multiplyMatrixOnHostThreads(float *A, float *B, float *C, const int nx){

    int i = 0;
    // use the pragma directive to automatically paralelize
    #pragma omp parallel for private(i) shared(A, B, C)
		for(i = 0; i < nx; i++) {
			 for(int j = 0; j < nx; j++) {
					 for(int k = 0; k < nx; k++) {
							 C[i * nx + j] += A[i * nx + k] * B[j + k * nx];
					 }
			 }
	 }

    return;
}

// kernel to multiply matrix on gpu
__global__ void multiplyMatrixOnGPU(float *A, float *B, float *C, const int nx){

		// get ix and iy from cuda defined variables
		unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0.0;
    if (ix < nx && iy < nx){
        for(int i = 0; i < nx ; i++)
            sum += A[iy * nx + i] * B[i * nx + ix];
        C[iy * nx + ix] = sum;
    }

}

// Kernel GPU Tiles
__global__ void multiplyMatrixOnGPUTiles(float *A, float *B, float *C, const int nx){
	// Create the shared memory space as tiles
	__shared__ float tileOne[T_SIZE][T_SIZE], tileTwo[T_SIZE][T_SIZE];

	// Get the ix and iy indexes
	unsigned int ix = T_SIZE * blockIdx.x + threadIdx.x;
	unsigned int iy = T_SIZE * blockIdx.y + threadIdx.y;

  // int limit = (T_SIZE + nx - 1)/T_SIZE;
	// Get other limit to experiment
	int limit = ceilf(((float)T_SIZE + (float)nx)/(float)T_SIZE);
	// Partial Sum acumulator
	float partialSum = 0.0;

  int i = 0;
  while(i < limit){
			// Fetch values for each value of the tiles with restriction
			if ((iy < nx) && ((i * T_SIZE + threadIdx.x) < nx)){
				int id = (iy * nx) + (i * T_SIZE) + threadIdx.x;
				tileOne[threadIdx.y][threadIdx.x] = A[id];
			}else{
        tileOne[threadIdx.y][threadIdx.x] = 0.0;
				// DO NOT PRINT RACE CONDITION GIVES WRONG OUTPUT
				// cuPrintf(""); <--- deprecated
				// printf("Improper Tile Size in X domain, zeroing\n");
			}

      // Wait for threads to finish
			__syncthreads();

			// Fetch values for each value of the tiles with restriction
			if ((ix < nx) && ((i * T_SIZE + threadIdx.y) < nx)){
				int id = (i * T_SIZE + threadIdx.y) * nx + ix;
				tileTwo[threadIdx.y][threadIdx.x] = B[id];
			}else{
        tileTwo[threadIdx.y][threadIdx.x] = 0.0;
				// DO NOT PRINT RACE CONDITION GIVES WRONG OUTPUT
				// printf("Improper Tile Size in Y domain, zeroing\n");
			}

			// Wait for threads to finish
			__syncthreads();

			//Perform partial sum on tile
      #pragma unroll // T_SIZE is constant
			for (int j = 0; j < T_SIZE; j++){
					partialSum += tileOne[threadIdx.y][j] * tileTwo[j][threadIdx.x];
			}

			// DO NOT PRINT RACE CONDITION GIVES WRONG OUTPUT
			//printf("Partial Sum fetched with value %f\n", partialSum);
			// Wait for threads to finish
			__syncthreads();
      i++;
		}
    if (ix < nx && iy < nx)
        C[((blockIdx.y * blockDim.y + threadIdx.y) * nx) + (blockIdx.x * blockDim.x) + threadIdx.x] = partialSum;
}

int main(int argc, char* argv[]) {
  printf("%s Starting...\n", argv[0]);

  // set up device
  int dev = 0;
  cudaDeviceProp deviceProp;
  SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev), "Error device prop");
  printf("Using Device %d: %s\n", dev, deviceProp.name);
  SAFE_CALL(cudaSetDevice(dev), "Error setting device");

  int nx = N;
  int ny = N;
  int nxy = nx * ny;
  int nBytes = nxy * sizeof(float*);
  printf("Matrix size: nx %d ny %d\n", nx, ny);

  // malloc host memory
  float *h_A = (float *)malloc(nBytes);
  float *h_B = (float *)malloc(nBytes);
  float *hostRef = (float *)malloc(nBytes);
  float *hostRefThreads = (float *)malloc(nBytes);
  float *gpuRef = (float *)malloc(nBytes);
	float *gpuRefTiles = (float *)malloc(nBytes);

  // initialize matrix
  initializeMatrix(h_A, nxy);
  initializeMatrix(h_B, nxy);

  // initialize to 0
  memset(hostRef, 0, nBytes);
  memset(hostRefThreads, 0, nBytes);
  memset(gpuRef, 0, nBytes);
	memset(gpuRefTiles, 0, nBytes);

  // // multiply matrix on host
  // auto start_cpu = std::chrono::high_resolution_clock::now();
  // multiplyMatrixOnHost(h_A, h_B, hostRef, nx);
  // auto end_cpu =  std::chrono::high_resolution_clock::now();
  // std::chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
  // printf("multiplyMatrixOnHost elapsed %f ms\n", duration_ms.count());

  // // multiply matrix on host with threads
  // start_cpu =  std::chrono::high_resolution_clock::now();
  // multiplyMatrixOnHostThreads(h_A, h_B, hostRefThreads, nx);
  // end_cpu =  std::chrono::high_resolution_clock::now();
  // duration_ms = end_cpu - start_cpu;
  // printf("multiplyMatrixOnHostThreads elapsed %f ms\n", duration_ms.count());

  // // check results
  // checkResult(hostRef, hostRefThreads, nx);

  // malloc device global memory
  float *d_MatA, *d_MatB, *d_MatC, *d_MatD;
  SAFE_CALL(cudaMalloc((void **)&d_MatA, nBytes), "Error allocating d_MatA");
  SAFE_CALL(cudaMalloc((void **)&d_MatB, nBytes), "Error allocating d_MatB");
  SAFE_CALL(cudaMalloc((void **)&d_MatC, nBytes), "Error allocating d_MatC");
	SAFE_CALL(cudaMalloc((void **)&d_MatD, nBytes), "Error allocating d_MatC");

  // transfer data from host to device
  SAFE_CALL(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice), "Error copying d_MatA");
  SAFE_CALL(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice), "Error copying d_MatB");
  SAFE_CALL(cudaMemset(d_MatC, 0, nBytes), "Error copying d_MatB");
	SAFE_CALL(cudaMemset(d_MatD, 0, nBytes), "Error copying d_MatB");

  // kernel definition and launch
  dim3 block(T_SIZE, T_SIZE);
  // use other grid to experiment
  // dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
  dim3 grid((int)ceil((float)nx / T_SIZE), (int)ceil((float)nx / T_SIZE));

  // launch
  auto start_cpu = std::chrono::high_resolution_clock::now();
  multiplyMatrixOnGPU<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx);
  SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel");
  auto end_cpu =  std::chrono::high_resolution_clock::now();

  // measure total time
  std::chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
  printf("multiplyMatrixOnGPU elapsed %f ms\n", duration_ms.count());

  // SAFE_CALL kernel error
  SAFE_CALL(cudaGetLastError(), "Error with last error");

  // copy kernel result back to host side
  SAFE_CALL(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost), "Error copying d_MatC");

  // check device results
  //checkResult(hostRef, gpuRef, nx);

	// GPU TILE VERSION AND COMPARISSON
	// launch
	start_cpu = std::chrono::high_resolution_clock::now();
	multiplyMatrixOnGPUTiles<<<grid, block>>>(d_MatA, d_MatB, d_MatD, nx);
	SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel");
	end_cpu =  std::chrono::high_resolution_clock::now();

	// measure total time
	duration_ms = end_cpu - start_cpu;
	printf("multiplyMatrixOnGPUTiles elapsed %f ms\n", duration_ms.count());

	// SAFE_CALL kernel error
	SAFE_CALL(cudaGetLastError(), "Error with last error");

 	// copy kernel result back to host side
	SAFE_CALL(cudaMemcpy(gpuRefTiles, d_MatD, nBytes, cudaMemcpyDeviceToHost), "Error copying d_MatC");

	// check device results
	checkResult(gpuRef, gpuRefTiles, nx);
	// END GPU TILE VERSION AND COMPARISSON

  // free device global memory
  SAFE_CALL(cudaFree(d_MatA), "Error freeing memory");
  SAFE_CALL(cudaFree(d_MatB), "Error freeing memory");
  SAFE_CALL(cudaFree(d_MatC), "Error freeing memory");
	SAFE_CALL(cudaFree(d_MatD), "Error freeing memory");

  // free host memory
  free(h_A);
  free(h_B);
  free(hostRef);
  free(hostRefThreads);
  free(gpuRef);
	free(gpuRefTiles);

  // reset device
  SAFE_CALL(cudaDeviceReset(), "Error reseting");

  return (0);

}
