#include <stdio.h>
#include <iostream>
#include <chrono>
#include <string>
#include <fstream>
#include <assert.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>

using namespace std::chrono;


template <typename T>
__global__
void multiplicationDevice(const T *A, const T *B, T *C, unsigned long N, unsigned long M) {
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < M && j < N) {
		T sum = 0;

		for(int k = 0; k < M; k++)
			sum += A[i + k * M] * B[k + j * M];

		C[i + j * M] = sum;
	}
}

template <typename T>
void multiplicationHost(const T* A, const T* B, T* C, unsigned long N, unsigned long M) {
	for(unsigned long i = 0; i < N; i++) {
		for(unsigned long j = 0; j < M; j++) {
			T sum = 0;

			for(int k = 0; k < M; k++)
				sum += A[i + k * M] * B[k + j * M];

			C[i + j * M] = sum;
		}
	}
}

inline cudaError_t checkCuda(cudaError_t result) {
	if (result != cudaSuccess) {
		std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
		assert(result == cudaSuccess);
	}
	return result;
}

template <typename T>
void allocateDevice(T* A, const size_t size) {
	checkCuda(cudaMalloc((void**)A, size));
}

template <typename T>
void freeDevice(T* A) {
	checkCuda(cudaFree(A));
}

template <typename T>
void allocateHost(T** A, const size_t size) {
	*A = (T*)malloc(size);

	if (A == nullptr) {
		std::cout << "Unable to allocate host memory" << std::endl;
		exit(0);
	}
}

void initializeMatrix(float* A, unsigned long numElements) {
	for(int i = 0; i < numElements; i++)
		A[i] = rand()/(float)RAND_MAX;
}

void initializeMatrix(int* A, unsigned long numElements) {
	for(int i = 0; i < numElements; i++)
		A[i] = rand() % 10;
}

template <typename T>
void printMatrix(T* A, unsigned long N, unsigned long M) {
	for(int j = 0; j < N; j++) {
		for(int i = 0; i < M; i++)
			std::cout << A[j * N + i] << " ";
		std::cout << std::endl;
	}
}

template <typename T>
void copyMemoryFromHostToDevice(T* d_A, T* h_A, const size_t size) {
    checkCuda(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
}

template <typename T>
void copyMemoryFromDeviceToHost(T* d_A, T* h_A, const size_t size) {
    checkCuda(cudaMemcpy(d_A, h_A, size, cudaMemcpyDeviceToHost));
}

template <typename T>
void benchmarkMalloc(unsigned long N, unsigned long M, bool isDevice, FILE** file) {
	int deviceId;
	int numberOfSMs;

	cudaGetDevice(&deviceId);
	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

	const unsigned long numElements = N * M;
	const size_t size = numElements * sizeof(T);

	auto start = high_resolution_clock::now();

	T *h_A = nullptr, *h_B = nullptr, *h_C = nullptr, *h_deviceResult = nullptr;
	allocateHost(&h_A, size);
	allocateHost(&h_B, size);
	allocateHost(&h_C, size);
	allocateHost(&h_deviceResult, size);

	initializeMatrix(h_A, numElements);
	initializeMatrix(h_B, numElements);

	if (!isDevice) {
		multiplicationHost(h_A, h_B, h_C, N, M);

		auto stopHost = high_resolution_clock::now();
		auto duration = duration_cast<microseconds>(stopHost - start);
		fprintf(*file, "%d\t%d\n", numElements, duration.count());
//		std::cout << numElements << "\t" << duration.count() << std::endl;
	}


	T *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

	allocateDevice(&d_A, size);
	allocateDevice(&d_B, size);
	allocateDevice(&d_C, size);

	copyMemoryFromHostToDevice(d_A, h_A, size);
	copyMemoryFromHostToDevice(d_B, h_B, size);

	int threadsPerBlock = 256;

	dim3 blockSize(threadsPerBlock, threadsPerBlock);
	int blockX = (N + threadsPerBlock - 1) / threadsPerBlock;
	int blockY = (M + threadsPerBlock - 1) / threadsPerBlock;
	dim3 blocksPerGrid = dim3(blockX, blockY);

	if (isDevice)
		multiplicationDevice<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N, M);


	copyMemoryFromDeviceToHost(h_deviceResult, d_C, size);


	if (isDevice) {
		auto stopDevice = high_resolution_clock::now();
		auto duration = duration_cast<microseconds>(stopDevice - start);
		fprintf(*file, "%d\t%d\n", numElements, duration.count());
//		std::cout << numElements << "\t" << duration.count() << std::endl;
	}

	freeDevice(d_A);
	freeDevice(d_B);
	freeDevice(d_C);

	free(h_A);
	free(h_B);
	free(h_C);
	free(h_deviceResult);
}

template <typename T>
void benchmarkUnified(unsigned long N, unsigned long M, bool isDevice, FILE** file) {
	int deviceId;
	int numberOfSMs;

	cudaGetDevice(&deviceId);
	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

	const unsigned long numElements = N * M;
	const size_t size = numElements * sizeof(T);

	T *A;
	T *B;
	T *C;

	auto start = high_resolution_clock::now();

	checkCuda(cudaMallocManaged(&A, size));
	checkCuda(cudaMallocManaged(&B, size));
	checkCuda(cudaMallocManaged(&C, size));

	initializeMatrix(A, numElements);
	initializeMatrix(B, numElements);

	size_t threadsPerBlock;

	threadsPerBlock = 256;
	dim3 blockSize(threadsPerBlock, threadsPerBlock);
	int blockX = (N + threadsPerBlock - 1) / threadsPerBlock;
	int blockY = (M + threadsPerBlock - 1) / threadsPerBlock;
	dim3 blocksPerGrid = dim3(blockX, blockY);

	cudaError_t addVectorsErr;
	cudaError_t asyncErr;



	if (isDevice)
		multiplicationDevice<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N, M);
	else
		multiplicationHost(A, B, C, N, M);




	addVectorsErr = cudaGetLastError();
	if(addVectorsErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(addVectorsErr));

	asyncErr = cudaDeviceSynchronize();
	if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));


	cudaFree(A);
	cudaFree(B);
	cudaFree(C);

	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	fprintf(*file, "%d\t%d\n", numElements, duration.count());
//	std::cout << numElements << "\t" << duration.count() << std::endl;
}

template <typename T>
void benchmarkUnifiedPrefetched(unsigned long N, unsigned long M, bool isDevice, FILE** file) {
	int deviceId;
	int numberOfSMs;

	cudaGetDevice(&deviceId);
	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

	const unsigned long numElements = N * M;
	const size_t size = numElements * sizeof(T);

	T *A;
	T *B;
	T *C;

	auto start = high_resolution_clock::now();

	checkCuda(cudaMallocManaged(&A, size));
	checkCuda(cudaMallocManaged(&B, size));
	checkCuda(cudaMallocManaged(&C, size));

	checkCuda(cudaMemPrefetchAsync(A, size, cudaCpuDeviceId));
	checkCuda(cudaMemPrefetchAsync(B, size, cudaCpuDeviceId));
	checkCuda(cudaMemPrefetchAsync(C, size, cudaCpuDeviceId));

	initializeMatrix(A, numElements);
	initializeMatrix(B, numElements);

	checkCuda(cudaMemPrefetchAsync(A, size, deviceId));
	checkCuda(cudaMemPrefetchAsync(B, size, deviceId));
	checkCuda(cudaMemPrefetchAsync(C, size, deviceId));

	size_t threadsPerBlock;

	threadsPerBlock = 256;
	dim3 blockSize(threadsPerBlock, threadsPerBlock);
	int blockX = (N + threadsPerBlock - 1) / threadsPerBlock;
	int blockY = (M + threadsPerBlock - 1) / threadsPerBlock;
	dim3 blocksPerGrid = dim3(blockX, blockY);

	cudaError_t addVectorsErr;
	cudaError_t asyncErr;



	if (isDevice)
		multiplicationDevice<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N, M);
	else
		multiplicationHost(A, B, C, N, M);


	addVectorsErr = cudaGetLastError();
	if(addVectorsErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(addVectorsErr));

	asyncErr = cudaDeviceSynchronize();
	if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));

	cudaMemPrefetchAsync(C, size, cudaCpuDeviceId);

	cudaFree(A);
	cudaFree(B);
	cudaFree(C);

	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	fprintf(*file, "%d\t%d\n", numElements, duration.count());
}


int main() {
  auto start = high_resolution_clock::now();

  FILE *fileMallocIntHost, *fileMallocFloatHost, *fileMallocIntDevice, *fileMallocFloatDevice;
  fileMallocIntHost = fopen("mallocIntHost.txt", "w");
  fileMallocFloatHost = fopen("mallocFloatHost.txt", "w");
  fileMallocIntDevice = fopen("mallocIntDevice.txt", "w");
  fileMallocFloatDevice = fopen("mallocFloatDevice.txt", "w");


  for (int N = 8; N*N <= 2000000; N *= 2) {
	  for (int i = 0; i < 10; i++) {
		  benchmarkMalloc<int>(N, N, false, &fileMallocIntHost);
		  benchmarkMalloc<float>(N, N, false, &fileMallocFloatHost);
		  benchmarkMalloc<int>(N, N, true, &fileMallocIntDevice);
		  benchmarkMalloc<float>(N, N, true, &fileMallocFloatDevice);
	  }
  }

  fclose(fileMallocIntHost);
  fclose(fileMallocFloatHost);
  fclose(fileMallocIntDevice);
  fclose(fileMallocFloatDevice);

  std::cout << "Standard malloc done" << std::endl;

  FILE *fileUnifiedIntHost, *fileUnifiedFloatHost, *fileUnifiedIntDevice, *fileUnifiedFloatDevice;
  fileUnifiedIntHost = fopen("unifiedIntHost.txt", "w");
  fileUnifiedFloatHost = fopen("unifiedFloatHost.txt", "w");
  fileUnifiedIntDevice = fopen("unifiedIntDevice.txt", "w");
  fileUnifiedFloatDevice = fopen("unifiedFloatDevice.txt", "w");

  for (int N = 8; N*N <= 2000000; N *= 2) {
	  for (int i = 0; i < 10; i++) {
		  benchmarkUnified<int>(N, N, false, &fileUnifiedIntHost);
		  benchmarkUnified<float>(N, N, false, &fileUnifiedFloatHost);
		  benchmarkUnified<int>(N, N, true, &fileUnifiedIntDevice);
		  benchmarkUnified<float>(N, N, true, &fileUnifiedFloatDevice);
	  }
  }

  fclose(fileUnifiedIntHost);
  fclose(fileUnifiedFloatHost);
  fclose(fileUnifiedIntDevice);
  fclose(fileUnifiedFloatDevice);

  std::cout << "Unified memory done" << std::endl;

  FILE *fileUnifiedPrefetchedIntHost, *fileUnifiedPrefetchedFloatHost, *fileUnifiedPrefetchedIntDevice, *fileUnifiedPrefetchedFloatDevice;
  fileUnifiedPrefetchedIntHost = fopen("unifiedPrefetchedIntHost.txt", "w");
  fileUnifiedPrefetchedFloatHost = fopen("unifiedPrefetchedFloatHost.txt", "w");
  fileUnifiedPrefetchedIntDevice = fopen("unifiedPrefetchedIntDevice.txt", "w");
  fileUnifiedPrefetchedFloatDevice = fopen("unifiedPrefetchedFloatDevice.txt", "w");

  for (int N = 8; N*N <= 2000000; N *= 2) {
	  for (int i = 0; i < 10; i++) {
		  benchmarkUnifiedPrefetched<int>(N, N, false, &fileUnifiedPrefetchedIntHost);
		  benchmarkUnifiedPrefetched<float>(N, N, false, &fileUnifiedPrefetchedFloatHost);
		  benchmarkUnifiedPrefetched<int>(N, N, true, &fileUnifiedPrefetchedIntDevice);
		  benchmarkUnifiedPrefetched<float>(N, N, true, &fileUnifiedPrefetchedFloatDevice);
	  }
  }

  fclose(fileUnifiedPrefetchedIntHost);
  fclose(fileUnifiedPrefetchedFloatHost);
  fclose(fileUnifiedPrefetchedIntDevice);
  fclose(fileUnifiedPrefetchedFloatDevice);

  std::cout << "Unified memory and prefetching data done" << std::endl;

  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<seconds>(stop - start);
  std::cout << "Program run for " << duration.count() << " seconds" << std::endl;

  return 0;
}
