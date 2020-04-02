#include <stdio.h>
#include <iostream>
#include <chrono>
#include <string>
#include <fstream>

#include <cuda_runtime.h>
#include <helper_cuda.h>

using namespace std::chrono;


template <typename T>
__global__ void additionDevice1D(const T* A, const T* B, T* C, unsigned long numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

template <typename T>
__global__ void hadamardDevice1D(const T *A, const T *B, T *C, unsigned long numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements) {
        C[i] = A[i] * B[i];
    }
}

template <typename T>
__global__ void additionDevice2D(const T *A, const T *B, T *C, unsigned long N, unsigned long M) {
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	int k = i + j * M;

	if(i < M && j < N) {
		C[k] = A[k] + B[k];
	}
}

template <typename T>
__global__ void hadamardDevice2D(const T *A, const T *B, T *C, unsigned long N, unsigned long M) {
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	int k = i + j * M;

	if(i < M && j < N) {
		C[k] = A[k] * B[k];
	}
}

template <typename T>
void additionHost(const T* A, const T* B, T* C, unsigned long numElements) {
	for (int i = 0; i < numElements; i++) {
		C[i] = A[i] + B[i];
	}
}

template <typename T>
void hadamardHost(const T* A, const T* B, T* C, unsigned long numElements) {
	for (int i = 0; i < numElements; i++) {
		C[i] = A[i] * B[i];
	}
}

template <typename T>
void testEqual(const T* A, const T* B, unsigned long numElements) {
	for (int i = 0; i < numElements; i++)
		if (A[i] != B[i])
			std::cout << "FAIL" << std::endl;

	std::cout << "PASS" << std::endl;
}

template <typename T>
void allocateDevice(T* A, const size_t size) {
	cudaError_t err = cudaSuccess;
	err = cudaMalloc((void**)A, size);

	if (err != cudaSuccess) {
		std::cout << "Unable to allocate device memory" << std::endl;
		exit(0);
	}
}

template <typename T>
void freeDevice(T* A) {
	cudaError_t err = cudaSuccess;
	err = cudaFree(A);

	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to free device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
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
void copyMemoryFromHostToDevice(T* d_A, T* h_A, const size_t size) {
	cudaError_t err = cudaSuccess;
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to copy memory from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

template <typename T>
void copyMemoryFromDeviceToHost(T* d_A, T* h_A, const size_t size) {
	cudaError_t err = cudaSuccess;
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyDeviceToHost);

	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to copy memory from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

template <typename T>
void printMatrix(T* A, unsigned long N, unsigned long M) {
	for(int j = 0; j < N; j++) {
		for(int i = 0; i < M; i++)
			std::cout << A[j * N + i] << " ";
		std::cout << std::endl;
	}
}


int main(int argc, char **argv) {
	cudaError_t err = cudaSuccess;
	FILE *fdevice;
	fdevice = fopen(".txt", "w");

	for (int i = 10; i <= 10000; i *= 10) {
		for (int j = 0; j < 10; j++) {

			const unsigned long N = i, M = i;
			const unsigned long numElements = N * M;
			std::cout << "numElements = " << numElements << std::endl;
			const size_t matrixSizeInt = numElements * sizeof(int);
		//	const size_t matrixSizeFloat = numElements * sizeof(float);

//			auto hostStart = high_resolution_clock::now();

			int *h_A = nullptr, *h_B = nullptr, *h_C = nullptr, *h_deviceResult = nullptr;
			allocateHost(&h_A, matrixSizeInt);
			allocateHost(&h_B, matrixSizeInt);
			allocateHost(&h_C, matrixSizeInt);
			allocateHost(&h_deviceResult, matrixSizeInt);

			initializeMatrix(h_A, numElements);
			initializeMatrix(h_B, numElements);


//			hadamardHost(h_A, h_B, h_C, numElements);
//			auto hostStop = high_resolution_clock::now();
//			auto hostDuration = duration_cast<microseconds>(hostStop - hostStart);
//			fprintf(fdevice, "%d\t%d\n", numElements, hostDuration.count());


			int *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

			auto start = high_resolution_clock::now();

			allocateDevice(&d_A, matrixSizeInt);
			allocateDevice(&d_B, matrixSizeInt);
			allocateDevice(&d_C, matrixSizeInt);

			copyMemoryFromHostToDevice(d_A, h_A, matrixSizeInt);
			copyMemoryFromHostToDevice(d_B, h_B, matrixSizeInt);

			int threadsPerBlock = 256;
			int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
			printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

			hadamardDevice1D<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

			copyMemoryFromDeviceToHost(h_deviceResult, d_C, matrixSizeInt);


		//	additionHost(h_A, h_B, h_C, numElements);

		//	testEqual(h_C, h_deviceResult, numElements);


			freeDevice(d_A);
			freeDevice(d_B);
			freeDevice(d_C);

			auto stop = high_resolution_clock::now();
			auto deviceDuration = duration_cast<microseconds>(stop - start);
			fprintf(fdevice, "%d\t%d\n", numElements, deviceDuration.count());


			free(h_A);
			free(h_B);
			free(h_C);
			free(h_deviceResult);
		}
	}

	std::cout << "DONE!" << std::endl;
	return 0;
}

