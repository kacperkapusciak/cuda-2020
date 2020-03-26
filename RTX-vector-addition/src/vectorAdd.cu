/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>
#include <iostream>
#include <chrono>
#include <string>
#include <fstream>
using namespace std::chrono;

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include <helper_cuda.h>
/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void
vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

/**
 * Host main routine
 */
int
main(void)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    int maxNumElements = 502267904;
    FILE *fdevice;
    fdevice = fopen("device.txt", "w");

    for (int numElements = 1; numElements < maxNumElements; numElements *= 10) {
    	for (int i = 0; i < 10; i++) {
    			size_t size = numElements * sizeof(float);


    			// Allocate the host input vector A
    			float *h_A = (float *)malloc(size);

    			// Allocate the host input vector B
    			float *h_B = (float *)malloc(size);

    			// Allocate the host output vector C
    			float *h_C = (float *)malloc(size);

    			// Verify that allocations succeeded
    			if (h_A == NULL || h_B == NULL || h_C == NULL)
    			{
    				fprintf(stderr, "Failed to allocate host vectors!\n");
    			}

    			// Initialize the host input vectors
    			for (int j = 0; j < numElements; ++j)
    			{
    				h_A[j] = rand()/(float)RAND_MAX;
    				h_B[j] = rand()/(float)RAND_MAX;
    			}

    			auto start = high_resolution_clock::now();

    			// Allocate the device input vector A
    			float *d_A = NULL;
    			err = cudaMalloc((void **)&d_A, size);

//    			 Allocate the device input vector B
    			float *d_B = NULL;
    			err = cudaMalloc((void **)&d_B, size);

    			// Allocate the device output vector C
    			float *d_C = NULL;
    			err = cudaMalloc((void **)&d_C, size);


    			// Copy the host input vectors A and B in host memory to the device input vectors in
    			// device memory
    		//        printf("Copy input data from the host memory to the CUDA device\n");
    			err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    			err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);


    			// Launch the Vector Add CUDA Kernel
    			int threadsPerBlock = 64;
    			int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    			printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    			vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    			err = cudaGetLastError();

    			// Copy the device result vector in device memory to the host result vector
    			// in host memory.
    		//        printf("Copy output data from the CUDA device to the host memory\n");
    			err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);


//    		    for (int i = 0; i < numElements; ++i)
//    		    {
//    		    	h_C[i] = h_A[i] + h_B[i];
//    		    }


    			// Free device global memory
    			err = cudaFree(d_A);
    			err = cudaFree(d_B);
    			err = cudaFree(d_C);

    			auto stop = high_resolution_clock::now();
    			auto duration = duration_cast<microseconds>(stop - start);
//    			device << i << "\t" << duration.count() << std::end;
    			fprintf(fdevice, "%d\t%d\n", numElements, duration.count());


    			// Free host memory
    			free(h_A);
    			free(h_B);
    			free(h_C);
    	}
    }


    fclose(fdevice);

    printf("Done\n");
    return 0;
}

