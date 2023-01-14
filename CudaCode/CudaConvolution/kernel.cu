#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include<memory>
#include<stdexcept>
#include<string>
using namespace std;

__device__ bool isInMat(int x, int y, int width, int height) {
	return x >= 0 && y >= 0 && x < width&& y < height;
}

__device__ int posInMatReflectBorder(int row, int col, int width, int height) {
	int x = col, y = row;
	if (col < 0)
		x = -col;
	if (row < 0)
		y = -row;
	if (col >= width)
		x = width - (col + 2 - width);
	if (row >= height)
		y = height - (row + 2 - height);

	return (y * width + x);
}

__global__ void convolvKernel(float* dst, const float* src, const float* filter, int width, int height, int filterWidth, int filterHeight/*, int x, int y*/)
{

	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int pos = y * width + x;
	bool isInArray = y < height&& x < width;
	float sum = 0;
	if (!isInArray)
		return;
	for (int filter_row = 0, row = (y - filterHeight / 2); filter_row < filterHeight; filter_row++, row++) {
		for (int filter_col = 0, col = x - filterWidth / 2; filter_col < filterWidth; filter_col++, col++) {
			sum += src[posInMatReflectBorder(row, col, width, height)] * filter[filter_row * filterWidth + filter_col];
		}
	}
	dst[pos] = sum;
}

#define ASSERT_CUDA_SUCCESS(cudaStatus,msg){      \
	if (cudaStatus != cudaSuccess) {      \
		throw runtime_error(msg);		  \
	}								      \
}

void  convolvWithCuda(float* dst, const float* src, const float* filter, int width, int height, int filterWidth, int filterHeight)
{
	cudaError_t cudaStatus;
	size_t size = width * height;

	int threadsPerBlock = 32;
	int numBlocks_x = (width / threadsPerBlock) + 1;
	int numBlocks_y = (height / threadsPerBlock) + 1;

	dim3 threads_per_block_dim(threadsPerBlock, threadsPerBlock);
	dim3 blocks_grid_dim(numBlocks_x, numBlocks_y);

	convolvKernel << <blocks_grid_dim, threads_per_block_dim >> > (dst, src, filter, width, height, filterWidth, filterHeight);

	cudaStatus = cudaGetLastError();
	ASSERT_CUDA_SUCCESS(cudaStatus, "addKernel launch failed: " + string(cudaGetErrorString(cudaStatus)) + "\n");

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	ASSERT_CUDA_SUCCESS(cudaStatus, "cudaDeviceSynchronize returned error code" + to_string(cudaStatus) + "after launching addKernel!\n");
}

