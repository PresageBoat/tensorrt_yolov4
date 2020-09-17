#include <cmath>
#include <stdio.h>
#include <cassert>
#include <iostream>
#include "mish.h"

namespace nvinfer1
{

	__device__ float tanh_activate_kernel(float x) { return (2 / (1 + expf(-2 * x)) - 1); }

	__device__ float softplus_kernel(float x, float threshold = 20) {
		if (x > threshold) return x;                // too large
		else if (x < -threshold) return expf(x);    // too small
		return logf(expf(x) + 1);
	}


	__device__ float mish_yashas(float x) {
		float e = __expf(x);
		if (x <= -18.0f)
		{
			return x * e;
		}
		float n = e * e + 2 * e;
		if (x < -5.0f)
		{
			return x * __fdividef(n, n + 2);
		}
		return x - 2 * __fdividef(x, n + 2);
	}

	__global__ void mish_kernel(const float *input, float *output, int num_elem) {

		//int idx = threadIdx.x + blockDim.x * blockIdx.x;
		int idx = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
		if (idx >= num_elem) return;
		//output[idx] = input[idx] * tanh_activate_kernel(softplus_kernel(input[idx]));
		output[idx] = mish_yashas(input[idx]);
	}

	void MishPlugin::forwardGpu(const float *const * inputs, float* output, cudaStream_t stream, int batchSize) {
		int block_size = thread_count_;
		int grid_size = (input_size_ * batchSize + block_size - 1) / block_size;
		mish_kernel << <grid_size, block_size >> > (inputs[0], output, input_size_ * batchSize);
	}


	int MishPlugin::enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream)
	{
		forwardGpu((const float *const *)inputs, (float*)outputs[0], stream, batchSize);
		return 0;
	}



}

