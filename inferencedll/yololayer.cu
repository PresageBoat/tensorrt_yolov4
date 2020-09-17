#include "yololayer.h"

using namespace YoloParam;

namespace nvinfer1
{
	__device__ float Logist(float data) { return 1. / (1. + exp(-data)); };

	__global__ void CalDetection(const float *input, float *output, int noElements,
		int yoloWidth, int yoloHeight, const float anchors[CHECK_COUNT * 2], int classes, int outputElem) {

		int idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= noElements) return;

		int total_grid = yoloWidth * yoloHeight;
		int bnIdx = idx / total_grid;
		idx = idx - total_grid * bnIdx;
		int info_len_i = 5 + classes;
		const float* curInput = input + bnIdx * (info_len_i * total_grid * CHECK_COUNT);

		for (int k = 0; k < 3; ++k) {
			int class_id = 0;
			float max_cls_prob = 0.0;
			for (int i = 5; i < info_len_i; ++i) {
				float p = Logist(curInput[idx + k * info_len_i * total_grid + i * total_grid]);
				if (p > max_cls_prob) {
					max_cls_prob = p;
					class_id = i - 5;
				}
			}
			float box_prob = Logist(curInput[idx + k * info_len_i * total_grid + 4 * total_grid]);
			if (max_cls_prob < IGNORE_THRESH || box_prob < IGNORE_THRESH) continue;

			float *res_count = output + bnIdx * outputElem;
			int count = (int)atomicAdd(res_count, 1);
			if (count >= MAX_OUTPUT_BBOX_COUNT) return;
			char* data = (char *)res_count + sizeof(float) + count * sizeof(Detection);
			Detection* det = (Detection*)(data);

			int row = idx / yoloWidth;
			int col = idx % yoloWidth;

			//Location
			det->bbox[0] = (col + Logist(curInput[idx + k * info_len_i * total_grid + 0 * total_grid])) * INPUT_W / yoloWidth;
			det->bbox[1] = (row + Logist(curInput[idx + k * info_len_i * total_grid + 1 * total_grid])) * INPUT_H / yoloHeight;
			det->bbox[2] = exp(curInput[idx + k * info_len_i * total_grid + 2 * total_grid]) * anchors[2 * k];
			det->bbox[3] = exp(curInput[idx + k * info_len_i * total_grid + 3 * total_grid]) * anchors[2 * k + 1];
			det->det_confidence = box_prob;
			det->class_id = class_id;
			det->class_confidence = max_cls_prob;
		}
	}

	void YoloLayerPlugin::forwardGpu(const float *const * inputs, float* output, cudaStream_t stream, int batchSize) {
		float* devAnchor = nullptr;
		size_t AnchorLen = CHECK_COUNT * 2 * sizeof(float);
		CUDA_CHECK(cudaMalloc((void**)&devAnchor, AnchorLen));

		int outputElem = 1 + MAX_OUTPUT_BBOX_COUNT * sizeof(Detection) / sizeof(float);

		for (int idx = 0; idx < batchSize; ++idx) {
			CUDA_CHECK(cudaMemset(output + idx * outputElem, 0, sizeof(float)));
		}
		int numElem = 0;
		for (unsigned int i = 0; i < mYoloKernel.size(); ++i)
		{
			const auto& yolo = mYoloKernel[i];
			numElem = yolo.width*yolo.height*batchSize;
			if (numElem < mThreadCount)
				mThreadCount = numElem;
			CUDA_CHECK(cudaMemcpy(devAnchor, yolo.anchors, AnchorLen, cudaMemcpyHostToDevice));

			// gpu data output
			//float* cpuPtr = new float[6];
			//cudaMemcpy(cpuPtr, devAnchor, 6 *sizeof(float),cudaMemcpyDeviceToHost);
			//for (int j=0;j< 6;j++){
			//	std::cout << cpuPtr[j] << ",";
			//	//std::cout << *cpuPtr++<< ",";
			//}
			//std::cout<<std::endl;
			//std::cout << "show next devAnchor" << std::endl;
			//
			CalDetection << < (yolo.width*yolo.height*batchSize + mThreadCount - 1) / mThreadCount, mThreadCount >> >
				(inputs[i], output, numElem, yolo.width, yolo.height, devAnchor, mClassCount, outputElem);
		}

		CUDA_CHECK(cudaFree(devAnchor));
	}


	int YoloLayerPlugin::enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream)
	{
		//assert(batchSize == 1);
		//GPU
		//CUDA_CHECK(cudaStreamSynchronize(stream));
		forwardGpu((const float *const *)inputs, (float*)outputs[0], stream, batchSize);

		return 0;
	}

}
