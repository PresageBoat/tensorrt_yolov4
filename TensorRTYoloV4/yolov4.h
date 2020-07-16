#ifndef _YOLOV4_H_
#define _YOLOV4_H_

#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include "yololayer.h"
#include "mish.h"
#include "../Export/include/PersonDetectionSdk.h"
#include "yoloparam.h"

#define NOMINMAX

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

#define BATCH_SIZE 128
#define GPU_DEVICE 0  // GPU id
#define NMS_THRESH 0.4
#define BBOX_CONF_THRESH 0.5

// stuff we know about the network and the input/output blobs
static const int INPUT_H = YoloParam::INPUT_H;
static const int INPUT_W = YoloParam::INPUT_W;
static const int DETECTION_SIZE = sizeof(YoloParam::Detection) / sizeof(float);//7
// we assume the yolo layer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
static const int OUTPUT_SIZE = YoloParam::MAX_OUTPUT_BBOX_COUNT * DETECTION_SIZE + 1;
static const char* INPUT_BLOB_NAME = "data";
static const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;

using namespace nvinfer1;

class YoloV4Detection
{
public:
	YoloV4Detection();
	~YoloV4Detection();
	int Init(const std::string enginepath);
	int RunInference(float* inputdata, float* outputprob, const int batchsize);

private:
	int doInference(IExecutionContext& context, float* input, float* output, int batchSize);
	//cv::Rect get_rect(cv::Mat& img, float bbox[4]);
	//cv::Mat preprocess_img(cv::Mat& img);
	//void nms(std::vector<YoloParam::Detection>& res, float *output, float nms_thresh = NMS_THRESH);
	//float iou(float lbox[4], float rbox[4]);
	//bool cmp(YoloParam::Detection& a, YoloParam::Detection& b);

private:
	size_t trt_size{ 0 };

	char *trtModelStream{ nullptr };
	char* engine_filepath;

	void* buffers[2];	//buffers for input  and output

	IRuntime* runtime;
	ICudaEngine* engine;
	IExecutionContext* context;
	cudaStream_t stream;//stream for inference
};

#endif //_YOLOV4_H_
