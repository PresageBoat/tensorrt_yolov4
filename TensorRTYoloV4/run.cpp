#include "../Export/include/PersonDetectionSdk.h"
#include "yolov4.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "cuda_runtime_api.h"

std::unique_ptr<YoloV4Detection> object_detector;

API_EXPORTS int Image_Init_Person(const char* engine_filepath)
{
	object_detector.reset(new YoloV4Detection());
	return object_detector->Init(engine_filepath);
}

API_EXPORTS int Image_Free_Person()
{
	try
	{
		if (nullptr != object_detector)
		{
			object_detector.reset();
		}
	}
	catch (const std::exception&)
	{
		return FREE_FAILED;
	}
	return FREE_SUCCESS;
}

API_EXPORTS int Image_Detection_Inference_Person(float* inputdata, float* prob, const int batchsize)
{
	return object_detector->RunInference(inputdata, prob, batchsize);
}

API_EXPORTS char* Image_Get_Version_Person()
{
	char* version = "Person_detect_Version.1.0.0";
	return version;
}