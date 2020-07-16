#ifndef _COMMON_H_
#define _COMMON_H_

//error information list
#define INIT_SUCCESS 0 //init success
#define RUN_DETECTION_SUCCESS 1 //run detection success
#define FREE_SUCCESS 2 //free success

#define NO_GPU_DEVICE -10000 //
#define GPU_DEVICE_INAVALIABLE  -10001//
#define GPU_FREE_MEMORY_CHECK_FAILED -10002//
#define LOAD_TENSORRT_ENGINE_FAILED -10003
#define FREE_FAILED -10004 //
#define CREATE_RUNTIME_FAILED  -10005
#define CREATE_ENGINE_FAILED -10006
#define CREATE_CONTEXT_FAILED  -10007


#define INPUT_BATCHSIZE_ERROR -10008
#define CREATE_INFERENCE_STREAM_FAILED  -10009
#define CREATE_INFERENCE_DATA_BINDING_FAILED -10010
#define CUDASTREAM_SYNC_FAILED -10011



#ifdef __linux__
#define API_EXPORTS extern "C"
#else
#ifdef WIN_API_EXPORT
#define API_EXPORTS __declspec(dllexport)
#else
#define API_EXPORTS __declspec(dllimport)
#endif
#endif


//
//struct BaseObjProp {
//	//x y w h
//	int bbox[4];
//	int class_id;
//	float det_confidence;
//	float class_confidence;
//};
//
//
//struct ImageProp {
//	int base_obj_size;
//	BaseObjProp base_obj_prop[1000];
//};
//
//struct DetectImagesResult {
//	int image_size;
//	ImageProp image_prop[64];
//};

API_EXPORTS int  Image_Init_Person(const char* engine_filepath);

API_EXPORTS int  Image_Free_Person();

API_EXPORTS int  Image_Detection_Inference_Person(float* inputdata, float* prob, const int batchsize);

API_EXPORTS char* Image_Get_Version_Person();


#endif //_COMMON_H_