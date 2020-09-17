#ifndef _COMMON_H_
#define _COMMON_H_

#define DEAL_BATCHSIZE 1
#define SUPPORT_OBJ_MAXSIZE 1000

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


//class id ==> class name 
#define PERSON 0
#define Car 1






#ifdef __linux__
#define API_EXPORTS extern "C"
#else
#ifdef WIN_API_EXPORT
#define API_EXPORTS __declspec(dllexport) 
#else
#define API_EXPORTS __declspec(dllimport) 
#endif
#endif


//base object property
struct BaseObjProp {
	//x y w h
	int bbox[4];
	int class_id;
	float det_confidence;
	float class_confidence;
	BaseObjProp() {
		bbox[4] = {0};
		class_id = -1;
		det_confidence = 0.f;
		class_confidence = 0.f;
	}
};

//image property
struct ImageProp {
	int base_obj_size;
	BaseObjProp base_obj_prop[SUPPORT_OBJ_MAXSIZE];
	ImageProp() {
		base_obj_size = 0;
	}
};

//images property
struct DetectImagesResult {
	ImageProp image_prop[DEAL_BATCHSIZE];
};

//input data range by RGB
//When the batch is greater than 1, all inputs must be of the same width and height
struct InputData {
	unsigned char* pdata;
	int width;
	int height;
	int nchannel;
	int batchsize;
	InputData() {
		width = 0;
		height = 0;
		nchannel = 3;
		batchsize = DEAL_BATCHSIZE;
	}
};

API_EXPORTS int  Image_Init_Person(const char* engine_filepath);

API_EXPORTS int  Image_Free_Person();

/*
*inputdata :输入的数据，经过预处理后的图像，数据长度为416*416，按照RGBRGB方式排列，
*batchsize：当前处理的图像数
*width：原始输入的图像的宽度
*height:原始输入的图像的高度
*result:检测结果
*/
API_EXPORTS int  Image_Detection_Inference_Person(float* inputdata, const int batchsize,const int width,const int height,DetectImagesResult* result);


API_EXPORTS char* Image_Get_Version_Person();


#endif //_COMMON_H_