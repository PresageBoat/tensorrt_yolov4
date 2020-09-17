#include "yolov4.h"

#include <algorithm>


REGISTER_TENSORRT_PLUGIN(MishPluginCreator);
REGISTER_TENSORRT_PLUGIN(YoloPluginCreator);


YoloV4Detection::YoloV4Detection()
{
}

YoloV4Detection::~YoloV4Detection()
{
	if (nullptr!= trtModelStream)
	{
		delete[] trtModelStream;
	}

	if (nullptr!= buffers[0])
	{
		cudaFree(buffers[0]);
	}
	if (nullptr != buffers[1])
	{
		cudaFree(buffers[1]);
	}
	if (nullptr != stream)
	{
		cudaStreamDestroy(stream);
	}
}


int YoloV4Detection::Init(const std::string enginepath) {

	//check gpu device
	int deviec;
	if (cudaSuccess!=cudaGetDevice(&deviec))
	{
		return NO_GPU_DEVICE;
	}

// This function checks the availability of GPU #device_id.
// It attempts to create a context on the device by calling cudaFree(0).
// cudaSetDevice() alone is not sufficient to check the availability.
// It lazily records device_id, however, does not initialize a
// context. So it does not know if the host thread has the permission to use
// the device or not.
// In a shared environment where the devices are set to EXCLUSIVE_PROCESS
  // or EXCLUSIVE_THREAD mode, cudaSetDevice() returns cudaSuccess
  // even if the device is exclusively occupied by another process or thread.
  // Cuda operations that initialize the context are needed to check
  // the permission. cudaFree(0) is one of those with no side effect,
  // except the context initialization.
	bool device_statue= ((cudaSuccess == cudaSetDevice(GPU_DEVICE)) &&
		(cudaSuccess == cudaFree(0)));
	cudaGetLastError();
	if (!device_statue)
	{
		return GPU_DEVICE_INAVALIABLE;
	}

	//check free memory
	void *device_data;
	cudaError_t err = cudaMalloc(&device_data, 1.5*(1<<30));
	if (err != cudaSuccess) {
		return GPU_FREE_MEMORY_CHECK_FAILED;
	}
	else
	{
		cudaFree(device_data);
	}

	std::ifstream file(enginepath, std::ios::binary);
	if (file.good()) {
		file.seekg(0, file.end);
		trt_size = file.tellg();
		file.seekg(0, file.beg);
		trtModelStream = new char[trt_size];
		assert(trtModelStream);
		file.read(trtModelStream, trt_size);
		file.close();
	}
	else
	{
		return LOAD_TENSORRT_ENGINE_FAILED;
	}

	//
	runtime = createInferRuntime(gLogger);
	if (runtime==nullptr)
	{
		return CREATE_RUNTIME_FAILED;
	}
	engine = runtime->deserializeCudaEngine(trtModelStream, trt_size);
	if (engine == nullptr)
	{
		return CREATE_ENGINE_FAILED;
	}
	context = engine->createExecutionContext();
	if (context == nullptr)
	{
		return CREATE_CONTEXT_FAILED;
	}

	cudaError_t err0=cudaStreamCreate(&stream);
	if (err0 != cudaSuccess)
	{
		return CREATE_INFERENCE_STREAM_FAILED;
	}


	// malloc buf  for input data  and output data 
	cudaError_t err1 = cudaMalloc(&buffers[0], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float));
	if (err1 != cudaSuccess) {
		return GPU_FREE_MEMORY_CHECK_FAILED;
	}
	cudaError_t err2 = cudaMalloc(&buffers[1], BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
	if (err2 != cudaSuccess) {
		return GPU_FREE_MEMORY_CHECK_FAILED;
	}



	return INIT_SUCCESS;
}

int YoloV4Detection::RunInference(float* inputdata,  const int batchsize, const int width, const int height, DetectImagesResult* result) {

	//The batch size is recommended to be integer multiples of 4
	if (batchsize != 1)
	{
		if (batchsize & 3 != 0)
		{
			return INPUT_BATCHSIZE_ERROR;
		}
	}

	int res = doInference(*context, inputdata, prob, batchsize);

	if (res != 1)
	{
		//tensorrt inference failed
		return res;
	}
	else {
		//after inference and deal tensorrt result to use struct
		std::vector<std::vector<YoloParam::Detection>> batch_res(batchsize);
		//nms deal
		for (int b = 0; b < batchsize; b++) {
			auto& res = batch_res[b];
			nms(res, &prob[b * OUTPUT_SIZE]);
		}

		//get rect and other params
		for (int b = 0; b < batchsize; b++) {
			auto& res = batch_res[b];
			for (size_t j = 0; j < res.size(); j++) {
				cv::Rect r = get_rect(width, height, res[j].bbox);
				result->image_prop[b].base_obj_size = j + 1;
				result->image_prop[b].base_obj_prop[j].bbox[0] = r.x;
				result->image_prop[b].base_obj_prop[j].bbox[1] = r.y;
				result->image_prop[b].base_obj_prop[j].bbox[2] = r.width;
				result->image_prop[b].base_obj_prop[j].bbox[3] = r.height;
				result->image_prop[b].base_obj_prop[j].class_confidence = res[j].class_confidence;
				result->image_prop[b].base_obj_prop[j].det_confidence = res[j].det_confidence;
				result->image_prop[b].base_obj_prop[j].class_id = res[j].class_id;
			}
		}
	}

	return res;

}

int YoloV4Detection::doInference(IExecutionContext& context, float* input, float* output, int batchSize) {

	const ICudaEngine& engine = context.getEngine();
	// Pointers to input and output device buffers to pass to engine.
	// Engine requires exactly IEngine::getNbBindings() number of buffers.
	if (engine.getNbBindings()!=2)
	{
		return CREATE_INFERENCE_DATA_BINDING_FAILED;
	}
	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// Note that indices are guaranteed to be less than IEngine::getNbBindings()
	const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
	const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

	// DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
	CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
	context.enqueue(batchSize, buffers, stream, nullptr);
	CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaError_t err1 = cudaStreamSynchronize(stream);
	if (err1 != cudaSuccess) {
		return CUDASTREAM_SYNC_FAILED;
	}
	return RUN_DETECTION_SUCCESS;
}

//image resized to setting size
cv::Mat YoloV4Detection::preprocess_img(cv::Mat& img) {
	int w, h, x, y;
	float r_w = INPUT_W / (img.cols*1.0);
	float r_h = INPUT_H / (img.rows*1.0);
	if (r_h > r_w) {
		w = INPUT_W;
		h = r_w * img.rows;
		x = 0;
		y = (INPUT_H - h) / 2;
	}
	else {
		w = r_h * img.cols;
		h = INPUT_H;
		x = (INPUT_W - w) / 2;
		y = 0;
	}
	cv::Mat re(h, w, CV_8UC3);
	cv::resize(img, re, re.size(), 0, 0, cv::INTER_CUBIC);
	cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(128, 128, 128));
	re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
	return out;
}

void YoloV4Detection::nms(std::vector<YoloParam::Detection>& res, float *output, float nms_thresh) {
	std::map<float, std::vector<YoloParam::Detection>> m;
	for (int i = 0; i < output[0] && i < YoloParam::MAX_OUTPUT_BBOX_COUNT; i++) {
		if (output[1 + DETECTION_SIZE * i + 4] <= BBOX_CONF_THRESH) continue;
		YoloParam::Detection det;
		memcpy(&det, &output[1 + DETECTION_SIZE * i], DETECTION_SIZE * sizeof(float));
		if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<YoloParam::Detection>());
		m[det.class_id].push_back(det);
	}
	for (auto it = m.begin(); it != m.end(); it++) {
		//std::cout << it->second[0].class_id << " --- " << std::endl;
		auto& dets = it->second;
		//std::sort(dets.begin(), dets.end(), bind(cmp,ref(std::placeholders::_1), ref(std::placeholders::_2)));  //???

		std::sort(dets.begin(), dets.end(), [](YoloParam::Detection& a, YoloParam::Detection& b) {return a.det_confidence > b.det_confidence; });

		for (size_t m = 0; m < dets.size(); ++m) {
			auto& item = dets[m];
			res.push_back(item);
			for (size_t n = m + 1; n < dets.size(); ++n) {
				if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
					dets.erase(dets.begin() + n);
					--n;
				}
			}
		}
	}
}

float YoloV4Detection::iou(float lbox[4], float rbox[4]) {
	float interBox[] = {
		std::max(lbox[0] - lbox[2] / 2.f , rbox[0] - rbox[2] / 2.f), //left
		std::min(lbox[0] + lbox[2] / 2.f , rbox[0] + rbox[2] / 2.f), //right
		std::max(lbox[1] - lbox[3] / 2.f , rbox[1] - rbox[3] / 2.f), //top
		std::min(lbox[1] + lbox[3] / 2.f , rbox[1] + rbox[3] / 2.f), //bottom
	};

	if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
		return 0.0f;

	float interBoxS = (interBox[1] - interBox[0])*(interBox[3] - interBox[2]);
	return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}

bool YoloV4Detection::cmp(YoloParam::Detection& a, YoloParam::Detection& b) {
	return a.det_confidence > b.det_confidence;
}

cv::Rect YoloV4Detection::get_rect(const int width,const int height, float bbox[4]) {
	int l, r, t, b;
	float r_w = INPUT_W / (width * 1.0);
	float r_h = INPUT_H / (height * 1.0);
	if (r_h > r_w) {
		l = bbox[0] - bbox[2] / 2.f;
		r = bbox[0] + bbox[2] / 2.f;
		t = bbox[1] - bbox[3] / 2.f - (INPUT_H - r_w * height) / 2;
		b = bbox[1] + bbox[3] / 2.f - (INPUT_H - r_w * height) / 2;
		l = l / r_w;
		r = r / r_w;
		t = t / r_w;
		b = b / r_w;
	}
	else {
		l = bbox[0] - bbox[2] / 2.f - (INPUT_W - r_h * width) / 2;
		r = bbox[0] + bbox[2] / 2.f - (INPUT_W - r_h * width) / 2;
		t = bbox[1] - bbox[3] / 2.f;
		b = bbox[1] + bbox[3] / 2.f;
		l = l / r_h;
		r = r / r_h;
		t = t / r_h;
		b = b / r_h;
	}
	return cv::Rect(l, t, r - l, b - t);
}