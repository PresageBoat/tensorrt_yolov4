#ifndef _CALIBRATOR_H_
#define _CALIBRATOR_H_


#include <NvInfer.h>
#include <string>
#include <vector>
#include "boost/filesystem.hpp"
#include "boost/system/error_code.hpp"
#include "opencv2/opencv.hpp"
#include "ds_image.h"


#define NV_CUDA_CHECK(status)                                                                      \
    {                                                                                              \
        if (status != 0)                                                                           \
        {                                                                                          \
            std::cout << "Cuda failure: " << cudaGetErrorString(status) << " in file " << __FILE__ \
                      << " at line " << __LINE__ << std::endl;                                     \
            abort();                                                                               \
        }                                                                                          \
    }

class Int8EntropyCalibrator:public nvinfer1::IInt8EntropyCalibrator
{
public:
	Int8EntropyCalibrator(const uint32_t& batchSize, const std::string& calibImagesDir,
		 const std::string& calibTableFilePath,
		 const uint32_t& inputH, const uint32_t& inputW,
		const uint32_t& inputC, const std::string& inputBlobName);
	virtual ~Int8EntropyCalibrator();

	int getBatchSize() const override { return batchsize; }
	bool getBatch(void* bindings[], const char* names[], int nbBindings) override;
	const void* readCalibrationCache(size_t& length) override;
	void writeCalibrationCache(const void* cache, size_t length) override;


private:
	bool IsFileExistent(const boost::filesystem::path& path);
	bool GetDirFiles(std::string dir, std::vector<std::string> &filepath);
	cv::Mat blobFromDsImages(const std::vector<DsImage>& inputImages, const int& inputH, const int& inputW);
private:
	const uint32_t batchsize;
	const uint32_t max_batchsize;//最大支持的batchsize
	const uint32_t input_h;//input image height
	const uint32_t input_w;//input image width
	const uint32_t input_c;//input image channels
	const uint64_t input_size;//input image size=input_w*input_h
	const uint64_t input_count;//input image total size count=input_w*input_h*input_c*batchsize
	const std::string input_blobname;//filename to save the generated calibration table

	const std::string calibtable_filepath;//the path to save calibration table
	std::vector<char> calibrationcache;
	bool read_calibcache{ true };

	//image list
	std::vector<std::string> imagepath_vec;
	uint32_t image_index;

	void* device_buff{ nullptr };
	void* host_buff{ nullptr };

};




#endif  //_CALIBRATOR_H_


