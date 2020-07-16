#include "calibrator.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <iomanip>
#include <cuda_runtime.h>





Int8EntropyCalibrator::Int8EntropyCalibrator(const uint32_t& batchSize, const std::string& calibImagesDir,
	const std::string& calibTableFilePath,
	const uint32_t& inputH, const uint32_t& inputW,
	const uint32_t& inputC, const std::string& inputBlobName):
	batchsize(batchSize),
	input_h(inputH),
	input_w(inputW),
	input_c(inputC),
	input_size(inputH*inputW),
	input_count(batchSize * inputH*inputW*inputC),
	input_blobname(inputBlobName),
	calibtable_filepath(calibTableFilePath),
	image_index(0),
	max_batchsize(64)
{
	if (!IsFileExistent(calibtable_filepath))
	{
		if (!GetDirFiles(calibImagesDir,imagepath_vec))
		{
			std::cout << "image calibrator dir not exit!" << std::endl;
			assert(0);
		}
		//����������ȡ����batchsize��������
		imagepath_vec.resize(static_cast<int>(imagepath_vec.size() / batchsize) * batchSize);
		std::random_shuffle(imagepath_vec.begin(), imagepath_vec.end());
	}

	//����gpu�ڴ�
	NV_CUDA_CHECK(cudaMalloc(&device_buff, input_count * sizeof(float)));
	//����cpu�ڴ�
	host_buff = new float[batchSize * inputH*inputW*inputC]();

}

Int8EntropyCalibrator::~Int8EntropyCalibrator()
{

	//�ͷ������gpu�ڴ�
	NV_CUDA_CHECK(cudaFree(device_buff));
	//�ͷ������cpu�ڴ�
	delete[] host_buff;
}

bool Int8EntropyCalibrator::getBatch(void* bindings[], const char* names[], int nbBindings)
{

	if (batchsize==0)
	{
		std::cout << "Please check Int8 batchsize,and make sure batchsize >=1" << std::endl;
		return false;
	}
	if (image_index+batchsize>= imagepath_vec.size())
	{
		return false;
	}


	// Load next batch
	std::vector<DsImage> dsImages(batchsize);

	for (int idx= image_index;idx< image_index+batchsize;idx++)
	{
		dsImages.at(idx - image_index) = DsImage(imagepath_vec.at(idx), input_h, input_w);
	}
	image_index += batchsize;//
	cv::Mat trtInput = blobFromDsImages(dsImages, input_h, input_w);

	// Load next batch
	NV_CUDA_CHECK(cudaMemcpy(device_buff, trtInput.ptr<float>(0), input_w * input_h*input_c *batchsize * sizeof(float),cudaMemcpyHostToDevice));
	assert(!strcmp(names[0], input_blobname.c_str()));
	bindings[0] = device_buff;
	return true;
}

const void* Int8EntropyCalibrator::readCalibrationCache(size_t& length)
{
	calibrationcache.clear();
	assert(!calibtable_filepath.empty());
	std::ifstream input(calibtable_filepath, std::ios::binary | std::ios::in);
	input >> std::noskipws;
	if (read_calibcache && input.good())
		std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(calibrationcache));
	length = calibrationcache.size();
	return length? &calibrationcache[0]:nullptr;
}

void Int8EntropyCalibrator::writeCalibrationCache(const void* cache, size_t length)
{
	assert(!calibtable_filepath.empty());
	std::ofstream output(calibtable_filepath, std::ios::binary);
	output.write(reinterpret_cast<const char*>(cache), length);
	//output.close();
}

bool Int8EntropyCalibrator::IsFileExistent(const boost::filesystem::path& path) {

	boost::system::error_code error;
	auto file_status = boost::filesystem::status(path, error);
	if (error) {
		return false;
	}

	if (!boost::filesystem::exists(file_status)) {
		return false;
	}

	if (boost::filesystem::is_directory(file_status)) {
		return false;
	}

	return true;
}

bool Int8EntropyCalibrator::GetDirFiles(std::string dir, std::vector<std::string> &filepath)
{
	boost::filesystem::path fullpath(dir);
	if (!boost::filesystem::exists(fullpath))
	{
		return false;
	}
	boost::filesystem::recursive_directory_iterator end_iter;
	for (boost::filesystem::recursive_directory_iterator iter(fullpath); iter != end_iter; iter++)
	{
		try {
			if (!boost::filesystem::is_directory(*iter) && 
				(boost::filesystem::extension(*iter) == ".JPG"|| boost::filesystem::extension(*iter) == ".jpeg" 
					|| boost::filesystem::extension(*iter) == ".png" || boost::filesystem::extension(*iter) == ".bmp"
					|| boost::filesystem::extension(*iter) == ".jpg"))
			{
				std::string name = iter->path().string();
				filepath.emplace_back(name);
			}
		}
		catch (const std::exception &ex) {
			continue;
		}
	}
	return true;
}


cv::Mat Int8EntropyCalibrator::blobFromDsImages(const std::vector<DsImage>& inputImages,	const int& inputH,	const int& inputW)
{
	std::vector<cv::Mat> letterboxStack(inputImages.size());
	for (uint32_t i = 0; i < inputImages.size(); ++i)
	{
		inputImages.at(i).getLetterBoxedImage().copyTo(letterboxStack.at(i));
	}
	return cv::dnn::blobFromImages(letterboxStack, 1.0, cv::Size(inputW, inputH),
		cv::Scalar(0.0, 0.0, 0.0), true);
}
