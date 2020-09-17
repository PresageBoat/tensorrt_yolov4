#include <iostream>
#include "../Export/include/PersonDetectionSdk.h"
#include "boost/filesystem.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs.hpp"
#include <chrono>

#include<boost/random.hpp>
#include<boost/tokenizer.hpp>

namespace fs = boost::filesystem;
using namespace std;
using namespace cv;


#define BATCH_SIZE 1
#define INPUT_W 416
#define INPUT_H 416
#define OUTPUT_SIZE 7001
#define MAX_OUTPUT_BBOX_COUNT 1000
#define DETECTION_SIZE 7
#define BBOX_CONF_THRESH 0.5
#define NMS_THRESH 0.4

struct alignas(float) DetectionResult {
	float bbox[4];//sort :x y w h --(x,y) - top-left corner, (w, h) - width & height of bounded box
	float det_confidence;//object detection confidence
	float class_id;// class of object - from range [0, classes-1]
	float class_confidence;//class of object confidence 
};


bool GetDirFiles(string dir, vector<string> &filepath)
{
	fs::path fullpath(dir);
	if (!fs::exists(fullpath))
	{
		return false;
	}
	fs::recursive_directory_iterator end_iter;
	for (fs::recursive_directory_iterator iter(fullpath); iter != end_iter; iter++)
	{
		try {
			if (!fs::is_directory(*iter) && (fs::extension(*iter) == ".jpg" || fs::extension(*iter) == ".JPG" || fs::extension(*iter) == ".jpeg" || fs::extension(*iter) == ".bmp" || fs::extension(*iter) == ".png"))
			{
				string name = iter->path().string();
				filepath.emplace_back(name);
			}
		}
		catch (const std::exception &ex) {
			continue;
		}
	}
	return true;
}

cv::Mat preprocess_img(cv::Mat& img) {
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

// Computes IOU between two bounding boxes
double GetIOU(Rect_<float> bb_test, Rect_<float> bb_gt)
{
	float in = (bb_test & bb_gt).area();
	float un = bb_test.area() + bb_gt.area() - in;

	if (un < DBL_EPSILON)
		return 0;

	return (double)(in / un);
}

void  imagedir_test_2() {
	int  init_flag = Image_Init_Person("D:/AAAA/tensorrtyolov4/Export/model/yolov4-fp16-b1.engine");
	if (init_flag == INIT_SUCCESS)
	{
		std::cout << "Detection init success!" << std::endl;
	}

	vector<string> image_filepath;
	const string images_dir = "D:/oldproject/mbjc/YOLODLL/configs/test";
	const string images_savedir = "D:/oldproject/mbjc/YOLODLL/configs/result-1/";
	GetDirFiles(images_dir, image_filepath);
	//将图像的个数弄成batchsize的整数倍
	image_filepath.resize(static_cast<int>(image_filepath.size() / BATCH_SIZE) * BATCH_SIZE);
	std::random_shuffle(image_filepath.begin(), image_filepath.end(), [](int i) { return rand() % i; });

	//输入数据：排列方式为NCHW ；C通道的排列顺序为RGB
	static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
	static float prob[BATCH_SIZE * OUTPUT_SIZE];

	//
	boost::char_separator<char> sep("\\");
	typedef boost::tokenizer<boost::char_separator<char>> CustomTokenizer;

	vector<pair<string, Mat>> out_result;

	double inference_time = 0.f;
	double total_imagecount = 0.f;
	total_imagecount = image_filepath.size();
	cout << "总的图像个数：" << total_imagecount << endl;

	//原始输入图像的大小
	int org_input_width = 0;
	int org_input_height = 0;
	bool get_wh = false;
	int fcount = 0;
	for (int f = 0; f < image_filepath.size(); f++) {

		if ((f+1) % 100 == 0)
		{
			std::cout << (f + 1) << "images --average inference time : " << inference_time * 1.0 / (f + 1)*1.0 << " ms" << endl;
		}
		fcount++;
		vector<pair<string, Mat>>swap(out_result);//清空释放信息
		out_result.clear();

		if (fcount < BATCH_SIZE && f + 1 != image_filepath.size())
			continue;
		for (int b = 0; b < fcount; b++) {
			cv::Mat img = cv::imread(image_filepath[f - fcount + 1 + b]);
			if (!get_wh)
			{
				org_input_width = img.cols;
				org_input_height = img.rows;
				get_wh = true;
			}
			//imshow("1", img);
			//waitKey(0);
			CustomTokenizer tok(image_filepath[f - fcount + 1 + b], sep);
			vector<string> vecseg_tag;
			for (CustomTokenizer::iterator beg = tok.begin(); beg != tok.end(); ++beg)
				vecseg_tag.emplace_back(*beg);
			out_result.push_back(std::make_pair(vecseg_tag[vecseg_tag.size() - 1], img));

			if (img.empty())
				continue;
			cv::Mat pr_img = preprocess_img(img);

			for (int i = 0; i < INPUT_H * INPUT_W; i++) {
				data[b * 3 * INPUT_H * INPUT_W + i] = pr_img.at<cv::Vec3b>(i)[2] / 255.0;
				data[b * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[1] / 255.0;
				data[b * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[0] / 255.0;
			}
		}


		DetectImagesResult detect_result;
		// Run inference
		auto start = std::chrono::system_clock::now();
		Image_Detection_Inference_Person(data, BATCH_SIZE, org_input_width, org_input_height,&detect_result);
		auto end = std::chrono::system_clock::now();
		inference_time += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

		//预测结果后处理
		std::vector<std::vector<DetectionResult>> batch_res(fcount);
		for (int b = 0; b < fcount; b++) {
			for (size_t j = 0; j < detect_result.image_prop[b].base_obj_size; j++) {
				Rect boundbox_rect;
				boundbox_rect.x = detect_result.image_prop[b].base_obj_prop[j].bbox[0];
				boundbox_rect.y = detect_result.image_prop[b].base_obj_prop[j].bbox[1];
				boundbox_rect.width = detect_result.image_prop[b].base_obj_prop[j].bbox[2];
				boundbox_rect.height = detect_result.image_prop[b].base_obj_prop[j].bbox[3];

				cv::rectangle(out_result[b].second, boundbox_rect, cv::Scalar(0x27, 0xC1, 0x36), 2);
				cv::putText(out_result[b].second, std::to_string(detect_result.image_prop[b].base_obj_prop[j].class_id), cv::Point(boundbox_rect.x, boundbox_rect.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
			}
			cv::imwrite(images_savedir + out_result[b].first, out_result[b].second);
		}

		fcount = 0;
	}

	std::cout << "average image inference time :" << inference_time * 1.0 / image_filepath.size()*1.f << std::endl;
	getchar();

	Image_Free_Person();

}

//void  vedio_test_2() {
//
//	int  init_flag = Image_Init_Person("D:/AAAA/tensorrtyolov4/Export/model/vehicledetection_fp16.engine");
//	if (init_flag == INIT_SUCCESS)
//	{
//		std::cout << "Detection init success!" << std::endl;
//	}
//
//
//	//输入数据：排列方式为NCHW ；C通道的排列顺序为RGB
//	static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
//	static float prob[BATCH_SIZE * OUTPUT_SIZE];
//
//	//原始输入图像的大小
//	int org_input_width = 0;
//	int org_input_height = 0;
//	bool get_wh = false;
//	int fcount = 0;
//	for (int f = 0; f < image_filepath.size(); f++) {
//
//		if ((f + 1) % 100 == 0)
//		{
//			std::cout << (f + 1) << "images --average inference time : " << inference_time * 1.0 / (f + 1)*1.0 << " ms" << endl;
//		}
//		fcount++;
//		vector<pair<string, Mat>>swap(out_result);//清空释放信息
//		out_result.clear();
//
//		if (fcount < BATCH_SIZE && f + 1 != image_filepath.size())
//			continue;
//		for (int b = 0; b < fcount; b++) {
//			cv::Mat img = cv::imread(image_filepath[f - fcount + 1 + b]);
//			if (!get_wh)
//			{
//				org_input_width = img.cols;
//				org_input_height = img.rows;
//				get_wh = true;
//			}
//			//imshow("1", img);
//			//waitKey(0);
//			CustomTokenizer tok(image_filepath[f - fcount + 1 + b], sep);
//			vector<string> vecseg_tag;
//			for (CustomTokenizer::iterator beg = tok.begin(); beg != tok.end(); ++beg)
//				vecseg_tag.emplace_back(*beg);
//			out_result.push_back(std::make_pair(vecseg_tag[vecseg_tag.size() - 1], img));
//
//			if (img.empty())
//				continue;
//			cv::Mat pr_img = preprocess_img(img);
//
//			for (int i = 0; i < INPUT_H * INPUT_W; i++) {
//				data[b * 3 * INPUT_H * INPUT_W + i] = pr_img.at<cv::Vec3b>(i)[2] / 255.0;
//				data[b * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[1] / 255.0;
//				data[b * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[0] / 255.0;
//			}
//		}
//
//
//		DetectImagesResult detect_result;
//		// Run inference
//		auto start = std::chrono::system_clock::now();
//		Image_Detection_Inference_Person(data, BATCH_SIZE, org_input_width, org_input_height, &detect_result);
//		auto end = std::chrono::system_clock::now();
//		inference_time += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//		std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
//
//		//预测结果后处理
//		std::vector<std::vector<DetectionResult>> batch_res(fcount);
//		for (int b = 0; b < fcount; b++) {
//			for (size_t j = 0; j < detect_result.image_prop[b].base_obj_size; j++) {
//				Rect boundbox_rect;
//				boundbox_rect.x = detect_result.image_prop[b].base_obj_prop[j].bbox[0];
//				boundbox_rect.y = detect_result.image_prop[b].base_obj_prop[j].bbox[1];
//				boundbox_rect.width = detect_result.image_prop[b].base_obj_prop[j].bbox[2];
//				boundbox_rect.height = detect_result.image_prop[b].base_obj_prop[j].bbox[3];
//
//				cv::rectangle(out_result[b].second, boundbox_rect, cv::Scalar(0x27, 0xC1, 0x36), 2);
//				cv::putText(out_result[b].second, std::to_string(detect_result.image_prop[b].base_obj_prop[j].class_id), cv::Point(boundbox_rect.x, boundbox_rect.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
//			}
//			cv::imwrite(images_savedir + out_result[b].first, out_result[b].second);
//		}
//
//		fcount = 0;
//	}
//
//	std::cout << "average image inference time :" << inference_time * 1.0 / image_filepath.size()*1.f << std::endl;
//	getchar();
//
//	Image_Free_Person();
//
//}



void test_imdecode() {
	string fname = "D:/oldproject/mbjc/yolo-tensorrt-master/configs/dog.jpg";
	//! 以二进制流方式读取图片到内存
	FILE* pFile = fopen(fname.c_str(), "rb");
	fseek(pFile, 0, SEEK_END);
	long lSize = ftell(pFile);
	rewind(pFile);
	char* pData = new char[lSize];
	fread(pData, sizeof(char), lSize, pFile);
	fclose(pFile);
	//! 解码内存数据，变成cv::Mat数据
	cv::Mat img_decode;
	vector<uchar> data;
	for (int i = 0; i < lSize; ++i) {
		data.push_back(pData[i]);
	}
	img_decode = cv::imdecode(data, CV_LOAD_IMAGE_COLOR);
	imshow("src", img_decode);
	waitKey(0);
	cv::flip(img_decode, img_decode, -1);
	img_decode.channels();
	//! 将cv::Mat数据编码成数据流
	vector<unsigned char> img_encode;
	cv::imencode(".jpg", img_decode, img_encode);
	unsigned char *encode_data = new unsigned char[lSize];
	for (int i = 0; i < lSize; i++) {
		encode_data[i] = img_encode[i];
	}
}

void signalimageparse() {
	Mat src = imread("D:/oldproject/mbjc/yolo-tensorrt-master/configs/dog.jpg");
	int org_height = src.rows;
	int org_width = src.cols;
	int org_channel = src.channels();

	long  img_size = org_channel * org_width*org_height;

	//将mat数据转化成uchar* BGR 按照行列式编码规则
	uchar* ptr_input_img = new uchar[org_height*org_width*org_channel]();
	for (int i = 0; i < org_height; i++) {
		for (int j = 0; j < org_width; j++) {
			for (int k = 0; k < org_channel; k++) {
				ptr_input_img[k*org_width*org_height + i * org_width + j] =
					((uchar*)(src.data))[i*src.step + j * src.channels() + k];
			}
		}
	}

	//将按照BGR方式排列的数据，转换为mat图像
	Mat recreate_src;
	recreate_src = cv::Mat(org_height, org_width, CV_8UC3);
	//recreate_src.data = ptr_input_img;
	for (int i = 0; i < org_height; i++) {
		for (int j = 0; j < org_width; j++) {
			for (int k = 0; k < org_channel; k++) {
				((uchar*)(recreate_src.data))[i*recreate_src.step + j * recreate_src.channels() + k]
					= ptr_input_img[k*org_width*org_height + i * org_width + j];
			}
		}
	}

	imshow("re_src", recreate_src);
	waitKey(0);
}

void batchimageparese() {
	int  bbbbnbbb_size = 2;
	Mat src = imread("D:/oldproject/mbjc/yolo-tensorrt-master/configs/dog.jpg");
	int org_height = src.rows;
	int org_width = src.cols;
	int org_channel = src.channels();

	long  img_size = org_channel * org_width*org_height;


	//将mat数据转化成uchar* BGR 按照行列式编码规则
	uchar* ptr_input_img = new uchar[bbbbnbbb_size*org_height*org_width*org_channel]();
	for (int b = 0; b < bbbbnbbb_size; b++)
	{
		for (int i = 0; i < org_height; i++) {
			for (int j = 0; j < org_width; j++) {
				for (int k = 0; k < org_channel; k++) {
					ptr_input_img[b*(org_height*org_width*org_channel) + k * org_width*org_height + i * org_width + j] =
						((uchar*)(src.data))[i*src.step + j * src.channels() + k];
				}
			}
		}
	}

	//将按照BGR方式排列的数据，转换为mat图像
	Mat recreate_src;
	recreate_src = cv::Mat(org_height, org_width, CV_8UC3);
	//recreate_src.data = ptr_input_img;
	for (int b = 0; b < bbbbnbbb_size; b++)
	{
		for (int i = 0; i < org_height; i++) {
			for (int j = 0; j < org_width; j++) {
				for (int k = 0; k < org_channel; k++) {
					((uchar*)(recreate_src.data))[i*recreate_src.step + j * recreate_src.channels() + k]
						= ptr_input_img[b*(org_height*org_width*org_channel) + k * org_width*org_height + i * org_width + j];
				}
			}
		}
		char savepath[1024];
		sprintf(savepath, "./sb/%d.png", b);
		imwrite(savepath, recreate_src);
		recreate_src.setTo(0);
	}
}

void test() {
	Mat src = imread("D:/oldproject/mbjc/yolo-tensorrt-master/configs/person.png");
	resize(src, src, Size(1920, 1080));
	int org_height = src.rows;
	int org_width = src.cols;
	int org_channel = src.channels();

	long  img_size = org_channel * org_width*org_height;

	//将mat数据转化成uchar* BGR 按照行列式编码规则
	uchar* ptr_input_img = new uchar[org_height*org_width*org_channel]();
	int top_index = 0;
	for (int h = 0; h < org_height; h++) {
		const uchar* psrcdata = src.ptr<uchar>(h);
		int img_index = 0;
		for (int w = 0; w < org_width; w++) {
			for (int c = 0; c < org_channel; c++) {
				top_index = (c*org_height + h)*org_width + w;
				ptr_input_img[top_index] = static_cast<uchar>(psrcdata[img_index++]);
			}
		}
	}

	//auto end = std::chrono::system_clock::now();
	//std::cout << "convert mat data to uchar* :" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

	//将按照BGR方式排列的数据，转换为mat图像
	//recreate_src.data = ptr_input_img;

	auto start0 = std::chrono::system_clock::now();
	Mat recreate_src;
	recreate_src = cv::Mat(org_height, org_width, CV_8UC3);

	int top2_index = 0;
	for (int h = 0; h < org_height; h++) {
		 uchar* psrcdata = recreate_src.ptr<uchar>(h);
		int img_index = 0;
		for (int w = 0; w < org_width; w++) {
			for (int c = 0; c < org_channel; c++) {
				top2_index = (c*org_height + h)*org_width + w;
				psrcdata[img_index++] = ptr_input_img[top2_index];
			}
		}
	}

	auto end0 = std::chrono::system_clock::now();
	std::cout << "convert uchar* data to mat  :" << std::chrono::duration_cast<std::chrono::milliseconds>(end0 - start0).count() << "ms" << std::endl;

	//imshow("re_src", recreate_src);
	//waitKey(0);
	//getchar();
}

int main()
{
	imagedir_test_2();
}

