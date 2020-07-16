#include <iostream>
#include "../Export/include/PersonDetectionSdk.h"
#include "boost/filesystem.hpp"
#include "opencv2/opencv.hpp"
#include <chrono>

#include <boost/random.hpp>
#include <boost/tokenizer.hpp>

namespace fs = boost::filesystem;
using namespace std;
using namespace cv;

#define BATCH_SIZE 128
#define INPUT_W 416
#define INPUT_H 416
#define OUTPUT_SIZE 7001
#define MAX_OUTPUT_BBOX_COUNT 1000
#define DETECTION_SIZE 7
#define BBOX_CONF_THRESH 0.5
#define NMS_THRESH 0.4

//set flag to save image result;if open save image result
#define SHOW_RESULT 1

struct alignas(float) DetectionResult
{
	float bbox[4];			//sort :x y w h --(x,y) - top-left corner, (w, h) - width & height of bounded box
	float det_confidence;	//object detection confidence
	float class_id;			// class of object - from range [0, classes-1]
	float class_confidence; //class of object confidence
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
		try
		{
			if (!fs::is_directory(*iter) && (fs::extension(*iter) == ".jpg" || fs::extension(*iter) == ".JPG" || fs::extension(*iter) == ".jpeg" || fs::extension(*iter) == ".bmp" || fs::extension(*iter) == ".png"))
			{
				string name = iter->path().string();
				filepath.emplace_back(name);
			}
		}
		catch (const std::exception &ex)
		{
			continue;
		}
	}
	return true;
}

cv::Mat preprocess_img(cv::Mat &img)
{
	int w, h, x, y;
	float r_w = INPUT_W / (img.cols * 1.0);
	float r_h = INPUT_H / (img.rows * 1.0);
	if (r_h > r_w)
	{
		w = INPUT_W;
		h = r_w * img.rows;
		x = 0;
		y = (INPUT_H - h) / 2;
	}
	else
	{
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

float iou(float lbox[4], float rbox[4])
{
	float interBox[] = {
		std::max(lbox[0] - lbox[2] / 2.f, rbox[0] - rbox[2] / 2.f), //left
		std::min(lbox[0] + lbox[2] / 2.f, rbox[0] + rbox[2] / 2.f), //right
		std::max(lbox[1] - lbox[3] / 2.f, rbox[1] - rbox[3] / 2.f), //top
		std::min(lbox[1] + lbox[3] / 2.f, rbox[1] + rbox[3] / 2.f), //bottom
	};

	if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
		return 0.0f;

	float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
	return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}

bool cmp(DetectionResult &a, DetectionResult &b)
{
	return a.det_confidence > b.det_confidence;
}

cv::Rect get_rect(cv::Mat &img, float bbox[4])
{
	int l, r, t, b;
	float r_w = INPUT_W / (img.cols * 1.0);
	float r_h = INPUT_H / (img.rows * 1.0);
	if (r_h > r_w)
	{
		l = bbox[0] - bbox[2] / 2.f;
		r = bbox[0] + bbox[2] / 2.f;
		t = bbox[1] - bbox[3] / 2.f - (INPUT_H - r_w * img.rows) / 2;
		b = bbox[1] + bbox[3] / 2.f - (INPUT_H - r_w * img.rows) / 2;
		l = l / r_w;
		r = r / r_w;
		t = t / r_w;
		b = b / r_w;
	}
	else
	{
		l = bbox[0] - bbox[2] / 2.f - (INPUT_W - r_h * img.cols) / 2;
		r = bbox[0] + bbox[2] / 2.f - (INPUT_W - r_h * img.cols) / 2;
		t = bbox[1] - bbox[3] / 2.f;
		b = bbox[1] + bbox[3] / 2.f;
		l = l / r_h;
		r = r / r_h;
		t = t / r_h;
		b = b / r_h;
	}
	return cv::Rect(l, t, r - l, b - t);
}

void nms(std::vector<DetectionResult> &res, float *output, float nms_thresh = NMS_THRESH)
{
	std::map<float, std::vector<DetectionResult>> m;
	for (int i = 0; i < output[0] && i < MAX_OUTPUT_BBOX_COUNT; i++)
	{
		if (output[1 + DETECTION_SIZE * i + 4] <= BBOX_CONF_THRESH)
			continue;
		DetectionResult det;
		memcpy(&det, &output[1 + DETECTION_SIZE * i], DETECTION_SIZE * sizeof(float));
		if (m.count(det.class_id) == 0)
			m.emplace(det.class_id, std::vector<DetectionResult>());
		m[det.class_id].push_back(det);
	}
	for (auto it = m.begin(); it != m.end(); it++)
	{
		//std::cout << it->second[0].class_id << " --- " << std::endl;
		auto &dets = it->second;
		std::sort(dets.begin(), dets.end(), [](DetectionResult &a, DetectionResult &b) { return a.det_confidence > b.det_confidence; });

		for (size_t m = 0; m < dets.size(); ++m)
		{
			auto &item = dets[m];
			res.push_back(item);
			for (size_t n = m + 1; n < dets.size(); ++n)
			{
				if (iou(item.bbox, dets[n].bbox) > nms_thresh)
				{
					dets.erase(dets.begin() + n);
					--n;
				}
			}
		}
	}
}

int main()
{
	int init_flag = Image_Init_Person("../Export/model/yolov4-int8-b128.engine");
	if (init_flag != INIT_SUCCESS)
	{
		std::cout << "Init error : return " << init_flag << std::endl;
	}
	std::cout << Image_Get_Version_Person() << std::endl;
	//
	vector<string> image_filepath;
	const string images_dir = "../../data/images";
	const string images_savedir = "../../data/images-result/";
	GetDirFiles(images_dir, image_filepath);
	//convert image count to N*BATCH_SIZE (N=1,2,3,...n)
	image_filepath.resize(static_cast<int>(image_filepath.size() / BATCH_SIZE) * BATCH_SIZE);
	std::random_shuffle(image_filepath.begin(), image_filepath.end(), [](int i) { return rand() % i; });

	//Data arrangement by NCHW
	//Image channel arrangement by RGB
	static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
	static float prob[BATCH_SIZE * OUTPUT_SIZE];

	boost::char_separator<char> sep("/");
	typedef boost::tokenizer<boost::char_separator<char>> CustomTokenizer;

#ifdef SHOW_RESULT
	vector<pair<string, Mat>> out_result;
#endif // SHOW_RESULT

	double inference_time = 0.f;
	double total_imagecount = 0.f;
	total_imagecount = image_filepath.size();
	cout << "image total count:" << total_imagecount << endl;
	int fcount = 0;
	for (int f = 0; f < image_filepath.size(); f++)
	{
		if (f % 1000 == 0)
		{
			std::cout << (f + 1) << "images --average inference time : " << inference_time * 1.0 / (f + 1) * 1.0 << " ms" << endl;
		}
		fcount++;
#ifdef SHOW_RESULT
		vector<pair<string, Mat>> swap(out_result); //release vector and free occupied space
		out_result.clear();
#endif // SHOW_RESUL

		if (fcount < BATCH_SIZE && f + 1 != image_filepath.size())
			continue;
		for (int b = 0; b < fcount; b++)
		{
			//std::cout<<"image path:"<<image_filepath[f - fcount + 1 + b]<<std::endl;
			cv::Mat img = cv::imread(image_filepath[f - fcount + 1 + b], 1);
//imshow("1", img);
//waitKey(0);
#ifdef SHOW_RESULT
			CustomTokenizer tok(image_filepath[f - fcount + 1 + b], sep);
			vector<string> vecseg_tag;
			for (CustomTokenizer::iterator beg = tok.begin(); beg != tok.end(); ++beg)
				vecseg_tag.emplace_back(*beg);
			out_result.push_back(std::make_pair(vecseg_tag[vecseg_tag.size() - 1], img));
#endif // SHOW_RESUL

			if (img.empty())
				continue;
			cv::Mat pr_img = preprocess_img(img);
			//imwrite("D:/mbjc/YOLODLL/configs/process_img.jpg", pr_img);

			for (int i = 0; i < INPUT_H * INPUT_W; i++)
			{
				data[b * 3 * INPUT_H * INPUT_W + i] = pr_img.at<cv::Vec3b>(i)[2] / 255.0;
				data[b * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[1] / 255.0;
				data[b * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[0] / 255.0;
			}
		}

		// Run inference
		auto start = std::chrono::system_clock::now();
		Image_Detection_Inference_Person(data, prob, BATCH_SIZE);
		auto end = std::chrono::system_clock::now();
		inference_time += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		//std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

#ifdef SHOW_RESULT
		//deal with prob and get information about each obj in each image
		std::vector<std::vector<DetectionResult>> batch_res(fcount);
		for (int b = 0; b < fcount; b++)
		{
			auto &res = batch_res[b];
			nms(res, &prob[b * OUTPUT_SIZE]);
		}

		for (int b = 0; b < fcount; b++)
		{
			auto &res = batch_res[b];
			for (size_t j = 0; j < res.size(); j++)
			{
				cv::Rect r = get_rect(out_result[b].second, res[j].bbox);
				cv::rectangle(out_result[b].second, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
				//cv::putText(out_result[b].second, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
			}
			//std::cout<<images_savedir + out_result[b].first<<std::endl;
			cv::imwrite(images_savedir + out_result[b].first, out_result[b].second);
		}
#endif // SHOW_RESUL

		fcount = 0;
	}

	std::cout << "average image inference time :" << inference_time * 1.0 / image_filepath.size() * 1.f << std::endl;
	getchar();

	Image_Free_Person();
}
