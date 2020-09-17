#include <iostream>
#include "../Export/include/PersonDetectionSdk.h"
#include "boost/filesystem.hpp"
#include "opencv2/opencv.hpp"
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

float iou(float lbox[4], float rbox[4]) {
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

bool cmp(DetectionResult& a, DetectionResult& b) {
	return a.det_confidence > b.det_confidence;
}

cv::Rect get_rect(cv::Mat& img, float bbox[4]) {
	int l, r, t, b;
	float r_w = INPUT_W / (img.cols * 1.0);
	float r_h = INPUT_H / (img.rows * 1.0);
	if (r_h > r_w) {
		l = bbox[0] - bbox[2] / 2.f;
		r = bbox[0] + bbox[2] / 2.f;
		t = bbox[1] - bbox[3] / 2.f - (INPUT_H - r_w * img.rows) / 2;
		b = bbox[1] + bbox[3] / 2.f - (INPUT_H - r_w * img.rows) / 2;
		l = l / r_w;
		r = r / r_w;
		t = t / r_w;
		b = b / r_w;
	}
	else {
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

void nms(std::vector<DetectionResult>& res, float *output, float nms_thresh = NMS_THRESH) {
	std::map<float, std::vector<DetectionResult>> m;

	for (int i = 0; i < output[0] && i < MAX_OUTPUT_BBOX_COUNT; i++) {
		if (output[1 + DETECTION_SIZE * i + 4] <= BBOX_CONF_THRESH) continue;
		DetectionResult det;
		memcpy(&det, &output[1 + DETECTION_SIZE * i], DETECTION_SIZE * sizeof(float));
		if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<DetectionResult>());
		m[det.class_id].push_back(det);
	}
	for (auto it = m.begin(); it != m.end(); it++) {
		//std::cout << it->second[0].class_id << " --- " << std::endl;
		auto& dets = it->second;
		std::sort(dets.begin(), dets.end(), [](DetectionResult& a, DetectionResult& b) {return a.det_confidence > b.det_confidence; });

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


//typedef struct TrackingBox
//{
//	char license[16];						//车牌字符串
//	unsigned char nCarColor;		//车的颜色
//	int track_id;
//	int loss_count;
//	int class_id;//车辆类型
//	float class_confidence;
//	Rect_<int> box;//车辆位置信息
//	vector<Point> box_center_vec;
//	TrackingBox() {
//		track_id = -1;
//		loss_count = 0;
//		class_id = -1;
//		class_confidence = 0.f;
//	}
//}TrackingBox;


// Computes IOU between two bounding boxes
double GetIOU(Rect_<float> bb_test, Rect_<float> bb_gt)
{
	float in = (bb_test & bb_gt).area();
	float un = bb_test.area() + bb_gt.area() - in;

	if (un < DBL_EPSILON)
		return 0;

	return (double)(in / un);
}


//判断两条线段相交vector<Point> line(2),代表一线段
bool JudgeCrossLine2(const vector<Point> & line1, const vector<Point> &line2)
{
	CV_Assert(line1.size() == line2.size());
	CV_Assert(line1.size() == 2);
	Point point1_11, point1_12, point1_21, point1_22;
	//首先判断line1的两个端点,在line2的两侧
	point1_11 = line2[0] - line1[0];
	point1_12 = line2[1] - line1[0];

	point1_21 = line2[0] - line1[1];
	point1_22 = line2[1] - line1[1];

	//point1_11.cross(point1_12)*point1_21.cross(point1_22)<0;//----------表明在两侧
	//再次判断line2的两个端点，在line1的两侧
	Point point2_11, point2_12, point2_21, point2_22;

	point2_11 = line1[0] - line2[0];
	point2_12 = line1[1] - line2[0];

	point2_21 = line1[0] - line2[1];
	point2_22 = line1[1] - line2[1];

	//point2_11.cross(point2_12)*point2_21.cross(point2_22)<0;
	return (point1_11.cross(point1_12)*point1_21.cross(point1_22) < 0 && point2_11.cross(point2_12)*point2_21.cross(point2_22) < 0);
}






void vedio_test() {
	int  init_flag = Image_Init_Person("D:/oldproject/mbjc/YOLODLL/configs/fp16-b1.engine");
	if (init_flag == INIT_SUCCESS)
	{
		std::cout << "Detection init success!" << std::endl;
	}

	//输入数据：排列方式为NCHW ；C通道的排列顺序为RGB
	static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
	static float prob[BATCH_SIZE * OUTPUT_SIZE];

	double inference_time = 0.f;

	const string vedio_path = "D:/objecttrack/16run/1.avi";
	VideoCapture capture;
	capture.open(vedio_path);
	if (!capture.isOpened())
	{
		cout << "Error load Vedio " << vedio_path << endl;
	}
	cv::Mat img;
	cv::Mat show_img;
	while (1)
	{
		capture >> img;
		img.copyTo(show_img);
		if (img.empty())
			break;
		cv::Mat pr_img = preprocess_img(img);

		for (int i = 0; i < INPUT_H * INPUT_W; i++) {
			data[i] = pr_img.at<cv::Vec3b>(i)[2] / 255.0;
			data[i + INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[1] / 255.0;
			data[i + 2 * INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[0] / 255.0;
		}

		// Run inference
		auto start = std::chrono::system_clock::now();
		Image_Detection_Inference_Person(data, prob, BATCH_SIZE);
		//预测结果后处理
		std::vector<std::vector<DetectionResult>> batch_res(1);
		auto& res = batch_res[0];
		nms(res, &prob[0]);

		auto end = std::chrono::system_clock::now();
		std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

		auto& res3 = batch_res[0];
		for (size_t j = 0; j < res3.size(); j++) {
			cv::Rect r = get_rect(show_img, res[j].bbox);
			cv::rectangle(show_img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
		}
		cv::namedWindow("result", 0);
		imshow("result", show_img);
		waitKey(1);

	}


	capture.release();

	Image_Free_Person();
}

#include "sorttracking.h"

void detect_sorttrack() {

	int  init_flag = Image_Init_Person("D:/oldproject/mbjc/YOLODLL/configs/fp16-b1.engine");
	if (init_flag == INIT_SUCCESS)
	{
		std::cout << "Detection init success!" << std::endl;
	}


	//输入数据：排列方式为NCHW ；C通道的排列顺序为RGB
	static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
	static float prob[BATCH_SIZE * OUTPUT_SIZE];

	double inference_time = 0.f;


	//视频流读取
	const string vedio_path = "D:/objecttrack/16run/img_test_plant.avi";
	//const string vedio_path = "rtsp://admin:hik12345@10.1.80.240/Streaming/Channels/101";
	VideoCapture capture;
	capture.open(vedio_path);
	if (!capture.isOpened())
	{
		cout << "Error load vedio " << vedio_path << endl;
	}
	cv::Mat img;
	cv::Mat show_img;

	SortTracking sorttracker;
	vector<TrackingBox> frameTrackingResult;//输入视频以后输出的跟踪结果

	static int frame_id = 0;

	while (1)
	{
		capture >> img;
		if (img.empty())
			break;
		img.copyTo(show_img);

		cv::Mat pr_img = preprocess_img(img);

		for (int i = 0; i < INPUT_H * INPUT_W; i++) {
			data[i] = pr_img.at<cv::Vec3b>(i)[2] / 255.0;
			data[i + INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[1] / 255.0;
			data[i + 2 * INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[0] / 255.0;
		}

		// Run inference
		auto start = std::chrono::system_clock::now();
		Image_Detection_Inference_Person(data, prob, BATCH_SIZE);
		//预测结果后处理
		std::vector<std::vector<DetectionResult>> batch_res(1);
		auto& res = batch_res[0];
		nms(res, &prob[0]);
		auto end = std::chrono::system_clock::now();
		//std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;


		vector<TrackingBox> detection_track;//检测到的有效目标

		auto& res3 = batch_res[0];
		for (size_t j = 0; j < res3.size(); j++) {
			//CarProperty temp_prop;
			TrackingBox temp_box;
			Rect ret;
			//增加有效类别判断,检测目标中只包含有效的信息-车辆类型
			////car-2//bus-5//truck-7
			//if ((int)res[j].class_id == 2
			//	|| (int)res[j].class_id == 5
			//	|| (int)res[j].class_id == 7)
			{
				ret = get_rect(show_img, res[j].bbox);
				if (ret.width <= img.cols / 3.0 && ret.height <= img.rows / 3.0)
				{
					temp_box.box = ret;
					detection_track.emplace_back(temp_box);
				}
			}
		}

		//清除上一次跟踪结果
		vector<TrackingBox>().swap(frameTrackingResult);
		frameTrackingResult.clear();

		auto track_begin = std::chrono::system_clock::now();
		sorttracker.RunSortTracker(detection_track, frameTrackingResult);
		auto track_end = std::chrono::system_clock::now();
		double time_terminal = std::chrono::duration_cast<std::chrono::microseconds>(track_end - track_begin).count();
		std::cout <<"FPS:"<<1000.0*1000/ time_terminal << std::endl;
		//std::cout << "deal time:" << std::chrono::duration_cast<std::chrono::milliseconds>(track_end - track_begin).count()<<" ms" << std::endl;


		for (int i = 0; i < frameTrackingResult.size(); i++)
		{
			rectangle(show_img, frameTrackingResult[i].box, Scalar(0, 255, 0), 2, 1);
			cv::putText(show_img, to_string(frameTrackingResult[i].track_id), cv::Point(frameTrackingResult[i].box.x, frameTrackingResult[i].box.y - 1), cv::FONT_HERSHEY_COMPLEX, 3, cv::Scalar(0, 0, 255), 2);

		}

		cv::namedWindow("result", 0);
		imshow("result", show_img);
		waitKey(1);
		//char savepath[1024];
		//sprintf(savepath, "E:/新建文件夹/1/%d.png", frame_id);
		//cv::imwrite(savepath, show_img);
		//frame_id++;
	}


	capture.release();

	Image_Free_Person();
}



int main()
{
	detect_sorttrack();

}

