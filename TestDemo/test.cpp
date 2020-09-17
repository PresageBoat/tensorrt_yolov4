#include <iostream>
#include "../Export/include/PersonDetectionSdk.h"
#include "boost/filesystem.hpp"
#include <chrono>
#include<boost/random.hpp>
#include<boost/tokenizer.hpp>
#include "PlateID.h"
#include "opencv2/opencv.hpp"

#include <opencv2/tracking.hpp>
#include "opencv2/video/tracking.hpp"
#include "sorttracking.h"
using namespace cv;

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



typedef struct ObjInfo
{
	char license[16];						//车牌字符串
	unsigned char nCarColor;		//车的颜色
	int track_id;
	int loss_count;
	int class_id;//车辆类型
	float class_confidence;
	Rect_<int> box;//车辆位置信息
	vector<Point> box_center_vec;
	ObjInfo() {
		track_id = -1;
		loss_count = 0;
		class_id = -1;
		class_confidence = 0.f;
	}
}ObjInfo;


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

int CheckResult(const int new_trackid, vector<ObjInfo> &track_result) {
	for (int i = 0; i < track_result.size(); i++)
	{
		if (track_result[i].track_id == new_trackid)
		{
			return i;
		}
	}
	return -1;
}

//将检测结果转化为跟踪所需数据结构
void ConvertDetection2Track(const DetectImagesResult detect_result, vector<vector<TrackingBox>> &detection_track)
{
	for (int b = 0; b < BATCH_SIZE; b++)
	{
		vector<TrackingBox> vec_temp_track;
		for (size_t j = 0; j < detect_result.image_prop[b].base_obj_size; j++) {
			TrackingBox temp_track;
			Rect boundbox_rect;
			boundbox_rect.x = detect_result.image_prop[b].base_obj_prop[j].bbox[0];
			boundbox_rect.y = detect_result.image_prop[b].base_obj_prop[j].bbox[1];
			boundbox_rect.width = detect_result.image_prop[b].base_obj_prop[j].bbox[2];
			boundbox_rect.height = detect_result.image_prop[b].base_obj_prop[j].bbox[3];
			temp_track.box = boundbox_rect;
			temp_track.cls_id = detect_result.image_prop[b].base_obj_prop[j].class_id;
			vec_temp_track.emplace_back(temp_track);
		}

		detection_track.emplace_back(vec_temp_track);

	}

}

void ConvertTrack2ObjInfo(const vector<TrackingBox> output_track, vector<ObjInfo> &currentTrackingResult) {
	for (int i = 0; i < output_track.size(); i++)
	{
		ObjInfo temp_info;
		temp_info.box = output_track[i].box;
		temp_info.track_id = output_track[i].track_id;
		temp_info.class_id = output_track[i].cls_id;
		currentTrackingResult.emplace_back(temp_info);
	}
}


void vedio_test() {
	int  init_flag = Image_Init_Person("D:/oldproject/mbjc/YOLODLL/configs/fp16-b1.engine");
	if (init_flag == INIT_SUCCESS)
	{
		std::cout << "Detection init success!" << std::endl;
	}

	////加载车牌识别程序
	//int nRet = InitPlateID();
	//if (nRet != 0)
	//{
	//	std::cout << "Plate init success!" << std::endl;
	//}


	//输入数据：排列方式为NCHW ；C通道的排列顺序为RGB
	static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];

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

	//原始输入图像的大小
	int org_input_width = 0;
	int org_input_height = 0;

	while (1)
	{
		capture >> img;
		img.copyTo(show_img);
		if (img.empty())
			break;
		cv::Mat pr_img = preprocess_img(img);
		org_input_height = img.rows;
		org_input_width = img.cols;
		for (int i = 0; i < INPUT_H * INPUT_W; i++) {
			data[i] = pr_img.at<cv::Vec3b>(i)[2] / 255.0;
			data[i + INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[1] / 255.0;
			data[i + 2 * INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[0] / 255.0;
		}

		// Run inference
		DetectImagesResult detect_result;
		auto start = std::chrono::system_clock::now();
		Image_Detection_Inference_Person(data, BATCH_SIZE, org_input_width, org_input_height, &detect_result);

		for (size_t j = 0; j < detect_result.image_prop[0].base_obj_size; j++) {
			Rect boundbox_rect;
			boundbox_rect.x = detect_result.image_prop[0].base_obj_prop[j].bbox[0];
			boundbox_rect.y = detect_result.image_prop[0].base_obj_prop[j].bbox[1];
			boundbox_rect.width = detect_result.image_prop[0].base_obj_prop[j].bbox[2];
			boundbox_rect.height = detect_result.image_prop[0].base_obj_prop[j].bbox[3];
			cv::rectangle(show_img, boundbox_rect, cv::Scalar(0x27, 0xC1, 0x36), 2);
		}

		cv::namedWindow("result", 0);
		imshow("result", show_img);
		waitKey(1);

	}


	capture.release();

	Image_Free_Person();
}


#include <boost/thread.hpp>

//set cross line
//Point ptstart(31, 437);
//Point ptend(1539, 457);
Point ptstart(100, 700);
Point ptend(1800, 700);
vector<Point> triggle_line;


void post_process(const vector<TrackingBox> detection_track, SortTracking &sorttracker,vector<ObjInfo> &tracking_result) {

vector<TrackingBox> output_track;//跟踪器输出的目标
sorttracker.RunSortTracker(detection_track, output_track);
////将跟踪器输出的目标转化为输出结果
vector<ObjInfo> currentTrackingResult;//输出的目标信息
ConvertTrack2ObjInfo(output_track, currentTrackingResult);
//将当前的帧信息写入到输出结果中
if (tracking_result.size() == 0)
{
	tracking_result = currentTrackingResult;
}
else
{
	//检查以前存储的目标是否已经丢失
	//for (int i = 0; i < tracking_result.size(); i++)
	//{
	//	int exit_idx = CheckResult(tracking_result[i].track_id, currentTrackingResult);
	//	if (exit_idx < 0)
	//	{
	//		tracking_result[i].loss_count = tracking_result[i].loss_count + 1;
	//	}
	//}
	for (auto iter = tracking_result.begin(); iter != tracking_result.end(); )
	{
		int exit_idx = CheckResult((*iter).track_id, currentTrackingResult);
		if (exit_idx < 0)
			iter = tracking_result.erase(iter);
		else
			iter++;
	}

	//检查是否存在新的目标
	for (int i = 0; i < currentTrackingResult.size(); i++)
	{
		int exit_idx = CheckResult(currentTrackingResult[i].track_id, tracking_result);
		if (exit_idx >= 0)
		{
			tracking_result[exit_idx].box = currentTrackingResult[i].box;
			tracking_result[exit_idx].loss_count = 0;
			Point centerpoint;
			centerpoint.x = currentTrackingResult[i].box.x + currentTrackingResult[i].box.width / 2;
			centerpoint.y = currentTrackingResult[i].box.y + currentTrackingResult[i].box.height / 2;
			tracking_result[exit_idx].box_center_vec.emplace_back(centerpoint);
		}
		else
		{
			tracking_result.emplace_back(currentTrackingResult[i]);
		}
	}
}
//将连续3帧都没有出现的目标信息丢掉
//for (auto iter = tracking_result.begin(); iter != tracking_result.end(); )
//{
//	if ((*iter).loss_count >= 3)
//		iter = tracking_result.erase(iter);
//	else
//		iter++;
//}
//触发线
//line(show_img, ptstart, ptend, Scalar(255, 0, 0), 10);
//碰线触发
for (int i = 0; i < tracking_result.size(); i++)
{
	if (tracking_result[i].box_center_vec.size() > 3)
	{
		Point p1, p2;
		p1 = tracking_result[i].box_center_vec[tracking_result[i].box_center_vec.size() - 1];
		p2 = tracking_result[i].box_center_vec[tracking_result[i].box_center_vec.size() - 2];
		vector<Point> temp_line;
		temp_line.emplace_back(p1);
		temp_line.emplace_back(p2);
		//触线 -可以处理自己的逻辑单元了
		bool is_cross = JudgeCrossLine2(temp_line, triggle_line);
		//if (is_cross)
		//{
		//	rectangle(show_img, tracking_result[i].box, Scalar(0, 255, 0), 2, 1);
		//	for (int j = 0; j < tracking_result[i].box_center_vec.size(); j++)
		//	{
		//		circle(show_img, tracking_result[i].box_center_vec[j], 1, Scalar(0, 255, 0), -1);
		//	}
		//}
		//else
		//{
		//	rectangle(show_img, tracking_result[i].box, Scalar(255, 0, 0), 2, 1);
		//	for (int j = 0; j < tracking_result[i].box_center_vec.size(); j++)
		//	{
		//		circle(show_img, tracking_result[i].box_center_vec[j], 1, Scalar(255, 123, 255), -1);
		//	}
		//}
	}
}

}

void detect_track_4road() {
	//触发线
	triggle_line.emplace_back(ptstart);
	triggle_line.emplace_back(ptend);


	int  init_flag = Image_Init_Person("D:/AAAA/tensorrtyolov4/Export/model/yolov4-fp16-b4.engine");
	if (init_flag == INIT_SUCCESS)
	{
		std::cout << "Detection init success!" << std::endl;
	}

	//颜色库
	vector<Scalar> color;
	color.emplace_back(Scalar(0, 0, 255));
	color.emplace_back(Scalar(255, 0, 0));
	color.emplace_back(Scalar(0, 255, 0));
	color.emplace_back(Scalar(0, 255, 255));

	//tracking 
	SortTracking sorttracker[BATCH_SIZE];

	//输入数据：排列方式为NCHW ；C通道的排列顺序为RGB
	static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];

	//视频流读取 --虚拟四路视频
	const string vedio_path1 = "D:/objecttrack/16run/road-1.avi";
	const string vedio_path2 = "D:/objecttrack/16run/road-2.avi";
	const string vedio_path3 = "D:/objecttrack/16run/road-3.avi";
	const string vedio_path4 = "D:/objecttrack/16run/road-4.avi";
	//const string vedio_path = "rtsp://admin:hik12345@10.1.80.240/Streaming/Channels/101";
	VideoCapture capture1, capture2, capture3, capture4;
	capture1.open(vedio_path1);
	if (!capture1.isOpened())
	{
		cout << "Error load vedio " << vedio_path1 << endl;
	}
	capture2.open(vedio_path2);
	if (!capture2.isOpened())
	{
		cout << "Error load vedio " << vedio_path2 << endl;
	}
	capture3.open(vedio_path3);
	if (!capture3.isOpened())
	{
		cout << "Error load vedio " << vedio_path3 << endl;
	}
	capture4.open(vedio_path4);
	if (!capture4.isOpened())
	{
		cout << "Error load vedio " << vedio_path4 << endl;
	}



	cv::Mat img1, img2, img3, img4;
	cv::Mat show_img1, show_img2, show_img3, show_img4;


	//原始输入图像的大小
	int org_input_width = 3840;
	int org_input_height = 2160;

	//输入图像的大小
	cv::Mat show_final_result(org_input_height *2, org_input_width *2,CV_8UC3);
	Rect rect1(0, 0, org_input_width, org_input_height);
	Rect rect2(org_input_width, 0, org_input_width, org_input_height);
	Rect rect3(0, org_input_height, org_input_width, org_input_height);
	Rect rect4(org_input_width, org_input_height, org_input_width, org_input_height);

	vector<vector<ObjInfo>> frameTrackingResult(BATCH_SIZE);//输出的目标信息

	while (1)
	{
		capture1 >> img1;
		capture2 >> img2;
		capture3 >> img3;
		capture4 >> img4;

		if (img1.empty() || img2.empty() || img3.empty() || img4.empty())
			break;

		vector<Mat> input_img;//输入数据
		input_img.emplace_back(img1);
		input_img.emplace_back(img2);
		input_img.emplace_back(img3);
		input_img.emplace_back(img4);

		//显示图片
		img1.copyTo(show_img1);
		img2.copyTo(show_img2);
		img3.copyTo(show_img3);
		img4.copyTo(show_img4);

		vector<Mat> show_img;//结果显示数据
		show_img.emplace_back(show_img1);
		show_img.emplace_back(show_img2);
		show_img.emplace_back(show_img3);
		show_img.emplace_back(show_img4);
		//获取输入图像的大小
		org_input_height = img1.rows;
		org_input_width = img1.cols;


		//将输入的多路数据进行处理
		for (int b = 0; b < BATCH_SIZE; b++)
		{
			cv::Mat pr_img = preprocess_img(input_img[b]);
			for (int i = 0; i < INPUT_H * INPUT_W; i++) {
				data[b * 3 * INPUT_H * INPUT_W + i] = pr_img.at<cv::Vec3b>(i)[2] / 255.0;
				data[b * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[1] / 255.0;
				data[b * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[0] / 255.0;
			}
		}

		// Run inference
		DetectImagesResult detect_result;
		auto start = std::chrono::system_clock::now();
		Image_Detection_Inference_Person(data, BATCH_SIZE, org_input_width, org_input_height, &detect_result);
		vector<vector<TrackingBox>> detection_track;//检测到的有效目标
		////将检测结果转化为跟踪需要的结构化数据
		ConvertDetection2Track(detect_result, detection_track);
		//处理多路跟踪和相应的操作
		boost::thread thrd1(boost::bind(&post_process, detection_track[0], boost::ref(sorttracker[0]), boost::ref(frameTrackingResult[0])));
		boost::thread thrd2(boost::bind(&post_process, detection_track[1], boost::ref(sorttracker[1]), boost::ref(frameTrackingResult[1])));
		boost::thread thrd3(boost::bind(&post_process, detection_track[2], boost::ref(sorttracker[2]), boost::ref(frameTrackingResult[2])));
		boost::thread thrd4(boost::bind(&post_process, detection_track[3], boost::ref(sorttracker[3]), boost::ref(frameTrackingResult[3])));
		if (thrd1.joinable())
		{
			thrd1.join();
		}
		if (thrd2.joinable())
		{
			thrd2.join();
		}
		if (thrd3.joinable())
		{
			thrd3.join();
		}
		if (thrd4.joinable())
		{
			thrd4.join();
		}

		auto end = std::chrono::system_clock::now();
		std::cout << "detection && tracking  time :"<<std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

		///
		for (int b = 0; b < BATCH_SIZE; b++) {

			for (int i = 0; i < frameTrackingResult[b].size(); i++)
			{
				if (frameTrackingResult[b].at(i).loss_count==0)
				{
					rectangle(show_img[b], frameTrackingResult[b].at(i).box, color[b], 2, 1);
					cv::putText(show_img[b], to_string(frameTrackingResult[b].at(i).class_id),
						cv::Point(frameTrackingResult[b].at(i).box.x, frameTrackingResult[b].at(i).box.y - 1),
						cv::FONT_HERSHEY_COMPLEX, 3, cv::Scalar(0, 0, 255), 2);

					for (int j = 0; j < frameTrackingResult[b].at(i).box_center_vec.size(); j++)
					{
						circle(show_img[b], frameTrackingResult[b].at(i).box_center_vec[j], 1, color[b], -1);
					}

				}

			}
		}


		show_img[0].copyTo(show_final_result(rect1));
		show_img[1].copyTo(show_final_result(rect2));
		show_img[2].copyTo(show_final_result(rect3));
		show_img[3].copyTo(show_final_result(rect4));

		cv::namedWindow("result", 0);
		imshow("result", show_final_result);
		waitKey(1);

	}


	capture1.release();
	capture2.release();
	capture3.release();
	capture4.release();

	Image_Free_Person();
}

void detect_track_2road() {
	//触发线
	triggle_line.emplace_back(ptstart);
	triggle_line.emplace_back(ptend);


	int  init_flag = Image_Init_Person("D:/AAAA/tensorrtyolov4/Export/model/yolov4-fp16-b2.engine");
	if (init_flag == INIT_SUCCESS)
	{
		std::cout << "Detection init success!" << std::endl;
	}

	//颜色库
	vector<Scalar> color;
	color.emplace_back(Scalar(0, 0, 255));
	color.emplace_back(Scalar(255, 0, 0));

	//tracking 
	SortTracking sorttracker[BATCH_SIZE];

	//输入数据：排列方式为NCHW ；C通道的排列顺序为RGB
	static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];

	//视频流读取 --虚拟四路视频
	const string vedio_path1 = "D:/objecttrack/16run/road-1.avi";
	const string vedio_path2 = "D:/objecttrack/16run/road-2.avi";
	//const string vedio_path = "rtsp://admin:hik12345@10.1.80.240/Streaming/Channels/101";
	VideoCapture capture1, capture2;
	capture1.open(vedio_path1);
	if (!capture1.isOpened())
	{
		cout << "Error load vedio " << vedio_path1 << endl;
	}
	capture2.open(vedio_path2);
	if (!capture2.isOpened())
	{
		cout << "Error load vedio " << vedio_path2 << endl;
	}

	cv::Mat img1, img2;
	cv::Mat show_img1, show_img2;


	//原始输入图像的大小
	int org_input_width = 3840;
	int org_input_height = 2160;

	//输入图像的大小
	cv::Mat show_final_result(org_input_height , org_input_width * 2, CV_8UC3);
	Rect rect1(0, 0, org_input_width, org_input_height);
	Rect rect2(org_input_width, 0, org_input_width, org_input_height);

	vector<vector<ObjInfo>> frameTrackingResult(BATCH_SIZE);//输出的目标信息

	while (1)
	{
		capture1 >> img1;
		capture2 >> img2;

		if (img1.empty() || img2.empty())
			break;

		vector<Mat> input_img;//输入数据
		input_img.emplace_back(img1);
		input_img.emplace_back(img2);

		//显示图片
		img1.copyTo(show_img1);
		img2.copyTo(show_img2);

		vector<Mat> show_img;//结果显示数据
		show_img.emplace_back(show_img1);
		show_img.emplace_back(show_img2);
		//获取输入图像的大小
		org_input_height = img1.rows;
		org_input_width = img1.cols;


		//将输入的多路数据进行处理
		for (int b = 0; b < BATCH_SIZE; b++)
		{
			cv::Mat pr_img = preprocess_img(input_img[b]);
			for (int i = 0; i < INPUT_H * INPUT_W; i++) {
				data[b * 3 * INPUT_H * INPUT_W + i] = pr_img.at<cv::Vec3b>(i)[2] / 255.0;
				data[b * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[1] / 255.0;
				data[b * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[0] / 255.0;
			}
		}

		// Run inference
		DetectImagesResult detect_result;
		auto start = std::chrono::system_clock::now();
		Image_Detection_Inference_Person(data, BATCH_SIZE, org_input_width, org_input_height, &detect_result);
		vector<vector<TrackingBox>> detection_track;//检测到的有效目标
		////将检测结果转化为跟踪需要的结构化数据
		ConvertDetection2Track(detect_result, detection_track);
		//处理多路跟踪和相应的操作
		boost::thread thrd1(boost::bind(&post_process, detection_track[0], boost::ref(sorttracker[0]), boost::ref(frameTrackingResult[0])));
		boost::thread thrd2(boost::bind(&post_process, detection_track[1], boost::ref(sorttracker[1]), boost::ref(frameTrackingResult[1])));
		if (thrd1.joinable())
		{
			thrd1.join();
		}
		if (thrd2.joinable())
		{
			thrd2.join();
		}

		auto end = std::chrono::system_clock::now();
		std::cout << "detection && tracking  time :" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

		///
		for (int b = 0; b < BATCH_SIZE; b++) {

			for (int i = 0; i < frameTrackingResult[b].size(); i++)
			{
				if (frameTrackingResult[b].at(i).loss_count == 0)
				{
					rectangle(show_img[b], frameTrackingResult[b].at(i).box, color[b], 2, 1);
					cv::putText(show_img[b], to_string(frameTrackingResult[b].at(i).class_id),
						cv::Point(frameTrackingResult[b].at(i).box.x, frameTrackingResult[b].at(i).box.y - 1),
						cv::FONT_HERSHEY_COMPLEX, 3, cv::Scalar(0, 0, 255), 2);

					for (int j = 0; j < frameTrackingResult[b].at(i).box_center_vec.size(); j++)
					{
						circle(show_img[b], frameTrackingResult[b].at(i).box_center_vec[j], 1, color[b], -1);
					}

				}

			}
		}


		show_img[0].copyTo(show_final_result(rect1));
		show_img[1].copyTo(show_final_result(rect2));

		cv::namedWindow("result", 0);
		imshow("result", show_final_result);
		waitKey(1);

	}


	capture1.release();
	capture2.release();

	Image_Free_Person();
}


void detect_track_1road() {

	triggle_line.emplace_back(ptstart);
	triggle_line.emplace_back(ptend);


	int  init_flag = Image_Init_Person("D:/AAAA/tensorrtyolov4/Export/model/vehicledetection_fp16.engine");
	if (init_flag == INIT_SUCCESS)
		std::cout << "Detection init success!" << std::endl;

	//颜色库
	vector<Scalar> color;
	color.emplace_back(Scalar(0, 0, 255));


	//tracking 
	SortTracking sorttracker[1];

	//输入数据：排列方式为NCHW ；C通道的排列顺序为RGB
	static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];

	//视频流读取 --虚拟四路视频
	const string vedio_path = "C:/Users/YY219/Desktop/新建文件夹 (2)/2.avi";
	//const string vedio_path = "rtsp://admin:hik12345@10.1.80.240/Streaming/Channels/101";
	VideoCapture capture1;
	capture1.open(vedio_path);
	if (!capture1.isOpened())
	{
		cout << "Error load vedio " << vedio_path << endl;
	}

	cv::Mat img1;
	cv::Mat show_img1;

	//原始输入图像的大小
	int org_input_width = 0;
	int org_input_height = 0;

	vector<vector<ObjInfo>> frameTrackingResult(BATCH_SIZE);//输出的目标信息

	while (1)
	{
		capture1 >> img1;
		if (img1.empty())
			break;
		//
		vector<Mat> input_img;//输入数据
		input_img.emplace_back(img1);
		//显示图片
		img1.copyTo(show_img1);
		vector<Mat> show_img;//结果显示数据
		show_img.emplace_back(show_img1);
		//获取输入图像的大小
		org_input_height = img1.rows;
		org_input_width = img1.cols;

		//将输入的多路数据进行处理
		for (int b = 0; b < BATCH_SIZE; b++)
		{
			cv::Mat pr_img = preprocess_img(input_img[b]);
			for (int i = 0; i < INPUT_H * INPUT_W; i++) {
				data[b * 3 * INPUT_H * INPUT_W + i] = pr_img.at<cv::Vec3b>(i)[2] / 255.0;
				data[b * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[1] / 255.0;
				data[b * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[0] / 255.0;
			}
		}


		// Run inference
		DetectImagesResult detect_result;
		auto start = std::chrono::system_clock::now();
		Image_Detection_Inference_Person(data, BATCH_SIZE, org_input_width, org_input_height, &detect_result);
		auto end = std::chrono::system_clock::now();
		std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

		////将检测结果转化为跟踪需要的结构化数据
		vector<vector<TrackingBox>> detection_track;//检测到的有效目标
		ConvertDetection2Track(detect_result, detection_track);

		//处理多路跟踪和相应的越线操作
		post_process(detection_track[0], sorttracker[0], frameTrackingResult[0]);
		//
		//GetClsID(detect_result, BATCH_SIZE, frameTrackingResult);


		///
		for (int b = 0; b < BATCH_SIZE; b++) {

			for (int i = 0; i < frameTrackingResult[b].size(); i++)
			{
				if (frameTrackingResult[b].at(i).loss_count == 0)
				{
					//if (frameTrackingResult[b].at(i).class_id<0)
					//{
					//	rectangle(show_img[b], frameTrackingResult[b].at(i).box, cv::Scalar(0, 255, 0), 2, 1);
					//	cv::putText(show_img[b], to_string(frameTrackingResult[b].at(i).class_id),
					//		cv::Point(frameTrackingResult[b].at(i).box.x, frameTrackingResult[b].at(i).box.y - 1),
					//		cv::FONT_HERSHEY_COMPLEX, 3, cv::Scalar(0, 255, 0), 2);
					//}
					rectangle(show_img[b], frameTrackingResult[b].at(i).box, color[b], 2, 1);
					cv::putText(show_img[b], to_string(frameTrackingResult[b].at(i).track_id),
						cv::Point(frameTrackingResult[b].at(i).box.x, frameTrackingResult[b].at(i).box.y - 1),
						cv::FONT_HERSHEY_COMPLEX, 3, cv::Scalar(0, 0, 255), 2);
					//cv::putText(show_img[b], to_string(frameTrackingResult[b].at(i).class_id),
					//	cv::Point(frameTrackingResult[b].at(i).box.x, frameTrackingResult[b].at(i).box.y - 1),
					//	cv::FONT_HERSHEY_COMPLEX, 3, cv::Scalar(0, 0, 255), 2);

					for (int j = 0; j < frameTrackingResult[b].at(i).box_center_vec.size(); j++)
					{
						circle(show_img[b], frameTrackingResult[b].at(i).box_center_vec[j], 1, color[b], -1);
					}

				}

			}
		}

		cv::namedWindow("result", 0);
		imshow("result", show_img[0]);
		waitKey(1);

	}


	capture1.release();
	Image_Free_Person();
}


#pragma comment( linker, "/subsystem:/"windows/" /entry:/"mainCRTStartup/"" )

int main()
{
	detect_track_1road();
}

