#ifndef _YOLO_PARAM_H
#define _YOLO_PARAM_H

/**
 * yolo layer basic params
 */

namespace YoloParam
{
	static constexpr int CHECK_COUNT = 3;//check input image channel
	static constexpr float IGNORE_THRESH = 0.1f;//
	static constexpr int MAX_OUTPUT_BBOX_COUNT = 1000;//the max output obj count
	static constexpr int CLASS_NUM = 3;//number of categories
	static constexpr int INPUT_H = 416;//input height
	static constexpr int INPUT_W = 416;//input width

	struct YoloKernel
	{
		int width;
		int height;
		float anchors[CHECK_COUNT * 2];
	};
	//yolo anchors mask=0,1,2
	static constexpr YoloKernel yolo1 = {
		INPUT_W / 8,
		INPUT_H / 8,
		{12,16, 19,36, 40,28}
	};
	//yolo anchors mask=3,4,5
	static constexpr YoloKernel yolo2 = {
		INPUT_W / 16,
		INPUT_H / 16,
		{36,75, 76,55, 72,146}
	};
	//yolo anchors mask=6,7,8
	static constexpr YoloKernel yolo3 = {
		INPUT_W / 32,
		INPUT_H / 32,
		{142,110, 192,243, 459,401}
	};

	static constexpr int LOCATIONS = 4;
	struct alignas(float) Detection {
		float bbox[LOCATIONS];//sort :x y w h --(x,y) - top-left corner, (w, h) - width & height of bounded box
		float det_confidence;//object detection confidence
		float class_id;// class of object - from range [0, classes-1]
		float class_confidence;//class of object confidence 
	};

}

#endif //_YOLO_PARAM_H
