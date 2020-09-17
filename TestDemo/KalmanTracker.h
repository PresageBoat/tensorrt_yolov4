#ifndef KALMAN_H
#define KALMAN_H

#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <cmath>

using namespace std;
using namespace cv;

#define StateType Rect_<float>


// This class represents the internel state of individual tracked objects observed as bounding box.
class KalmanTracker
{
public:
	//KalmanTracker()
	//{
	//	init_kf(StateType());
	//	m_time_since_update = 0;
	//	m_hits = 0;
	//	m_hit_streak = 0;
	//	m_age = 0;
	//	m_id = kf_count;
	//	cls_idx = -1;
	//}

	KalmanTracker(StateType initRect,const int cls_id)
	{
		init_kf(initRect);
		m_time_since_update = 0;
		m_hits = 0;
		m_hit_streak = 0;
		m_age = 0;
		m_id = kf_count;
		kf_count++;
		cls_idx = cls_id;
	}

	~KalmanTracker()
	{
		m_history.clear();
	}

	StateType predict();
	void update(StateType stateMat);

	StateType get_state();
	StateType get_rect_xysr(float cx, float cy, float s, float r);

	static int kf_count;

	int m_time_since_update;
	int m_hits;
	int m_hit_streak;
	int m_age;
	int m_id;
	int cls_idx;

private:
	void init_kf(StateType stateMat);

	KalmanFilter kf;
	Mat measurement;

	std::vector<StateType> m_history;
};




#endif