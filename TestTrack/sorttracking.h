#ifndef SORT_TRACKING_H
#define SORT_TRACKING_H
#include "KalmanTracker.h"
#include "Hungarian.h"
#include <vector>
#include <set>

typedef struct TrackingBox
{
	int track_id;
	Rect_<int> box;//����λ����Ϣ
	TrackingBox() {
		track_id = -1;
		box = { 0,0,0,0 };
	}
}TrackingBox;


class SortTracking
{
public:
	SortTracking();
	~SortTracking();

	int RunSortTracker(const vector<TrackingBox>detectresult,vector<TrackingBox>& trackresult);

private:
	double GetIOU(Rect_<float> bb_test, Rect_<float> bb_gt);

	vector<KalmanTracker> trackers;//������

	vector<Rect_<float>> predictedBoxes;//Ԥ���λ����Ϣ
	vector<vector<double>> iouMatrix;
	vector<int> assignment;
	std::set<int> unmatchedDetections;
	set<int> unmatchedTrajectories;
	set<int> allItems;
	set<int> matchedItems;
	vector<cv::Point> matchedPairs;//��������Ŀ��� һ һ �ɶԴ���
	HungarianAlgorithm HungAlgo;

	int frame_count = 0;
	int max_age = 1;
	int min_hits = 3;
	double iouThreshold = 0.3;

	unsigned int trkNum = 0;//���ٵ�Ŀ����
	unsigned int detNum = 0;//����Ŀ����


};


#endif // SORT_TRACKING_H

