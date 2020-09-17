#include "sorttracking.h"

SortTracking::SortTracking() {
	KalmanTracker::kf_count = 0;
}

SortTracking::~SortTracking() {

}

int SortTracking::RunSortTracker(const vector<TrackingBox> &detectresult, vector<TrackingBox>& trackresult) {

	//vector<int> cls_id_vec;

	//当跟踪器为空时，用检测结果进行初始化
	if (trackers.size() == 0)
	{
		for (unsigned int i = 0; i < detectresult.size(); i++)
		{
			KalmanTracker trk = KalmanTracker(detectresult[i].box,detectresult[i].cls_id);
			trackers.emplace_back(trk);
		}
	}

	//清除预测列表中的信息
	vector<Rect_<float>>().swap(predictedBoxes);
	predictedBoxes.clear();

	for (auto it = trackers.begin(); it != trackers.end();)
	{
		Rect_<float> pBox = (*it).predict();
		if (pBox.x >= 0 && pBox.y >= 0)
		{
			predictedBoxes.push_back(pBox);
			it++;
		}
		else
		{
			it = trackers.erase(it);
		}
	}

	//associate detections to tracked object (both represented as bounding boxes)
	trkNum = predictedBoxes.size();
	detNum = detectresult.size();

	iouMatrix.clear();
	iouMatrix.resize(trkNum, vector<double>(detNum, 0));

	for (unsigned int i = 0; i < trkNum; i++) // compute iou matrix as a distance matrix
	{
		for (unsigned int j = 0; j < detNum; j++)
		{
			// use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
			iouMatrix[i][j] = 1 - GetIOU(predictedBoxes[i], detectresult[j].box);
		}
	}

	// solve the assignment problem using hungarian algorithm.
// the resulting assignment is [track(prediction) : detection], with len=preNum

	vector<int>().swap(assignment);
	assignment.clear();

	HungAlgo.Solve(iouMatrix, assignment);

	// find matches, unmatched_detections and unmatched_predictions

	unmatchedTrajectories.clear();
	unmatchedDetections.clear();
	allItems.clear();
	matchedItems.clear();

	if (detNum > trkNum) //	there are unmatched detections
	{
		for (unsigned int n = 0; n < detNum; n++)
			allItems.insert(n);

		for (unsigned int i = 0; i < trkNum; ++i)
			matchedItems.insert(assignment[i]);

		set_difference(allItems.begin(), allItems.end(),
			matchedItems.begin(), matchedItems.end(),
			insert_iterator<set<int>>(unmatchedDetections, unmatchedDetections.begin()));
	}
	else if (detNum < trkNum) // there are unmatched trajectory/predictions
	{
		for (unsigned int i = 0; i < trkNum; ++i)
			if (assignment[i] == -1) // unassigned label will be set as -1 in the assignment algorithm
				unmatchedTrajectories.insert(i);
	}
	else
		;
	// filter out matched with low IOU
	vector<cv::Point>().swap(matchedPairs);
	matchedPairs.clear();
	for (unsigned int i = 0; i < trkNum; ++i)
	{
		if (assignment[i] == -1) // pass over invalid values
			continue;
		if (1 - iouMatrix[i][assignment[i]] < iouThreshold)
		{
			unmatchedTrajectories.insert(i);
			unmatchedDetections.insert(assignment[i]);
		}
		else
			matchedPairs.push_back(cv::Point(i, assignment[i]));
	}

//updating trackers
// update matched trackers with assigned detections.
// each prediction is corresponding to a tracker
	int detIdx, trkIdx;
	for (unsigned int i = 0; i < matchedPairs.size(); i++)
	{
		trkIdx = matchedPairs[i].x;
		detIdx = matchedPairs[i].y;
		trackers[trkIdx].update(detectresult[detIdx].box);
	}

	// create and initialise new trackers for unmatched detections
	for (auto umd : unmatchedDetections)
	{
		KalmanTracker tracker = KalmanTracker(detectresult[umd].box,detectresult[umd].cls_id);
		trackers.push_back(tracker);
	}

	// get trackers' output
	for (auto it = trackers.begin(); it != trackers.end();)
	{
		if (((*it).m_time_since_update < 1) &&
			((*it).m_hit_streak >= min_hits || frame_count <= min_hits))
		{
			TrackingBox res;
			res.box = (*it).get_state();
			res.track_id = ((*it).m_id + 1)%MAXTRACKINGID;
			res.cls_id = (*it).cls_idx;
			trackresult.push_back(res);
			it++;
		}
		else
			it++;

		// remove dead tracklet
		if (it != trackers.end() && (*it).m_time_since_update > max_age)
			it = trackers.erase(it);
	}

	return 0;
}


double SortTracking::GetIOU(Rect_<float> bb_test, Rect_<float> bb_gt)
{
	float in = (bb_test & bb_gt).area();
	float un = bb_test.area() + bb_gt.area() - in;

	if (un < DBL_EPSILON)
		return 0;

	return (double)(in / un);
}


