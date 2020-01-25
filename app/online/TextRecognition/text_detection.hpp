#include  "opencv2/text.hpp"
#include  "opencv2/highgui.hpp"
#include  "opencv2/imgproc.hpp"

#include  <vector>
#include  <iostream>
#include  <iomanip>

#define NM1_CLASSIFIER "trained_classifierNM1.xml"
#define NM2_CLASSIFIER "trained_classifierNM2.xml"

using namespace std;

namespace txtdecode {

    vector<cv::Mat> createChannels(cv::Mat src);

    cv::Ptr<cv::text::ERFilter> obtainFilterStage1(const cv::String& filename);

    cv::Ptr<cv::text::ERFilter> obtainFilterStage2(const cv::String& filename);

    int runFilter(vector<cv::Ptr<cv::text::ERFilter>>cb_vector, cv::Mat src, vector<cv::Mat> channels, 
    vector<vector<cv::text::ERStat> > &regions, vector<vector<cv::Vec2i> > &groups, vector<cv::Rect> group_rects);

}
