#define NM1_CLASSIFIER "trained_classifierNM1.xml"
#define NM2_CLASSIFIER "trained_classifierNM2.xml"

#include "textdetection.h"

  TextDetection::TextDetection(cv::Mat3f img) {
    image = img;
  }

  cv::Ptr<cv::text::ERFilter> TextDetection::getFilterStage1() {
    return filterStage1;
  }

  cv::Ptr<cv::text::ERFilter> TextDetection::getFilterStage2() {
    return filterStage2;
  }

  vector<cv::Mat3f> TextDetection::getChannels() {
    return channels;
  }

  vector<cv::Mat3f> TextDetection::createChannels(cv::Mat3f src) {
    vector<cv::Mat3f> channels;
    cv::text::computeNMChannels(src, channels, cv::text::ERFILTER_NM_IHSGrad);
    // preprocess channels to include black and the degree of hue factor
    for(int i = 0; i < channels.size()-1; i++) {
        channels.push_back(255-channels[i]);
    }
    return channels;
  }

  cv::Ptr<cv::text::ERFilter> TextDetection::obtainFilterStage1(const cv::String& filename, 
        int thresholdDelta, float minArea,
        float maxArea, float minProbability, 
        bool nonMaxSuppression, float minProbabilityDiff) {
          cb1 = cv::text::loadClassifierNM1(filename);
          return cv::text::createERFilterNM1(cb1,thresholdDelta,minArea,maxArea,minProbability,nonMaxSuppression,minProbabilityDiff);
        }

  cv::Ptr<cv::text::ERFilter> TextDetection::obtainFilterStage2(const cv::String& filename, float minProbability) {
    cb2 = cv::text::loadClassifierNM2(filename);
    return cv::text::createERFilterNM2(cb2,minProbability);
  }
  void TextDetection::ChanneliseFilters() {
    vector<cv::Mat3f> channels = createChannels(image);
    filterStage1 = obtainFilterStage1(NM1_CLASSIFIER,16, 0.00015f,0.13f, 0.2f, true, 0.1f);
    filterStage2 = obtainFilterStage2(NM2_CLASSIFIER, 0.5f);
  }

  int TextDetection::runFilter(vector<cv::Ptr<cv::text::ERFilter>>cb_vector, cv::Mat3f src, vector<cv::Mat3f> channels, 
  vector<vector<cv::text::ERStat>> &regions, vector<vector<cv::Vec2i> > &groups, vector<cv::Rect> &groups_rects) {
    for(int j = 0; j < cb_vector.size(); j++) {
        for(int i = 0; i < channels.size(); i++) {
            cb_vector[j]->run(channels[i], regions[i]);
        }
    }

    cv::text::erGrouping(src, channels, regions, groups, groups_rects, cv::text::ERGROUPING_ORIENTATION_ANY);

    // memory clean-up
    for(int j = 0; j < cb_vector.size(); j++) {
        cb_vector[j].release();
    }

    return 0;
  }

  void TextDetection::RunFilters(vector<cv::Mat3f> channels, vector<vector<cv::Vec2i>> &groups, vector<cv::Rect> &groups_rects, 
vector<cv::Ptr<cv::text::ERFilter>> cb_vector) {

        vector<vector<cv::text::ERStat>> regions(channels.size());
        runFilter(cb_vector, image, channels, regions, groups, groups_rects);

        regions.clear();
        if (!groups_rects.empty()) {
            groups_rects.clear();
        }

    }
