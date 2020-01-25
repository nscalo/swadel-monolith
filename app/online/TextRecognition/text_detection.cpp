#include "text_detection.hpp"

namespace txtdecode {
    
    vector<cv::Mat> createChannels(cv::Mat src) {
        vector<cv::Mat> channels;
        cv::text::computeNMChannels(src, channels, cv::text::ERFILTER_NM_IHSGrad);
        // preprocess channels to include black and the degree of hue factor
        for(size_t i = 0; i < channels.size()-1;i++) {
            channels.push_back(255-channels[i]);
        }
        return channels;
    }

    cv::Ptr<cv::text::ERFilter> obtainFilterStage1(const cv::String& filename) {

        cv::Ptr<cv::text::ERFilter::Callback> cb = cv::text::loadClassifierNM1(filename);
        return cv::text::createERFilterNM1(cb,16,0.00015f,0.13f,0.2f,true,0.1f);

    }

    cv::Ptr<cv::text::ERFilter> obtainFilterStage2(const cv::String& filename) {

        cv::Ptr<cv::text::ERFilter::Callback> cb = cv::text::loadClassifierNM2(filename);
        return cv::text::createERFilterNM2(cb,0.5);

    }

    int runFilter(vector<cv::Ptr<cv::text::ERFilter>>cb_vector, cv::Mat src, vector<cv::Mat> channels, 
    vector<vector<cv::text::ERStat> > &regions, vector<vector<cv::Vec2i> > &groups, vector<cv::Rect> group_rects) {

        for(int j = 0; j < cb_vector.size(); j++) {
            for(int i = 0; i < channels.size(); i++) {
                cb_vector[j]->run(channels[i], regions[i]);
            }
        }

        cv::text::erGrouping(src, channels, regions, groups, group_rects, cv::text::ERGROUPING_ORIENTATION_ANY);

        // memory clean-up
        for(int j = 0; j < cb_vector.size(); j++) {
            cb_vector[j].release();
        }

        return 0;

    }

}
