#include <opencv2/text.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

#include  <iostream>
#include  <fstream>

using namespace std;

#define OCRHMM_KNN_MODEL "OCRHMM_knn_model_data.xml.gz"
#define OCRHMM_TRANSITIONS_TABLE "OCRHMM_transitions_table.xml"

namespace txtdecode {
    
    cv::Mat obtainHMMDecoder(const std::string& filename, vector<string>& lexicon, std::string& output_text, 
    vector<cv::Rect> boxes, vector<string> words, vector<float> confidences);
    
}

