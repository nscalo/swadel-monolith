// Produce deprecation warnings (needs to come before arrayobject.h inclusion).
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#define TXT_DECODE_VERSION "0.0.1"

#include "Python.h"  // NOLINT(build/include_alpha)
#include "object.h"
#include "Repository/Python-3.6.1/Include/unicodeobject.h"
#include "pycapsule.h"
#include "capsulethunk.h"
// #include "cxxabi.h"

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python/object.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/detail/defaults_gen.hpp>
#include <boost/python/raw_function.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <numpy/arrayobject.h>

#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/core/cvstd.hpp"
#include "opencv2/core.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/text.hpp"

// these need to be included after boost on OS X
#include <string>  // NOLINT(build/include_order)
#include <vector>  // NOLINT(build/include_order)
#include <fstream>  // NOLINT
#include <iostream>
#include <stdio.h>

// #include "text_detection.h"
// #include "text_recognition.h"

#include <iostream>     // std::cout
#include <functional>

#include "tbb/tbb_config.h"

#include "list_file.h"

#define STRINGIFY(m) #m
#define AS_STRING(m) STRINGIFY(m)

using namespace std;
using namespace cv;

namespace bp = boost::python;
namespace np = boost::python::numpy;

typedef cv::Mat3f Ttype;

#include  <vector>
#include  <iostream>
#include  <iomanip>

#include <sstream>
#include <string>

namespace bp = boost::python;
namespace np = boost::python::numpy;

#define NM1_CLASSIFIER "/home/aswin/Documents/Courses/Udacity/Intel-Edge/Work/EdgeApp/License_Plate_Recognition/SOLID-Project-Framework/app/online/TextRecognition/main/trained_classifierNM1.xml"
#define NM2_CLASSIFIER "/home/aswin/Documents/Courses/Udacity/Intel-Edge/Work/EdgeApp/License_Plate_Recognition/SOLID-Project-Framework/app/online/TextRecognition/main/trained_classifierNM2.xml"
#define ERGROUPING_CLASSIFIER "/home/aswin/Documents/Courses/Udacity/Intel-Edge/Work/EdgeApp/License_Plate_Recognition/SOLID-Project-Framework/app/online/TextRecognition/main/trained_classifier_erGrouping.xml"

int obtainHMMDecoder(cv::Mat image, const cv::String& filename, std::string& output_text, 
    vector<cv::Rect>* boxes, vector<string>* words, vector<float>* confidences, const std::string& transition_filename) {

    cv::Ptr<cv::text::OCRHMMDecoder::ClassifierCallback> hmm = cv::text::loadOCRHMMClassifierNM(filename);

    // character recognition vocabulary
    cv::String voc = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

    cv::Mat transition_probabilities;
    cv::Mat emission_probabilities = cv::Mat::eye((int)voc.size(), (int)voc.size(), CV_64FC1);

    // cv::text::createOCRHMMTransitionsTable(voc, lexicon, transition_probabilities);

    cv::FileStorage fs("./OCRHMM_transitions_table.xml", cv::FileStorage::READ);
    fs["transition_probabilities"] >> transition_probabilities;
    fs.release();

    cv::Ptr<cv::text::OCRHMMDecoder> ocrNM  = cv::text::OCRHMMDecoder::create(hmm, voc, 
    transition_probabilities, emission_probabilities, cv::text::OCR_DECODER_VITERBI);

    ocrNM->run(image, output_text, boxes, words, confidences, cv::text::OCR_LEVEL_TEXTLINE);

    return 0;

}


  // boost::python::tuple TextRecognition_HMMDecode(TextDetection<Ttype>* txt, cv::Mat3f image) {

  //   string output_text;
  //   vector<cv::Rect> boxes;
  //   vector<string> words;
  //   vector<float> confidences;

  //   obtainHMMDecoder(image, OCRHMM_KNN_MODEL, output_text, &boxes, &words, &confidences, OCRHMM_TRANSITIONS_TABLE);

  //   return boost::python::make_tuple(output_text, confidences, words, boxes);
  // }

  char const* greet()
  {
    return "hello, world";
  }

class TextDetection
{
public:
  TextDetection(np::ndarray src) {
    np::ndarray nd = np::array(src);
    int rows = (int) nd.shape(0);
    int cols = (int) nd.shape(1);
    
    boostPythonObject2Mat(nd, rows, cols);

    // unsigned char *p;
    // p = image.ptr(1,1,0);
    // printf("value: %f", (float)*p);
    // p = image.ptr(2,2,1);
    // printf("value: %f", (float)*p);
    // p = image.ptr(3,3,2);
    // printf("value: %f", (float)*p);

    ChanneliseFilters();
  }
  TextDetection() {}
  ~TextDetection() {}

  cv::Ptr<cv::text::ERFilter> getFilterStage1() {
    return filterStage1;
  }

  cv::Ptr<cv::text::ERFilter> getFilterStage2() {
    return filterStage2;
  }

  vector<cv::Mat> getChannels() {
    return channels;
  }

  void boostPythonObject2Mat(np::ndarray nd, int rows, int cols) {
    image = cv::Mat(rows, cols, CV_8UC3);

    char * data = nd.get_data();

    for(int i = 0; i < rows; i++) {
      for(int j = 0; j < cols; j++) {
        image.at<cv::Vec3b>(i,j)[0] = data[i*cols+j+0];
        image.at<cv::Vec3b>(i,j)[1] = data[i*cols+j+1];
        image.at<cv::Vec3b>(i,j)[2] = data[i*cols+j+2];
      }
    }
  }

  vector<cv::Mat> createChannels(cv::Mat src) {
    vector<cv::Mat> channels;
    cv::text::computeNMChannels(src, channels, cv::text::ERFILTER_NM_IHSGrad);
    cv::Mat expr;
    size_t length = channels.size();
    // preprocess channels to include black and the degree of hue factor
    for(size_t i = 0; i < length-1; i++) {
        expr = channels[i].mul(-1);
        cv::Mat full = cv::Mat::ones(channels[i].size(), channels[i].type()) * 255;
        channels.push_back(full + expr);
    }
    return channels;
  }

  virtual cv::Ptr<cv::text::ERFilter> obtainFilterStage1(const cv::String& filename, 
        int thresholdDelta = 16, float minArea = (float)0.00015,
        float maxArea = (float)0.13, float minProbability = (float)0.2, 
        bool nonMaxSuppression = true, float minProbabilityDiff = (float)0.1) {
    cb1 = cv::text::loadClassifierNM1(filename);
    return cv::text::createERFilterNM1(cb1,thresholdDelta,minArea,maxArea,minProbability,nonMaxSuppression,minProbabilityDiff);
  }

  virtual cv::Ptr<cv::text::ERFilter> obtainFilterStage2(const cv::String& filename, float minProbability = (float)0.5) {
    cb2 = cv::text::loadClassifierNM2(filename);
    return cv::text::createERFilterNM2(cb2,minProbability);
  }

  void ChanneliseFilters() {
    channels = createChannels(image);
    filterStage1 = obtainFilterStage1(NM1_CLASSIFIER);
    filterStage2 = obtainFilterStage2(NM2_CLASSIFIER);
  }

  int runFilter(vector<cv::Ptr<cv::text::ERFilter>>cb_vector, cv::Mat src, vector<cv::Mat> channels, 
    vector<vector<cv::text::ERStat>> &regions, vector<vector<cv::Vec2i> > &groups, vector<cv::Rect> groups_rects) {
    for(int j = 0; j < cb_vector.size(); j++) {
        for(int i = 0; i < channels.size(); i++) {
            cb_vector[j]->run(channels[i], regions[i]);
        }
    }

    cv::text::erGrouping(src, channels, regions, groups, groups_rects, cv::text::ERGROUPING_ORIENTATION_HORIZ);

    // memory clean-up
    for(int j = 0; j < cb_vector.size(); j++) {
        cb_vector[j].release();
    }

    return 0;
  }

  void RunFilters(vector<cv::Mat> channels, vector<vector<cv::Vec2i>> groups, vector<cv::Rect> groups_rects, 
vector<cv::Ptr<cv::text::ERFilter>> cb_vector) {

    vector<vector<cv::text::ERStat>> regions(channels.size());
    runFilter(cb_vector, image, channels, regions, groups, groups_rects);

    printf("Rect: %d %d %d %d", groups_rects[0].x, groups_rects[0].y, groups_rects[0].width, groups_rects[0].height);

    regions.clear();
    if (!groups_rects.empty()) {
        groups_rects.clear();
    }

  }

  void Run_Filters()
  {
    vector<vector<cv::Vec2i>> groups;
    vector<cv::Rect> groups_rects;

    vector<cv::Ptr<cv::text::ERFilter>> cb_vector { getFilterStage1(), getFilterStage2() };
    RunFilters(channels, groups, groups_rects, cb_vector);
  }

  private:
    cv::Mat image;
    cv::Ptr<cv::text::ERFilter> filterStage1;
    cv::Ptr<cv::text::ERFilter> filterStage2;
    cv::Ptr<cv::text::ERFilter::Callback> cb1;
    cv::Ptr<cv::text::ERFilter::Callback> cb2;
    vector<cv::Mat> channels;
};

BOOST_PYTHON_MODULE(libmain) {

  Py_Initialize();
  np::initialize();

  using namespace boost::python;

  bp::scope().attr("__version__") = AS_STRING(TXT_DECODE_VERSION);

  bp::class_<TextDetection>("TextDetection")
      .def(bp::init<np::ndarray>())
      .def("Run_Filters", &TextDetection::Run_Filters);

  import_array1();

}
