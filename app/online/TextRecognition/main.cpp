#include <Python.h>  // NOLINT(build/include_alpha)

// Produce deprecation warnings (needs to come before arrayobject.h inclusion).
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#define TXT_DECODE_VERSION "0.0.1"

#include <boost/make_shared.hpp>
#include <boost/python.hpp>
#include <boost/python/detail/defaults_gen.hpp>
#include <boost/python/raw_function.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <numpy/arrayobject.h>

// these need to be included after boost on OS X
#include <string>  // NOLINT(build/include_order)
#include <vector>  // NOLINT(build/include_order)
#include <fstream>  // NOLINT

#include "text_detection.hpp"
#include "text_recognition.hpp"

using namespace std;

class TextDetection {

public:

    
    
    void ChanneliseFilters(cv::Mat image) {
      cv::Mat channels = txtdecode::createChannels(image);
      cv::Ptr<cv::text::ERFilter> filterStage1 = txtdecode::obtainFilterStage1(NM1_CLASSIFIER);
      cv::Ptr<cv::text::ERFilter> filterStage2 = txtdecode::obtainFilterStage1(NM2_CLASSIFIER);
    }

    void RunFilters(cv::Mat image, ) {
      reinterpret_cast<vector<cv::text::ERFilter::Callback>&>(cb_vector);
      vector<cv::Rect> group_rects;
      txtdecode::runFilter(cb_vector, image, channels);
    }

}

BOOST_PYTHON_MODULE(txtdecode) {
  // below, we prepend an underscore to methods that will be replaced
  // in Python

  bp::scope().attr("__version__") = AS_STRING(TXT_DECODE_VERSION);

  bp::def("channelise_filters", &ChanneliseFilters);

}