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

namespace bp = boost::python;
namespace np = boost::python::numpy;

typedef cv::Mat3f Ttype;

#include  <vector>
#include  <iostream>
#include  <iomanip>

#include <sstream>
#include <string>

#define NM1_CLASSIFIER "trained_classifierNM1.xml"
#define NM2_CLASSIFIER "trained_classifierNM2.xml"

using namespace std;

namespace bp = boost::python;
namespace np = boost::python::numpy;

#define NM1_CLASSIFIER "trained_classifierNM1.xml"
#define NM2_CLASSIFIER "trained_classifierNM2.xml"

template <class Ttype>
class TextDetection {

private:

  Ttype image;
  cv::Ptr<cv::text::ERFilter> filterStage1;
  cv::Ptr<cv::text::ERFilter> filterStage2;
  cv::Ptr<cv::text::ERFilter::Callback> cb1;
  cv::Ptr<cv::text::ERFilter::Callback> cb2;
  vector<Ttype> channels;

public:

  TextDetection(Ttype img) {
    image = img;
  }
  virtual ~TextDetection();

  cv::Ptr<cv::text::ERFilter> getFilterStage1() {
    return filterStage1;
  }

  cv::Ptr<cv::text::ERFilter> getFilterStage2() {
    return filterStage2;
  }

  vector<Ttype> getChannels() {
    return channels;
  }

  static Ttype boostPythonObject2Mat(np::ndarray nd, int rows, int cols) {
    Ttype image(rows, cols, CV_32FC3);

    char const * data = boost::python::extract<char const *>(boost::python::str(nd));

    for(int i = 0; i < rows; i++) {
      for(int j = 0; j < cols; j++) {
        image(i, j, 0) = (float) data[i*cols + j*3];
        image(i, j, 1) = (float) data[i*cols + j*3 + 1];
        image(i, j, 2) = (float) data[i*cols + j*3 + 2];
      }
    }

    return image;
  }

  vector<Ttype> createChannels(Ttype src) {
    vector<Ttype> channels;
    cv::text::computeNMChannels(src, channels, cv::text::ERFILTER_NM_IHSGrad);
    // preprocess channels to include black and the degree of hue factor
    for(int i = 0; i < channels.size()-1; i++) {
        channels.push_back(255-channels[i]);
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
    vector<Ttype> channels = createChannels(image);
    filterStage1 = obtainFilterStage1(NM1_CLASSIFIER);
    filterStage2 = obtainFilterStage2(NM2_CLASSIFIER);
  }

  int runFilter(vector<cv::Ptr<cv::text::ERFilter>>cb_vector, cv::Mat3f src, vector<cv::Mat3f> channels, 
  vector<vector<cv::text::ERStat>> &regions, vector<vector<cv::Vec2i> > &groups, vector<cv::Rect> groups_rects) {
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

  void RunFilters(vector<Ttype> channels, vector<vector<cv::Vec2i>> groups, vector<cv::Rect> groups_rects, 
vector<cv::Ptr<cv::text::ERFilter>> cb_vector) {

    vector<vector<cv::text::ERStat>> regions(channels.size());
    runFilter(cb_vector, image, channels, regions, groups, groups_rects);

    regions.clear();
    if (!groups_rects.empty()) {
        groups_rects.clear();
    }

}

};

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


  // TextDetection<Ttype>* TextDetection_Init(boost::python::object src) {
  
  //   np::ndarray nd = np::array(src);
  //   const int* rows = (int*) nd.shape(0);
  //   const int* cols = (int*) nd.shape(1);
    
  //   cv::Mat3f image = TextDetection<Ttype>::boostPythonObject2Mat(nd, *rows, *cols);
    
  //   // Initialize text detection
  //   TextDetection<Ttype>* text(new TextDetection<Ttype>(image));

  //   text->ChanneliseFilters();

  //   return text;

  // }

  // boost::python::tuple TextRecognition_HMMDecode(TextDetection<Ttype>* txt, cv::Mat3f image) {

  //   string output_text;
  //   vector<cv::Rect> boxes;
  //   vector<string> words;
  //   vector<float> confidences;

  //   obtainHMMDecoder(image, OCRHMM_KNN_MODEL, output_text, &boxes, &words, &confidences, OCRHMM_TRANSITIONS_TABLE);

  //   return boost::python::make_tuple(output_text, confidences, words, boxes);
  // }

  // boost::python::tuple TextDetection_RunFilters(TextDetection<Ttype>* txt, cv::Mat3f channels) {

  //   vector<vector<cv::Vec2i>> groups;
  //   vector<cv::Rect> groups_rects;

  //   boost::python::iterator<vector<bp::list>> get_vec_iter;

  //   vector<cv::Ptr<cv::text::ERFilter>> cb_vector { txt->getFilterStage1(), txt->getFilterStage2() };
  //   txt->RunFilters(channels, groups, groups_rects, cb_vector);

  //   bp::object get_iterator = boost::python::iterator<vector<int>>();

  //   bp::object _obj_iter1;
  //   bp::object _obj_iter2;
    
  //   if (groups.size()) {
  //     vector<bp::list> _l1;
  //     vector<bp::list> _l2;
  //     for (auto item : groups) {
  //       vector<bp::list> l1;
  //       vector<bp::list> l2;
  //       for (auto vec : item) {
  //         bp::object iter1 = get_iterator(vec[0]);
  //         bp::object iter2 = get_iterator(vec[1]);
  //         bp::list list1(iter1);
  //         bp::list list2(iter2);
  //         l1.push_back(list1);
  //         l2.push_back(list2);
  //       }
  //       bp::object obj_iter1 = get_vec_iter(l1);
  //       bp::object obj_iter2 = get_vec_iter(l2);
  //       bp::list obj1(obj_iter1);
  //       bp::list obj2(obj_iter2);
  //       _l1.push_back(obj1);
  //       _l2.push_back(obj2);
  //     }
  //     bp::object _obj_iter1 = get_vec_iter(_l1);
  //     bp::object _obj_iter2 = get_vec_iter(_l2);
  //   }

  //   return boost::python::make_tuple(_obj_iter1, _obj_iter2);

  // }

  char const* greet()
  {
    return "hello, world";
  }

class World
{
  public:
    World(int x) {}
    World() {}
    void set(std::string msg) { mMsg = msg; }
    void many(boost::python::list msgs) {
        long l = len(msgs);
        std::stringstream ss;
        for (long i = 0; i<l; ++i) {
            if (i>0) ss << ", ";
            std::string s = boost::python::extract<std::string>(msgs[i]);
            ss << s;
        }
        mMsg = ss.str();
    }
    std::string greet() { return mMsg; }
  private:
    std::string mMsg;
};

  // Py_Initialize();

  // PyObject* PyInit_libmain(void* a) {
  //   printf("abcd");
  // };

  BOOST_PYTHON_MODULE(libmain) {

    Py_Initialize();
    // np::initialize();

    // using namespace boost::python;

    // bp::def("greet", greet)
    bp::class_<World>("World")
        .def(bp::init<int>())
        .def("greet", &World::greet)
        .def("set", &World::set)
        .def("many", &World::many)
    ;
    // bp::class_<vector<Ttype> >("TtypeVec")
    //   .def(bp::vector_indexing_suite<vector<Ttype> >());

    // bp::scope().attr("__version__") = AS_STRING(TXT_DECODE_VERSION);

    // bp::class_<TextDetection<Ttype>, TextDetection<Ttype>, boost::noncopyable>("TextDetection", bp::no_init);
    //   Constructor
    //   .def("__init__", bp::make_constructor(&TextDetection_Init));

      // .def("RunFilters", &TextDetection_RunFilters)

      // .def("ObtainHMMDecoder", &TextRecognition_HMMDecode);

    // Py_Finalize();

    import_array1();

  }

// static PyObject *
// demo_add(PyObject *self, PyObject *args)
// {
//     PyObject *ret = NULL;
//     double x, y, result;

//     if (!PyArg_ParseTuple(args, "dd", &x, &y)) {
//         goto out;
//     }
//     result = x + y;
//     ret = Py_BuildValue("d", result);
// out:
//     return ret;
// }

// static PyMethodDef demo_methods[] = {
//     {"add", (PyCFunction) demo_add, METH_VARARGS,
//          "Print a lovely skit to standard output."},
//     {NULL, NULL, 0, NULL}   /* sentinel */
// };

// static struct PyModuleDef demomodule = {
//     PyModuleDef_HEAD_INIT,
//     "demo",
//     NULL,
//     -1,
//     demo_methods
// };

// PyInit_demo(void)
// {
//     return PyModule_Create(&demomodule);
// }
