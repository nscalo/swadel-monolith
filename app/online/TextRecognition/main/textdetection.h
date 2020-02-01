#include <vector>

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/detail/defaults_gen.hpp>
#include <boost/python/raw_function.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/core/cvstd.hpp"
#include "opencv2/core.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/text.hpp"

using namespace std;
namespace bp = boost::python;
namespace np = boost::python::numpy;

#ifdef __cplusplus
  class TextDetection {

private:

  cv::Mat3f image;
  cv::Ptr<cv::text::ERFilter> filterStage1;
  cv::Ptr<cv::text::ERFilter> filterStage2;
  cv::Ptr<cv::text::ERFilter::Callback> cb1;
  cv::Ptr<cv::text::ERFilter::Callback> cb2;
  vector<cv::Mat3f> channels;

public:

  TextDetection(cv::Mat3f img);
  virtual ~TextDetection();

  cv::Ptr<cv::text::ERFilter> getFilterStage1();

  cv::Ptr<cv::text::ERFilter> getFilterStage2();

  vector<cv::Mat3f> getChannels();

  vector<cv::Mat3f> createChannels(cv::Mat3f src);
  
  virtual cv::Ptr<cv::text::ERFilter> obtainFilterStage1(const cv::String& filename, 
        int thresholdDelta, float minArea,
        float maxArea, float minProbability, 
        bool nonMaxSuppression, float minProbabilityDiff);

  virtual cv::Ptr<cv::text::ERFilter> obtainFilterStage2(const cv::String& filename, float minProbability);
  void ChanneliseFilters();

  int runFilter(vector<cv::Ptr<cv::text::ERFilter>>cb_vector, cv::Mat3f src, vector<cv::Mat3f> channels, 
  vector<vector<cv::text::ERStat>> &regions, vector<vector<cv::Vec2i> > &groups, vector<cv::Rect> &groups_rects);

  void RunFilters(vector<cv::Mat3f> channels, vector<vector<cv::Vec2i>> &groups, vector<cv::Rect> &groups_rects, 
vector<cv::Ptr<cv::text::ERFilter>> cb_vector);

};
#else
//   struct TextDetection {
//     cv::Mat3f image;
//     cv::Ptr<cv::text::ERFilter> filterStage1;
//     cv::Ptr<cv::text::ERFilter> filterStage2;
//     cv::Ptr<cv::text::ERFilter::Callback> cb1;
//     cv::Ptr<cv::text::ERFilter::Callback> cb2;
//     vector<cv::Mat3f> channels;

//     cv::Ptr<cv::text::ERFilter> getFilterStage1();

//   cv::Ptr<cv::text::ERFilter> getFilterStage2();

//   vector<cv::Mat3f> getChannels();

//   vector<cv::Mat3f> createChannels(cv::Mat3f src);
  
//   virtual cv::Ptr<cv::text::ERFilter> obtainFilterStage1(const cv::String& filename, 
//         int thresholdDelta, float minArea,
//         float maxArea, float minProbability, 
//         bool nonMaxSuppression, float minProbabilityDiff);

//   virtual cv::Ptr<cv::text::ERFilter> obtainFilterStage2(const cv::String& filename, float minProbability);
//   void ChanneliseFilters();

//   int runFilter(vector<cv::Ptr<cv::text::ERFilter>>cb_vector, cv::Mat3f src, vector<cv::Mat3f> channels, 
//   vector<vector<cv::text::ERStat>> &regions, vector<vector<cv::Vec2i> > &groups, vector<cv::Rect> &groups_rects);

//   void RunFilters(vector<cv::Mat3f> channels, vector<vector<cv::Vec2i>> &groups, vector<cv::Rect> &groups_rects, 
// vector<cv::Ptr<cv::text::ERFilter>> cb_vector);
//   }
  typedef
    struct TextDetection
      TextDetection;
#endif
// #ifdef __cplusplus
// extern "C" {
// #endif
// #if defined(__STDC__) || defined(__cplusplus)
//   extern void c_function(TextDetection*);   /* ANSI C prototypes */
//   extern TextDetection* cplusplus_callback_function(TextDetection*);
// #else
//   extern void c_function();        /* K&R style */
//   extern TextDetection* cplusplus_callback_function();
// #endif
// #ifdef __cplusplus
// }
// #endif
