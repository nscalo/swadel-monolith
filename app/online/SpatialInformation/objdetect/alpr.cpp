#include <iostream>
#include <stdio.h>
#include <string>
#include "alpr.h"
#include <opencv2/opencv.hpp>
#include <opencv2/videoio/videoio_c.h>
#include <opencv2/imgproc/imgproc_c.h>

class ObjDetectALPR {

public:
    ObjDetectALPR(cv::Mat m_imgBGR) {

    }




protected:
    std::string       m_windowName;
    cv::VideoCapture  m_cap;
    cv::Mat           m_imgBGR;
    cv::Mat           m_imgRGB;

}

int main( int argc, const char * argv[] ) {

    // Initialize the library using United States style license plates.
    // You can use other countries/regions as well (for example: "eu", "au", or "kr")
    alpr::Alpr openalpr("us", "gb.conf");

    // Optionally specify the top N possible plates to return (with confidences).  Default is 10
    openalpr.setTopN(20);

    // Optionally, provide the library with a region for pattern matching.  This improves accuracy by
    // comparing the plate text with the regional pattern.
    openalpr.setDefaultRegion("md");

    // Make sure the library loaded before continuing.
    // For example, it could fail if the config/runtime_data is not found
    if (openalpr.isLoaded() == false)
    {
        std::cerr << "Error loading OpenALPR" << std::endl;
        return 1;
    }

    // Recognize an image file.  You could alternatively provide the image bytes in-memory.
    alpr::AlprResults results = openalpr.recognize("car_image.jpg");

    // Iterate through the results.  There may be multiple plates in an image,
    // and each plate return sthe top N candidates.
    for (int i = 0; i < results.plates.size(); i++)
    {
        alpr::AlprPlateResult plate = results.plates[i];
        std::cout << "plate" << i << ": " << plate.topNPlates.size() << " results" << std::endl;

        for (int k = 0; k < plate.topNPlates.size(); k++)
        {
        alpr::AlprPlate candidate = plate.topNPlates[k];
        std::cout << "    - " << candidate.characters << "\t confidence: " << candidate.overall_confidence;
        std::cout << "\t pattern_match: " << candidate.matches_template << std::endl;
        }
    }

    return 0;
}

