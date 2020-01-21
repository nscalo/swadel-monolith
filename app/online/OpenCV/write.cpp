#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {
    float *output = (float *) malloc((long) 28*28);

    std::string s;

    std::string path = "/home/workspace/intel/mnist-gan-simple-gan/output_cpp_images/output";

    for (size_t a = 0; a < inferRequestsQueue.requests.size(); a++) {
        InferenceEngine::InferRequest request_ptr = inferRequestsQueue.requests[a]->getRequest();
        request_ptr.getReq()->GetUserData(reinterpret_cast<void**>(&output), NULL);
        cv::Mat mat(28, 28, CV_32F);
        for (int i = 0; i < mat.rows; ++i) {
            for (int j = 0; j < mat.cols; ++j) {
                mat.at<float>(i, j) = output[j*mat.rows + i];
            }
        }
        std::stringstream result;
        result << a;
        s = path + result.str() + ".png";
        cv::Mat dst(28, 28, CV_32F);
        cv::threshold(mat, dst, 0.0d, 255.0d, cv::THRESH_BINARY_INV);
        cv::imwrite(s, dst);
    }
}