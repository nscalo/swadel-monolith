#include "text_recognition.hpp"

namespace txtdecode {

    int obtainHMMDecoder(cv::Mat image, const std::string& filename, vector<string>& lexicon, std::string& output_text, 
    vector<cv::Rect>* boxes, vector<string>* words, vector<float>* confidences) {

        cv::Ptr<cv::text::OCRHMMDecoder::ClassifierCallback> hmm = cv::text::loadOCRHMMClassifierNM(filename);

        // character recognition vocabulary
        string voc = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

        cv::Mat transition_probabilities;
        cv::Mat emission_probabilities = cv::Mat::eye((int)voc.size(), (int)voc.size(), CV_64FC1);

        cv::text::createOCRHMMTransitionsTable(voc, lexicon, transition_probabilities);

        cv::Ptr<cv::text::OCRHMMDecoder> ocrNM  = cv::text::OCRHMMDecoder::create(hmm, voc, 
        transition_probabilities, emission_probabilities, cv::text::OCR_DECODER_VITERBI);

        ocrNM->run(image, output_text, boxes, words, confidences);

        return 0;

    }

}