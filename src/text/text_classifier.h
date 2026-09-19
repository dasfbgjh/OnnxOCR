#ifndef TEXT_CLASSIFIER_H
#define TEXT_CLASSIFIER_H

#include <memory>
#include <string>
#include <vector>
#include "onnx_session.h"
#include <opencv2/opencv.hpp>

namespace ocr {

/* Text angle classifier (0 or 180 degrees).
   Wraps the classification ONNX model and performs pre/post-processing.
   Rotates images classified as 180 degrees. */
class TextClassifier {
public:
    struct Config {
        std::string model_path;
        bool use_gpu = false;
        int gpu_id = 0;
        int cls_batch_num = 6;
        int cls_image_c = 3;
        int cls_image_h = 48;
        int cls_image_w = 192;
        float cls_thresh = 0.9f;
    };

    explicit TextClassifier( const Config &cfg );
    ~TextClassifier();

    TextClassifier( const TextClassifier & ) = delete;
    TextClassifier &operator=( const TextClassifier & ) = delete;

    /* Classify and rotate images.
       Returns a list of (label, score) pairs, same order as input.
       Images in img_list may be rotated in-place if classified as 180. */
    struct Result {
        std::string label;
        float score;
    };
    std::vector<Result> classify( std::vector<cv::Mat> &img_list );

private:
    std::unique_ptr<OnnxSession> session_;
    int img_c_, img_h_, img_w_;
    int batch_num_;
    float cls_thresh_;
    std::vector<std::string> labels_{ "0", "180" };
};

} // namespace ocr

#endif /* TEXT_CLASSIFIER_H */
