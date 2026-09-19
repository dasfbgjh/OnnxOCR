#ifndef RAPID_ORIENTATION_H
#define RAPID_ORIENTATION_H

#include <memory>
#include <string>
#include <vector>
#include "onnx_session.h"
#include <opencv2/opencv.hpp>

namespace ocr {

class RapidOrientationClassifier {
public:
    struct Config {
        std::string model_path;
        bool use_gpu = false;
        int gpu_id = 0;
        int batch_num = 6;
        float cls_thresh = 0.9f;
    };

    explicit RapidOrientationClassifier( const Config &cfg );
    ~RapidOrientationClassifier();

    RapidOrientationClassifier( const RapidOrientationClassifier & ) = delete;
    RapidOrientationClassifier &operator=( const RapidOrientationClassifier & ) = delete;

    struct Result {
        std::string label;
        float score;
    };
    std::vector<Result> classify( std::vector<cv::Mat> &img_list );

    static std::vector<float> softmax( const float *logits, int n );

private:
    static cv::Mat preprocess( const cv::Mat &img );
    static cv::Mat resize_short( const cv::Mat &img, int short_size = 256 );
    static cv::Mat center_crop( const cv::Mat &img, int size = 224 );

    std::unique_ptr<OnnxSession> session_;
    int batch_num_;
    float cls_thresh_;
    std::vector<std::string> labels_{ "0", "90", "180", "270" };
};

} // namespace ocr

#endif /* RAPID_ORIENTATION_H */