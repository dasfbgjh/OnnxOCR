#ifndef TEXT_DETECTOR_H
#define TEXT_DETECTOR_H

#include <memory>
#include <string>
#include <vector>
#include "onnx_session.h"
#include "pre_process.h"
#include "db_post_process.h"
#include "utils.h"
#include <opencv2/opencv.hpp>

namespace ocr {

/* Text detector using DB (Differentiable Binarization) algorithm.
   Wraps the detection ONNX model and performs pre/post-processing. */
class TextDetector {
public:
    struct Config {
        std::string model_path;
        bool   use_gpu = false;
        int    gpu_id = 0;

        float  det_limit_side_len = 960.0f;
        std::string det_limit_type = "max";
        float  det_db_thresh = 0.3f;
        float  det_db_box_thresh = 0.6f;
        float  det_db_unclip_ratio = 1.5f;
        bool   use_dilation = false;
        std::string det_db_score_mode = "fast";
        std::string det_box_type = "quad";
    };

    explicit TextDetector(const Config& cfg);
    ~TextDetector();

    TextDetector(const TextDetector&) = delete;
    TextDetector& operator=(const TextDetector&) = delete;

    /* Run text detection on a BGR image.
       Returns detected text boxes (4 corner points each). */
    std::vector<TextBox> detect(const cv::Mat& img);

private:
    std::unique_ptr<OnnxSession> session_;
    DetPreProcessConfig pre_cfg_;
    DBPostProcessConfig post_cfg_;
};

} // namespace ocr

#endif /* TEXT_DETECTOR_H */
