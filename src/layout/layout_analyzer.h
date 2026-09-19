#ifndef LAYOUT_ANALYZER_H
#define LAYOUT_ANALYZER_H

#include <memory>
#include <string>
#include <vector>
#include "onnx_session.h"
#include <opencv2/opencv.hpp>

namespace ocr {

/* A single layout detection result. */
struct LayoutResult {
    /* Bounding box [x1, y1, x2, y2] in original image coordinates. */
    float box[4];
    /* Confidence score. */
    float score;
    /* Class index. */
    int class_id;
    /* Class name (from model metadata). */
    std::string class_name;
};

/* Layout type enum for model selection. */
enum class LayoutModelType {
    PP_LAYOUT_CDLA,       /* Chinese document layout: pp_layout_cdla */
    PP_LAYOUT_PUBLAYNET,  /* English document layout: pp_layout_publaynet */
};

/* Layout analyzer using PP-series detection models.
   Input: 800x608, ImageNet normalization.
   Output: 8 tensors (4 scores + 4 box distributions at strides 8, 16, 32, 64).
   Post-processing: DFL decode + per-class NMS. */
class LayoutAnalyzer {
public:
    struct Config {
        std::string model_path;             /* layout ONNX model path */
        LayoutModelType model_type = LayoutModelType::PP_LAYOUT_CDLA;
        bool   use_gpu = false;
        int    gpu_id = 0;
        float  conf_thresh = 0.5f;
        float  iou_thresh = 0.5f;
    };

    explicit LayoutAnalyzer(const Config& cfg);
    ~LayoutAnalyzer();

    LayoutAnalyzer(const LayoutAnalyzer&) = delete;
    LayoutAnalyzer& operator=(const LayoutAnalyzer&) = delete;

    /* Run layout analysis on a BGR image.
       Returns a list of layout results (boxes, scores, class names). */
    std::vector<LayoutResult> analyze(const cv::Mat& img);

    /* Get the list of class names loaded from the model. */
    const std::vector<std::string>& get_labels() const { return labels_; }

private:
    std::unique_ptr<OnnxSession> session_;
    std::vector<std::string> labels_;
    float conf_thresh_;
    float iou_thresh_;
    int img_h_;  /* 800 */
    int img_w_;  /* 608 */

    /* Strides for multiscale feature maps. */
    static constexpr int STRIDES[4] = {8, 16, 32, 64};
    static constexpr int NMS_TOP_K = 1000;
    static constexpr int KEEP_TOP_K = 100;

    /* Softmax along axis. */
    static void softmax(std::vector<float>& x, int rows, int cols);

    /* Hard NMS for boxes. */
    std::vector<int> hard_nms(const std::vector<std::vector<float>>& box_scores,
                              float iou_thresh, int top_k);

    /* Compute IoU between one box and a set of boxes. */
    static float iou_of(const float box1[4], const float box2[4]);

    /* Clip boxes to image bounds. */
    static void clip_boxes(std::vector<std::vector<float>>& boxes,
                           int img_h, int img_w);
};

} // namespace ocr

#endif /* LAYOUT_ANALYZER_H */
