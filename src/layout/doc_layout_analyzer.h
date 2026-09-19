#ifndef DOC_LAYOUT_ANALYZER_H
#define DOC_LAYOUT_ANALYZER_H

#include <memory>
#include <string>
#include <vector>
#include "onnx_session.h"
#include <opencv2/opencv.hpp>

namespace ocr {

/* A single document layout detection result. */
struct DocLayoutResult {
    /* Bounding box [x1, y1, x2, y2] in original image coordinates. */
    float box[4];
    /* Confidence score. */
    float score;
    /* Class index. */
    int class_id;
    /* Class name (from model metadata). */
    std::string class_name;
    /* Reading order index (0-based, for ordered models). */
    int order;
};

/* Document layout analyzer using pp_doclayoutv2 ONNX model.
   Input: 800x800, letterbox padding, BGR->RGB, /255 (no mean/std).
   Inputs: 2 tensors (image, scale_factor).
   Outputs: 2 tensors (boxes [total,6], box_nums [batch]).
   Box format: [class_id, score, x1, y1, x2, y2]. */
class DocLayoutAnalyzer {
public:
    struct Config {
        std::string model_path;     /* pp_doclayoutv2.onnx path (required) */
        bool   use_gpu = false;
        int    gpu_id = 0;
        float  conf_thresh = 0.5f;  /* confidence threshold */
        float  iou_thresh = 0.5f;   /* NMS IoU threshold */
    };

    explicit DocLayoutAnalyzer(const Config& cfg);
    ~DocLayoutAnalyzer();

    DocLayoutAnalyzer(const DocLayoutAnalyzer&) = delete;
    DocLayoutAnalyzer& operator=(const DocLayoutAnalyzer&) = delete;

    /* Analyze document layout in a BGR image.
       Returns layout results sorted by reading order. */
    std::vector<DocLayoutResult> analyze(const cv::Mat& img);

    /* Get the list of class names loaded from the model. */
    const std::vector<std::string>& get_labels() const { return labels_; }

private:
    std::unique_ptr<OnnxSession> session_;
    std::vector<std::string> labels_;
    float conf_thresh_;
    float iou_thresh_;
    static constexpr int IMG_SIZE = 800;

    /* Compute IoU between two boxes. */
    static float iou_of(const float box1[4], const float box2[4]);

    /* Hard NMS for boxes with scores. */
    std::vector<int> hard_nms(const std::vector<std::vector<float>>& box_scores,
                              float iou_thresh);

    /* Letterbox preprocessing: resize + pad to IMG_SIZE x IMG_SIZE. */
    cv::Mat letterbox(const cv::Mat& img, float& ratio, int& pad_left, int& pad_top);

    /* Scale boxes from letterbox space back to original image coordinates. */
    void scale_boxes(std::vector<std::vector<float>>& boxes,
                     float ratio, int pad_left, int pad_top,
                     int ori_h, int ori_w);
};

} // namespace ocr

#endif /* DOC_LAYOUT_ANALYZER_H */
