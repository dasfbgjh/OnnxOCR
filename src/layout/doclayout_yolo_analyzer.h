#ifndef DOCLAYOUT_YOLO_ANALYZER_H
#define DOCLAYOUT_YOLO_ANALYZER_H

#include <memory>
#include <string>
#include <vector>
#include "onnx_session.h"
#include <opencv2/opencv.hpp>

namespace ocr {

struct DocLayoutYOLOResult {
    float box[4];
    float score;
    int class_id;
    std::string class_name;
    int order;
};

enum class DocLayoutYOLOModelType {
    /* PP-Structure: 800x608, ImageNet normalization, DFL decode */
    PP_LAYOUT_CDLA,
    PP_LAYOUT_PUBLAYNET,
    /* pp_doclayoutv2: 800x800, 3-input, letterbox (scaleup=False) */
    PP_DOCLAYOUT_V2,
    /* YOLOv8 series: 640x640, direct resize (no aspect-ratio preservation) */
    YOLOV8_PAPER,
    YOLOV8_REPORT,
    YOLOV8_PUBLAYNET,
    YOLOV8_GENERAL6,
    /* DocLayout YOLO series: letterbox preprocessing (preserves aspect ratio) */
    DOCSTRUCTBENCH,
    D4LA,
    DOCSYNTH,
};

class DocLayoutAnalyzer;
class LayoutAnalyzer;

class DocLayoutYOLOAnalyzer {
public:
    struct Config {
        std::string model_path;
        DocLayoutYOLOModelType model_type = DocLayoutYOLOModelType::DOCSTRUCTBENCH;
        bool use_gpu = false;
        int gpu_id = 0;
        float conf_thresh = 0.2f;
        float iou_thresh = 0.5f;
    };

    explicit DocLayoutYOLOAnalyzer(const Config& cfg);
    ~DocLayoutYOLOAnalyzer();

    DocLayoutYOLOAnalyzer(const DocLayoutYOLOAnalyzer&) = delete;
    DocLayoutYOLOAnalyzer& operator=(const DocLayoutYOLOAnalyzer&) = delete;

    std::vector<DocLayoutYOLOResult> analyze(const cv::Mat& img);

    const std::vector<std::string>& get_labels() const { return labels_; }

private:
    std::unique_ptr<OnnxSession> session_;
    std::unique_ptr<DocLayoutAnalyzer> doc_layout_analyzer_;
    std::unique_ptr<LayoutAnalyzer> layout_analyzer_;
    std::vector<std::string> labels_;
    float conf_thresh_;
    float iou_thresh_;
    int img_size_;
    bool use_letterbox_;
    bool is_pp_doclayout_;
    bool is_pp_layout_;

    static float iou_of(const float box1[4], const float box2[4]);
    std::vector<int> nms(const std::vector<std::vector<float>>& box_scores,
                         float iou_thresh);

    cv::Mat letterbox(const cv::Mat& img, float& ratio,
                      int& pad_left, int& pad_top);
    void scale_boxes(std::vector<std::vector<float>>& boxes,
                     float ratio, int pad_left, int pad_top,
                     int ori_h, int ori_w);
};

} // namespace ocr

#endif /* DOCLAYOUT_YOLO_ANALYZER_H */