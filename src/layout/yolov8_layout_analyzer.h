#ifndef YOLOV8_LAYOUT_ANALYZER_H
#define YOLOV8_LAYOUT_ANALYZER_H

#include <memory>
#include <string>
#include <vector>
#include "onnx_session.h"
#include <opencv2/opencv.hpp>

namespace ocr {

struct YOLOv8LayoutResult {
    float box[4];
    float score;
    int class_id;
    std::string class_name;
};

enum class YOLOv8LayoutModelType {
    YOLOV8_PAPER,
    YOLOV8_REPORT,
    YOLOV8_PUBLAYNET,
    YOLOV8_GENERAL6,
};

class YOLOv8LayoutAnalyzer {
public:
    struct Config {
        std::string model_path;
        YOLOv8LayoutModelType model_type = YOLOv8LayoutModelType::YOLOV8_PAPER;
        bool use_gpu = false;
        int gpu_id = 0;
        float conf_thresh = 0.5f;
        float iou_thresh = 0.5f;
    };

    explicit YOLOv8LayoutAnalyzer(const Config& cfg);
    ~YOLOv8LayoutAnalyzer();

    YOLOv8LayoutAnalyzer(const YOLOv8LayoutAnalyzer&) = delete;
    YOLOv8LayoutAnalyzer& operator=(const YOLOv8LayoutAnalyzer&) = delete;

    std::vector<YOLOv8LayoutResult> analyze(const cv::Mat& img);

    const std::vector<std::string>& get_labels() const { return labels_; }

private:
    std::unique_ptr<OnnxSession> session_;
    std::vector<std::string> labels_;
    float conf_thresh_;
    float iou_thresh_;
    static constexpr int IMG_SIZE = 640;

    static float iou_of(const float box1[4], const float box2[4]);
    std::vector<int> nms(const std::vector<std::vector<float>>& box_scores,
                         float iou_thresh);
};

} // namespace ocr

#endif /* YOLOV8_LAYOUT_ANALYZER_H */