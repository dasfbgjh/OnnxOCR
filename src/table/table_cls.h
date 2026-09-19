#ifndef TABLE_CLS_H
#define TABLE_CLS_H

#include <memory>
#include <string>
#include "onnx_session.h"
#include <opencv2/opencv.hpp>

namespace ocr {

enum class TableClsType {
    Q_CLS,
    PADDLE_CLS,
};

class TableClassifier {
public:
    struct Config {
        std::string model_path;
        TableClsType cls_type = TableClsType::Q_CLS;
        bool use_gpu = false;
        int gpu_id = 0;
    };

    explicit TableClassifier(const Config& cfg);
    ~TableClassifier();

    TableClassifier(const TableClassifier&) = delete;
    TableClassifier& operator=(const TableClassifier&) = delete;

    /* Classify table type: returns "wired" or "wireless" */
    std::string classify(const cv::Mat& img);

private:
    std::unique_ptr<OnnxSession> session_;
    TableClsType cls_type_;

    static constexpr int INP_H = 224;
    static constexpr int INP_W = 224;
    static constexpr float MEAN[3] = {0.485f, 0.456f, 0.406f};
    static constexpr float STD[3]  = {0.229f, 0.224f, 0.225f};

    /* PaddleCls: resize_short=256, center crop 224x224 */
    static constexpr int RESIZE_SHORT = 256;

    cv::Mat preprocess_paddle(const cv::Mat& img);
    cv::Mat preprocess_q(const cv::Mat& img);
};

} // namespace ocr

#endif /* TABLE_CLS_H */