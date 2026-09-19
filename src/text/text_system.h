#ifndef TEXT_SYSTEM_H
#define TEXT_SYSTEM_H

#include <memory>
#include <vector>
#include "text_detector.h"
#include "text_recognizer.h"
#include "text_classifier.h"
#include "rapid_orientation.h"
#include <opencv2/opencv.hpp>

namespace ocr {

/* Configuration for the entire OCR system. */
struct TextSystemConfig {
    TextDetector::Config det_cfg;
    TextRecognizer::Config rec_cfg;
    TextClassifier::Config cls_cfg;
    RapidOrientationClassifier::Config orientation_cfg;
    bool use_angle_cls = false;
    bool use_rapid_orientation = false;
    float drop_score = 0.5f;
};

/* Complete OCR pipeline: detection -> crop -> classification -> recognition. */
class TextSystem {
public:
    explicit TextSystem(const TextSystemConfig& cfg);
    ~TextSystem();

    TextSystem(const TextSystem&) = delete;
    TextSystem& operator=(const TextSystem&) = delete;

    /* A single OCR result: text box and recognized text + score. */
    struct Result {
        TextBox box;
        std::string text;
        float score;
    };

    /* Run the full OCR pipeline on a BGR image.
       If cls is true and angle classification is enabled, images will be
       classified and rotated before recognition. */
    std::vector<Result> run(const cv::Mat& img, bool cls = true);

private:
    std::unique_ptr<TextDetector> detector_;
    std::unique_ptr<TextRecognizer> recognizer_;
    std::unique_ptr<TextClassifier> classifier_;
    std::unique_ptr<RapidOrientationClassifier> orientation_classifier_;
    bool use_angle_cls_;
    bool use_rapid_orientation_;
    float drop_score_;
};

} // namespace ocr

#endif /* TEXT_SYSTEM_H */