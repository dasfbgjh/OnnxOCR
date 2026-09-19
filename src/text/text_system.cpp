#include "text_system.h"

#include "utils.h"

namespace ocr {

TextSystem::TextSystem(const TextSystemConfig& cfg)
    : use_angle_cls_(cfg.use_angle_cls),
      use_rapid_orientation_(cfg.use_rapid_orientation),
      drop_score_(cfg.drop_score) {

    detector_ = std::make_unique<TextDetector>(cfg.det_cfg);
    recognizer_ = std::make_unique<TextRecognizer>(cfg.rec_cfg);
    if (use_angle_cls_ && !use_rapid_orientation_) {
        classifier_ = std::make_unique<TextClassifier>(cfg.cls_cfg);
    }
    if (use_angle_cls_ && use_rapid_orientation_) {
        orientation_classifier_ = std::make_unique<RapidOrientationClassifier>(cfg.orientation_cfg);
    }
}

TextSystem::~TextSystem() = default;

std::vector<TextSystem::Result> TextSystem::run(const cv::Mat& img, bool cls) {
    std::vector<Result> results;

    // 1. Text detection
    auto dt_boxes = detector_->detect(img);
    if (dt_boxes.empty()) return results;

    // 2. Sort boxes (top to bottom, left to right)
    dt_boxes = sorted_boxes(dt_boxes);

    // 3. Crop text regions
    std::vector<cv::Mat> img_crop_list;
    img_crop_list.reserve(dt_boxes.size());
    for (const auto& box : dt_boxes) {
        cv::Mat crop = get_rotate_crop_image(img, box);
        if (!crop.empty()) {
            img_crop_list.push_back(crop);
        }
    }

    if (img_crop_list.empty()) return results;

    // 4. Angle classification (optional)
    if (use_angle_cls_ && cls) {
        if (use_rapid_orientation_ && orientation_classifier_) {
            orientation_classifier_->classify(img_crop_list);
        } else if (classifier_) {
            classifier_->classify(img_crop_list);
        }
    }

    // 5. Text recognition
    auto rec_res = recognizer_->recognize(img_crop_list);

    // 6. Filter by drop_score and build results
    for (size_t i = 0; i < dt_boxes.size() && i < rec_res.size(); ++i) {
        if (rec_res[i].score >= drop_score_) {
            Result r;
            r.box = dt_boxes[i];
            r.text = rec_res[i].text;
            r.score = rec_res[i].score;
            results.push_back(std::move(r));
        }
    }

    return results;
}

} // namespace ocr