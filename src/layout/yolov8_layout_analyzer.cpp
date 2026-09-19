#include "yolov8_layout_analyzer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

namespace ocr {

YOLOv8LayoutAnalyzer::YOLOv8LayoutAnalyzer(const Config& cfg)
    : conf_thresh_(cfg.conf_thresh),
      iou_thresh_(cfg.iou_thresh) {

    session_ = std::make_unique<OnnxSession>(cfg.model_path, cfg.use_gpu, cfg.gpu_id);

    auto meta = session_->get_metadata();
    auto it = meta.find("character");
    if (it != meta.end()) {
        std::string chars = it->second;
        size_t start = 0;
        while (start < chars.size()) {
            size_t end = chars.find('\n', start);
            if (end == std::string::npos) end = chars.size();
            labels_.push_back(chars.substr(start, end - start));
            start = end + 1;
        }
    }

    if (labels_.empty()) {
        switch (cfg.model_type) {
        case YOLOv8LayoutModelType::YOLOV8_PAPER:
            labels_ = {"title", "text", "figure", "table", "caption", "equation", "reference"};
            break;
        case YOLOv8LayoutModelType::YOLOV8_REPORT:
            labels_ = {"title", "text", "figure", "table", "header", "footer"};
            break;
        case YOLOv8LayoutModelType::YOLOV8_PUBLAYNET:
            labels_ = {"text", "title", "list", "table", "figure"};
            break;
        case YOLOv8LayoutModelType::YOLOV8_GENERAL6:
            labels_ = {"text", "title", "figure", "table", "caption", "equation"};
            break;
        }
    }

    try {
        std::vector<int64_t> warmup_shape = {1, 3, IMG_SIZE, IMG_SIZE};
        std::vector<float> dummy(3 * IMG_SIZE * IMG_SIZE, 0.0f);
        session_->run(warmup_shape, dummy);
    } catch (...) {}
}

YOLOv8LayoutAnalyzer::~YOLOv8LayoutAnalyzer() = default;

float YOLOv8LayoutAnalyzer::iou_of(const float box1[4], const float box2[4]) {
    float x1 = std::max(box1[0], box2[0]);
    float y1 = std::max(box1[1], box2[1]);
    float x2 = std::min(box1[2], box2[2]);
    float y2 = std::min(box1[3], box2[3]);
    float w = std::max(0.0f, x2 - x1);
    float h = std::max(0.0f, y2 - y1);
    float inter = w * h;
    float area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]);
    float area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]);
    float union_area = area1 + area2 - inter;
    return inter / (union_area + 1e-5f);
}

std::vector<int> YOLOv8LayoutAnalyzer::nms(
    const std::vector<std::vector<float>>& box_scores,
    float iou_thresh) {

    int n = static_cast<int>(box_scores.size());
    if (n == 0) return {};

    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return box_scores[a][4] > box_scores[b][4];
    });

    std::vector<int> picked;
    while (!indices.empty()) {
        int current = indices[0];
        picked.push_back(current);
        if (indices.size() == 1) break;

        float current_box[4] = {box_scores[current][0], box_scores[current][1],
                                 box_scores[current][2], box_scores[current][3]};

        std::vector<int> remaining;
        for (size_t i = 1; i < indices.size(); ++i) {
            int idx = indices[i];
            float other_box[4] = {box_scores[idx][0], box_scores[idx][1],
                                   box_scores[idx][2], box_scores[idx][3]};
            if (iou_of(current_box, other_box) < iou_thresh) {
                remaining.push_back(idx);
            }
        }
        indices = std::move(remaining);
    }
    return picked;
}

std::vector<YOLOv8LayoutResult> YOLOv8LayoutAnalyzer::analyze(const cv::Mat& img) {
    std::vector<YOLOv8LayoutResult> results;
    if (img.empty()) return results;

    int ori_h = img.rows;
    int ori_w = img.cols;

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(IMG_SIZE, IMG_SIZE));
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);

    std::vector<float> input_data(3 * IMG_SIZE * IMG_SIZE);
    std::vector<cv::Mat> channels(3);
    cv::split(resized, channels);
    for (int c = 0; c < 3; ++c) {
        std::memcpy(input_data.data() + c * IMG_SIZE * IMG_SIZE,
                    channels[c].data, IMG_SIZE * IMG_SIZE * sizeof(float));
    }

    std::vector<int64_t> input_shape = {1, 3, IMG_SIZE, IMG_SIZE};
    auto outputs = session_->run(input_shape, input_data);

    if (outputs.empty()) return results;

    auto& out = outputs[0];
    auto info = out.GetTensorTypeAndShapeInfo();
    auto shape = info.GetShape();
    const float* data = out.GetTensorData<float>();

    if (shape.size() != 3) return results;
    int64_t num_classes_plus_4 = shape[1];
    int64_t num_preds = shape[2];
    int num_classes = static_cast<int>(num_classes_plus_4 - 4);

    if (num_classes <= 0) return results;

    std::vector<std::vector<float>> all_boxes;
    std::vector<float> all_scores;
    std::vector<int> all_class_ids;

    float scale_x = static_cast<float>(ori_w) / IMG_SIZE;
    float scale_y = static_cast<float>(ori_h) / IMG_SIZE;

    for (int64_t i = 0; i < num_preds; ++i) {
        float cx = data[0 * num_preds + i];
        float cy = data[1 * num_preds + i];
        float w = data[2 * num_preds + i];
        float h = data[3 * num_preds + i];

        float best_score = 0.0f;
        int best_class = 0;
        for (int c = 0; c < num_classes; ++c) {
            float s = data[(4 + c) * num_preds + i];
            if (s > best_score) {
                best_score = s;
                best_class = c;
            }
        }

        if (best_score <= conf_thresh_) continue;

        float x1 = (cx - w / 2.0f) * scale_x;
        float y1 = (cy - h / 2.0f) * scale_y;
        float x2 = (cx + w / 2.0f) * scale_x;
        float y2 = (cy + h / 2.0f) * scale_y;

        x1 = std::max(0.0f, std::min(x1, static_cast<float>(ori_w)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(ori_h)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(ori_w)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(ori_h)));

        all_boxes.push_back({x1, y1, x2, y2, best_score});
        all_scores.push_back(best_score);
        all_class_ids.push_back(best_class);
    }

    std::vector<std::vector<float>> per_class_boxes;
    std::vector<int> per_class_ids;
    std::vector<int> per_class_orig_idx;

    std::vector<std::vector<float>> nms_input;
    for (size_t i = 0; i < all_boxes.size(); ++i) {
        nms_input.push_back(all_boxes[i]);
    }
    auto keep = nms(nms_input, iou_thresh_);

    for (int idx : keep) {
        YOLOv8LayoutResult r;
        r.box[0] = all_boxes[idx][0];
        r.box[1] = all_boxes[idx][1];
        r.box[2] = all_boxes[idx][2];
        r.box[3] = all_boxes[idx][3];
        r.score = all_boxes[idx][4];
        r.class_id = all_class_ids[idx];
        r.class_name = (r.class_id < static_cast<int>(labels_.size()))
                       ? labels_[r.class_id]
                       : ("class_" + std::to_string(r.class_id));
        results.push_back(std::move(r));
    }

    std::sort(results.begin(), results.end(), [](const YOLOv8LayoutResult& a, const YOLOv8LayoutResult& b) {
        if (a.class_id != b.class_id) return a.class_id < b.class_id;
        return a.score > b.score;
    });

    return results;
}

} // namespace ocr