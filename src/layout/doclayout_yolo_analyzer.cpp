#include "doclayout_yolo_analyzer.h"
#include "doc_layout_analyzer.h"
#include "layout_analyzer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

namespace ocr {

DocLayoutYOLOAnalyzer::DocLayoutYOLOAnalyzer(const Config& cfg)
    : conf_thresh_(cfg.conf_thresh),
      iou_thresh_(cfg.iou_thresh),
      is_pp_doclayout_(cfg.model_type == DocLayoutYOLOModelType::PP_DOCLAYOUT_V2),
      is_pp_layout_(cfg.model_type == DocLayoutYOLOModelType::PP_LAYOUT_CDLA ||
                    cfg.model_type == DocLayoutYOLOModelType::PP_LAYOUT_PUBLAYNET) {

    if (is_pp_layout_) {
        img_size_ = 800;
        use_letterbox_ = false;
        LayoutAnalyzer::Config la_cfg;
        la_cfg.model_path = cfg.model_path;
        la_cfg.model_type = (cfg.model_type == DocLayoutYOLOModelType::PP_LAYOUT_PUBLAYNET)
                                ? LayoutModelType::PP_LAYOUT_PUBLAYNET
                                : LayoutModelType::PP_LAYOUT_CDLA;
        la_cfg.use_gpu = cfg.use_gpu;
        la_cfg.gpu_id = cfg.gpu_id;
        la_cfg.conf_thresh = cfg.conf_thresh;
        la_cfg.iou_thresh = cfg.iou_thresh;
        layout_analyzer_ = std::make_unique<LayoutAnalyzer>(la_cfg);
        labels_ = layout_analyzer_->get_labels();
        return;
    }

    if (is_pp_doclayout_) {
        img_size_ = 800;
        use_letterbox_ = true;
        DocLayoutAnalyzer::Config dl_cfg;
        dl_cfg.model_path = cfg.model_path;
        dl_cfg.use_gpu = cfg.use_gpu;
        dl_cfg.gpu_id = cfg.gpu_id;
        dl_cfg.conf_thresh = cfg.conf_thresh;
        dl_cfg.iou_thresh = cfg.iou_thresh;
        doc_layout_analyzer_ = std::make_unique<DocLayoutAnalyzer>(dl_cfg);
        labels_ = doc_layout_analyzer_->get_labels();
        return;
    }

    switch (cfg.model_type) {
    case DocLayoutYOLOModelType::YOLOV8_PAPER:
    case DocLayoutYOLOModelType::YOLOV8_REPORT:
    case DocLayoutYOLOModelType::YOLOV8_PUBLAYNET:
    case DocLayoutYOLOModelType::YOLOV8_GENERAL6:
        img_size_ = 640;
        use_letterbox_ = false;
        break;
    case DocLayoutYOLOModelType::D4LA:
        img_size_ = 1600;
        use_letterbox_ = true;
        break;
    case DocLayoutYOLOModelType::DOCSYNTH:
        img_size_ = 1120;
        use_letterbox_ = true;
        break;
    case DocLayoutYOLOModelType::DOCSTRUCTBENCH:
    default:
        img_size_ = 1024;
        use_letterbox_ = true;
        break;
    }

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
        case DocLayoutYOLOModelType::YOLOV8_PAPER:
            labels_ = {"title", "text", "figure", "table", "caption", "equation", "reference"};
            break;
        case DocLayoutYOLOModelType::YOLOV8_REPORT:
            labels_ = {"title", "text", "figure", "table", "header", "footer"};
            break;
        case DocLayoutYOLOModelType::YOLOV8_PUBLAYNET:
            labels_ = {"text", "title", "list", "table", "figure"};
            break;
        case DocLayoutYOLOModelType::YOLOV8_GENERAL6:
            labels_ = {"text", "title", "figure", "table", "caption", "equation"};
            break;
        default:
            labels_ = {
                "caption", "footnote", "formula", "list-item", "page-footer",
                "page-header", "picture", "section-header", "table", "text",
                "title"
            };
            break;
        }
    }

    try {
        std::vector<int64_t> warmup_shape = {1, 3, img_size_, img_size_};
        std::vector<float> dummy(3 * img_size_ * img_size_, 0.0f);
        session_->run(warmup_shape, dummy);
    } catch (...) {}
}

DocLayoutYOLOAnalyzer::~DocLayoutYOLOAnalyzer() = default;

float DocLayoutYOLOAnalyzer::iou_of(const float box1[4], const float box2[4]) {
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

std::vector<int> DocLayoutYOLOAnalyzer::nms(
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

cv::Mat DocLayoutYOLOAnalyzer::letterbox(const cv::Mat& img, float& ratio,
                                          int& pad_left, int& pad_top) {
    int h = img.rows, w = img.cols;
    ratio = std::min(static_cast<float>(img_size_) / h,
                     static_cast<float>(img_size_) / w);
    ratio = std::min(ratio, 1.0f);

    int new_w = static_cast<int>(std::round(w * ratio));
    int new_h = static_cast<int>(std::round(h * ratio));

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    int dw = img_size_ - new_w;
    int dh = img_size_ - new_h;
    pad_top = static_cast<int>(std::round(dh / 2.0f - 0.1));
    int pad_bottom = dh - pad_top;
    pad_left = static_cast<int>(std::round(dw / 2.0f - 0.1));
    int pad_right = dw - pad_left;

    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, pad_top, pad_bottom, pad_left, pad_right,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    return padded;
}

void DocLayoutYOLOAnalyzer::scale_boxes(std::vector<std::vector<float>>& boxes,
                                         float ratio, int pad_left, int pad_top,
                                         int ori_h, int ori_w) {
    for (auto& box : boxes) {
        box[0] -= pad_left;
        box[1] -= pad_top;
        box[2] -= pad_left;
        box[3] -= pad_top;
        box[0] /= ratio;
        box[1] /= ratio;
        box[2] /= ratio;
        box[3] /= ratio;
        box[0] = std::max(0.0f, std::min(box[0], static_cast<float>(ori_w)));
        box[1] = std::max(0.0f, std::min(box[1], static_cast<float>(ori_h)));
        box[2] = std::max(0.0f, std::min(box[2], static_cast<float>(ori_w)));
        box[3] = std::max(0.0f, std::min(box[3], static_cast<float>(ori_h)));
    }
}

std::vector<DocLayoutYOLOResult> DocLayoutYOLOAnalyzer::analyze(const cv::Mat& img) {
    std::vector<DocLayoutYOLOResult> results;
    if (img.empty()) return results;

    if (is_pp_layout_) {
        auto la_results = layout_analyzer_->analyze(img);
        results.reserve(la_results.size());
        for (int i = 0; i < static_cast<int>(la_results.size()); ++i) {
            const auto& r = la_results[i];
            DocLayoutYOLOResult yr;
            yr.box[0] = r.box[0];
            yr.box[1] = r.box[1];
            yr.box[2] = r.box[2];
            yr.box[3] = r.box[3];
            yr.score = r.score;
            yr.class_id = r.class_id;
            yr.class_name = r.class_name;
            yr.order = i;
            results.push_back(std::move(yr));
        }
        return results;
    }

    if (is_pp_doclayout_) {
        auto dl_results = doc_layout_analyzer_->analyze(img);
        results.reserve(dl_results.size());
        for (const auto& r : dl_results) {
            DocLayoutYOLOResult yr;
            yr.box[0] = r.box[0];
            yr.box[1] = r.box[1];
            yr.box[2] = r.box[2];
            yr.box[3] = r.box[3];
            yr.score = r.score;
            yr.class_id = r.class_id;
            yr.class_name = r.class_name;
            yr.order = r.order;
            results.push_back(std::move(yr));
        }
        return results;
    }

    int ori_h = img.rows;
    int ori_w = img.cols;

    float ratio = 1.0f;
    int pad_left = 0, pad_top = 0;
    cv::Mat preprocessed;

    if (use_letterbox_) {
        preprocessed = letterbox(img, ratio, pad_left, pad_top);
    } else {
        cv::resize(img, preprocessed, cv::Size(img_size_, img_size_));
    }

    cv::Mat rgb;
    cv::cvtColor(preprocessed, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

    std::vector<float> input_data(3 * img_size_ * img_size_);
    std::vector<cv::Mat> channels(3);
    cv::split(rgb, channels);
    for (int c = 0; c < 3; ++c) {
        std::memcpy(input_data.data() + c * img_size_ * img_size_,
                    channels[c].data, img_size_ * img_size_ * sizeof(float));
    }

    std::vector<int64_t> input_shape = {1, 3, img_size_, img_size_};
    auto outputs = session_->run(input_shape, input_data);

    if (outputs.empty()) return results;

    auto& out = outputs[0];
    auto info = out.GetTensorTypeAndShapeInfo();
    auto shape = info.GetShape();
    const float* data = out.GetTensorData<float>();

    if (shape.size() != 3) return results;

    int64_t num_preds_raw = shape[1];
    int64_t attr_per_pred = shape[2];

    int num_classes = 0;
    if (attr_per_pred > 4) {
        num_classes = static_cast<int>(attr_per_pred - 4);
    } else if (num_preds_raw > 4) {
        num_classes = static_cast<int>(num_preds_raw - 4);
    }
    if (num_classes <= 0) return results;

    std::vector<std::vector<float>> all_boxes;

    if (attr_per_pred > 4) {
        for (int64_t i = 0; i < num_preds_raw; ++i) {
            const float* row = data + i * attr_per_pred;
            float best_score = 0.0f;
            int best_class = 0;
            for (int c = 0; c < num_classes; ++c) {
                float s = row[4 + c];
                if (s > best_score) {
                    best_score = s;
                    best_class = c;
                }
            }
            if (best_score <= conf_thresh_) continue;
            all_boxes.push_back({row[0], row[1], row[2], row[3], best_score,
                                 static_cast<float>(best_class)});
        }
    } else {
        for (int64_t i = 0; i < attr_per_pred; ++i) {
            float cx = data[0 * attr_per_pred + i];
            float cy = data[1 * attr_per_pred + i];
            float w = data[2 * attr_per_pred + i];
            float h = data[3 * attr_per_pred + i];

            float best_score = 0.0f;
            int best_class = 0;
            for (int c = 0; c < num_classes; ++c) {
                float s = data[(4 + c) * attr_per_pred + i];
                if (s > best_score) {
                    best_score = s;
                    best_class = c;
                }
            }
            if (best_score <= conf_thresh_) continue;

            float x1 = cx - w / 2.0f;
            float y1 = cy - h / 2.0f;
            float x2 = cx + w / 2.0f;
            float y2 = cy + h / 2.0f;
            all_boxes.push_back({x1, y1, x2, y2, best_score,
                                 static_cast<float>(best_class)});
        }
    }

    if (all_boxes.empty()) return results;

    std::vector<std::vector<float>> nms_input;
    for (const auto& b : all_boxes) {
        nms_input.push_back({b[0], b[1], b[2], b[3], b[4]});
    }
    auto keep = nms(nms_input, iou_thresh_);

    std::vector<std::vector<float>> kept_boxes;
    for (int idx : keep) {
        kept_boxes.push_back(all_boxes[idx]);
    }

    if (use_letterbox_) {
        scale_boxes(kept_boxes, ratio, pad_left, pad_top, ori_h, ori_w);
    } else {
        float scale_x = static_cast<float>(ori_w) / img_size_;
        float scale_y = static_cast<float>(ori_h) / img_size_;
        for (auto& box : kept_boxes) {
            box[0] = std::max(0.0f, std::min(box[0] * scale_x, static_cast<float>(ori_w)));
            box[1] = std::max(0.0f, std::min(box[1] * scale_y, static_cast<float>(ori_h)));
            box[2] = std::max(0.0f, std::min(box[2] * scale_x, static_cast<float>(ori_w)));
            box[3] = std::max(0.0f, std::min(box[3] * scale_y, static_cast<float>(ori_h)));
        }
    }

    for (const auto& kb : kept_boxes) {
        DocLayoutYOLOResult r;
        r.box[0] = kb[0];
        r.box[1] = kb[1];
        r.box[2] = kb[2];
        r.box[3] = kb[3];
        r.score = kb[4];
        r.class_id = static_cast<int>(kb[5]);
        r.class_name = (r.class_id < static_cast<int>(labels_.size()))
                       ? labels_[r.class_id]
                       : ("class_" + std::to_string(r.class_id));
        r.order = 0;
        results.push_back(std::move(r));
    }

    std::sort(results.begin(), results.end(), [](const DocLayoutYOLOResult& a, const DocLayoutYOLOResult& b) {
        float ay = (a.box[1] + a.box[3]) / 2.0f;
        float by = (b.box[1] + b.box[3]) / 2.0f;
        if (std::abs(ay - by) > 10.0f) return ay < by;
        return a.box[0] < b.box[0];
    });

    for (int i = 0; i < static_cast<int>(results.size()); ++i) {
        results[i].order = i;
    }

    return results;
}

} // namespace ocr