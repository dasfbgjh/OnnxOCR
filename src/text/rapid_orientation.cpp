#include "rapid_orientation.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace ocr {

RapidOrientationClassifier::RapidOrientationClassifier(const Config& cfg)
    : batch_num_(cfg.batch_num), cls_thresh_(cfg.cls_thresh) {

    session_ = std::make_unique<OnnxSession>(cfg.model_path, cfg.use_gpu, cfg.gpu_id);

    std::vector<int64_t> warmup_shape = {1, 3, 224, 224};
    std::vector<float> dummy(3 * 224 * 224, 0.0f);
    try {
        session_->run(warmup_shape, dummy);
    } catch (...) {}
}

RapidOrientationClassifier::~RapidOrientationClassifier() = default;

cv::Mat RapidOrientationClassifier::resize_short(const cv::Mat& img, int short_size) {
    int h = img.rows;
    int w = img.cols;
    if (h <= 0 || w <= 0) return img;

    float scale = static_cast<float>(short_size) / std::min(w, h);
    int new_w = std::max(1, static_cast<int>(std::round(w * scale)));
    int new_h = std::max(1, static_cast<int>(std::round(h * scale)));
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LANCZOS4);
    return resized;
}

cv::Mat RapidOrientationClassifier::center_crop(const cv::Mat& img, int size) {
    int h = img.rows;
    int w = img.cols;
    cv::Mat src = img;
    if (h < size || w < size) {
        float scale = static_cast<float>(size) / std::min(w, h);
        int new_w = static_cast<int>(std::ceil(w * scale));
        int new_h = static_cast<int>(std::ceil(h * scale));
        cv::resize(src, src, cv::Size(new_w, new_h), 0, 0, cv::INTER_LANCZOS4);
        h = src.rows;
        w = src.cols;
    }
    int left = std::max(0, (w - size) / 2);
    int top = std::max(0, (h - size) / 2);
    return src(cv::Rect(left, top, size, size));
}

cv::Mat RapidOrientationClassifier::preprocess(const cv::Mat& img) {
    cv::Mat bgr = img;
    if (bgr.channels() == 1) {
        cv::cvtColor(bgr, bgr, cv::COLOR_GRAY2BGR);
    }

    bgr = resize_short(bgr, 256);
    bgr = center_crop(bgr, 224);

    bgr.convertTo(bgr, CV_32F, 1.0 / 255.0);
    cv::Scalar mean_s(0.485, 0.456, 0.406);
    cv::Scalar std_s(0.229, 0.224, 0.225);
    bgr = (bgr - mean_s) / std_s;

    return bgr;
}

std::vector<float> RapidOrientationClassifier::softmax(const float* logits, int n) {
    float max_val = *std::max_element(logits, logits + n);
    std::vector<float> exp_vals(n);
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        exp_vals[i] = std::exp(logits[i] - max_val);
        sum += exp_vals[i];
    }
    for (int i = 0; i < n; ++i) {
        exp_vals[i] /= sum;
    }
    return exp_vals;
}

static bool is_probabilities(const float* data, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        if (data[i] < -1e-4f) return false;
        sum += data[i];
    }
    return std::abs(sum - 1.0f) < 1e-2f;
}

static std::vector<float> normalize_outputs(const float* row, int n) {
    if (is_probabilities(row, n)) {
        return std::vector<float>(row, row + n);
    }
    return RapidOrientationClassifier::softmax(row, n);
}

std::vector<RapidOrientationClassifier::Result>
RapidOrientationClassifier::classify(std::vector<cv::Mat>& img_list) {
    int img_num = static_cast<int>(img_list.size());
    if (img_num == 0) return {};

    std::vector<Result> cls_res(img_num, {"0", 0.0f});

    for (int beg = 0; beg < img_num; beg += batch_num_) {
        int end = std::min(img_num, beg + batch_num_);
        int batch_size = end - beg;

        std::vector<float> batch_data;
        batch_data.reserve(static_cast<size_t>(batch_size * 3 * 224 * 224));

        for (int i = beg; i < end; ++i) {
            cv::Mat norm = preprocess(img_list[i]);

            std::vector<cv::Mat> channels(3);
            cv::split(norm, channels);
            size_t total = static_cast<size_t>(norm.total());
            for (int c = 0; c < 3; ++c) {
                const float* ptr = channels[c].ptr<float>();
                batch_data.insert(batch_data.end(), ptr, ptr + total);
            }
        }

        std::vector<int64_t> input_shape = {batch_size, 3, 224, 224};
        auto outputs = session_->run(input_shape, batch_data);

        auto& output = outputs[0];
        const float* output_data = output.GetTensorData<float>();
        auto info = output.GetTensorTypeAndShapeInfo();
        auto shape = info.GetShape();
        int out_classes = static_cast<int>(shape.back());
        if (out_classes > static_cast<int>(labels_.size())) {
            out_classes = static_cast<int>(labels_.size());
        }

        for (int r = 0; r < batch_size; ++r) {
            const float* row = output_data + r * static_cast<int>(shape.back());
            auto probs = normalize_outputs(row, static_cast<int>(shape.back()));

            int label_idx = 0;
            float max_prob = probs[0];
            for (int c = 1; c < static_cast<int>(probs.size()); ++c) {
                if (probs[c] > max_prob) {
                    max_prob = probs[c];
                    label_idx = c;
                }
            }

            int orig_idx = beg + r;
            std::string label = (label_idx < static_cast<int>(labels_.size()))
                                    ? labels_[label_idx]
                                    : std::to_string(label_idx);
            cls_res[orig_idx] = {label, max_prob};

            if (max_prob >= cls_thresh_) {
                if (label == "90") {
                    cv::rotate(img_list[orig_idx], img_list[orig_idx], cv::ROTATE_90_COUNTERCLOCKWISE);
                } else if (label == "180") {
                    cv::rotate(img_list[orig_idx], img_list[orig_idx], cv::ROTATE_180);
                } else if (label == "270") {
                    cv::rotate(img_list[orig_idx], img_list[orig_idx], cv::ROTATE_90_CLOCKWISE);
                }
            }
        }
    }

    return cls_res;
}

} // namespace ocr