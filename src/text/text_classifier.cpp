#include "text_classifier.h"

#include "pre_process.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ocr {

TextClassifier::TextClassifier(const Config& cfg)
    : img_c_(cfg.cls_image_c), img_h_(cfg.cls_image_h), img_w_(cfg.cls_image_w),
      batch_num_(cfg.cls_batch_num), cls_thresh_(cfg.cls_thresh) {

    session_ = std::make_unique<OnnxSession>(cfg.model_path, cfg.use_gpu, cfg.gpu_id);

    // Warm-up
    std::vector<int64_t> warmup_shape = {1, img_c_, img_h_, img_w_};
    std::vector<float> dummy(img_c_ * img_h_ * img_w_, 0.0f);
    try {
        session_->run(warmup_shape, dummy);
    } catch (...) {}
}

TextClassifier::~TextClassifier() = default;

std::vector<TextClassifier::Result> TextClassifier::classify(std::vector<cv::Mat>& img_list) {
    int img_num = static_cast<int>(img_list.size());
    if (img_num == 0) return {};

    // Calculate aspect ratios and sort
    std::vector<float> width_list(img_num);
    for (int i = 0; i < img_num; ++i) {
        width_list[i] = static_cast<float>(img_list[i].cols) / img_list[i].rows;
    }

    std::vector<int> indices(img_num);
    for (int i = 0; i < img_num; ++i) indices[i] = i;
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return width_list[a] < width_list[b];
    });

    std::vector<Result> cls_res(img_num, {"0", 0.0f});

    for (int beg = 0; beg < img_num; beg += batch_num_) {
        int end = std::min(img_num, beg + batch_num_);
        int batch_size = end - beg;

        // Find max wh_ratio
        float max_wh_ratio = 0.0f;
        for (int i = beg; i < end; ++i) {
            int idx = indices[i];
            float wh_ratio = static_cast<float>(img_list[idx].cols) / img_list[idx].rows;
            max_wh_ratio = std::max(max_wh_ratio, wh_ratio);
        }

        // Preprocess
        std::vector<std::vector<float>> norm_imgs;
        for (int i = beg; i < end; ++i) {
            int idx = indices[i];
            auto norm_img = cls_resize_norm_img(img_list[idx], img_c_, img_h_, img_w_);
            norm_imgs.push_back(std::move(norm_img));
        }

        // Concatenate
        size_t single_size = static_cast<size_t>(img_c_ * img_h_ * img_w_);
        std::vector<float> batch_data;
        batch_data.reserve(batch_size * single_size);
        for (const auto& img : norm_imgs) {
            batch_data.insert(batch_data.end(), img.begin(), img.end());
        }

        // Run inference
        std::vector<int64_t> input_shape = {batch_size, img_c_, img_h_, img_w_};
        auto outputs = session_->run(input_shape, batch_data);

        // Get output
        auto& output = outputs[0];
        const float* output_data = output.GetTensorData<float>();

        // Process results: argmax, get label, rotate if 180
        for (int r = 0; r < batch_size; ++r) {
            const float* row = output_data + r * 2;  // 2 classes: 0, 180
            int label_idx = (row[0] >= row[1]) ? 0 : 1;
            float score = row[label_idx];

            int orig_idx = indices[beg + r];
            cls_res[orig_idx] = {labels_[label_idx], score};

            if (labels_[label_idx] == "180" && score > cls_thresh_) {
                cv::rotate(img_list[orig_idx], img_list[orig_idx], cv::ROTATE_180);
            }
        }
    }

    return cls_res;
}

} // namespace ocr
