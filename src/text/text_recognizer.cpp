#include "text_recognizer.h"

#include "pre_process.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <cstdio>

namespace ocr {

TextRecognizer::TextRecognizer(const Config& cfg)
    : img_c_(cfg.rec_image_c), img_h_(cfg.rec_image_h), img_w_(cfg.rec_image_w),
      batch_num_(cfg.rec_batch_num), drop_score_(cfg.drop_score) {

    session_ = std::make_unique<OnnxSession>(cfg.model_path, cfg.use_gpu, cfg.gpu_id);
    postprocess_ = std::make_unique<CTCLabelDecode>(cfg.char_dict_path, cfg.use_space_char);

    // Warm-up
    std::vector<int64_t> warmup_shape = {1, img_c_, img_h_, img_w_};
    std::vector<float> dummy(img_c_ * img_h_ * img_w_, 0.0f);
    try {
        session_->run(warmup_shape, dummy);
    } catch (...) {}
}

TextRecognizer::~TextRecognizer() = default;

std::vector<TextRecognizer::Result> TextRecognizer::recognize(const std::vector<cv::Mat>& img_list) {
    int img_num = static_cast<int>(img_list.size());
    if (img_num == 0) return {};

    // Calculate aspect ratios and sort
    std::vector<float> width_list(img_num);
    for (int i = 0; i < img_num; ++i) {
        width_list[i] = static_cast<float>(img_list[i].cols) / img_list[i].rows;
    }

    // Get sort indices (ascending order of width ratio)
    std::vector<int> indices(img_num);
    for (int i = 0; i < img_num; ++i) indices[i] = i;
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return width_list[a] < width_list[b];
    });

    std::vector<Result> rec_res(img_num, {"", 0.0f});

    for (int beg = 0; beg < img_num; beg += batch_num_) {
        int end = std::min(img_num, beg + batch_num_);
        int batch_size = end - beg;

        // Find max wh_ratio in this batch
        float max_wh_ratio = static_cast<float>(img_w_) / img_h_;
        for (int i = beg; i < end; ++i) {
            int idx = indices[i];
            float wh_ratio = static_cast<float>(img_list[idx].cols) / img_list[idx].rows;
            max_wh_ratio = std::max(max_wh_ratio, wh_ratio);
        }

        // Preprocess all images in this batch
        std::vector<std::vector<float>> norm_imgs;
        for (int i = beg; i < end; ++i) {
            int idx = indices[i];
            auto norm_img = rec_resize_norm_img(img_list[idx], img_c_, img_h_, img_w_, max_wh_ratio);
            norm_imgs.push_back(std::move(norm_img));
        }

        // Determine actual width (may be larger than img_w_ due to max_wh_ratio)
        int actual_w = static_cast<int>(img_h_ * max_wh_ratio);
        if (actual_w < img_w_) actual_w = img_w_;

        // Concatenate into batch
        std::vector<float> batch_data;
        size_t single_size = static_cast<size_t>(img_c_ * img_h_ * actual_w);
        batch_data.reserve(batch_size * single_size);
        for (const auto& img : norm_imgs) {
            batch_data.insert(batch_data.end(), img.begin(), img.end());
        }

        // Run inference
        std::vector<int64_t> input_shape = {batch_size, img_c_, img_h_, actual_w};
        auto outputs = session_->run(input_shape, batch_data);

        // Get output
        auto& output = outputs[0];
        auto info = output.GetTensorTypeAndShapeInfo();
        auto shape = info.GetShape();
        const float* output_data = output.GetTensorData<float>();

        int out_batch = static_cast<int>(shape[0]);
        int out_seq = static_cast<int>(shape[1]);
        int out_classes = static_cast<int>(shape[2]);

        // CTC decode
        auto results = postprocess_->decode(output_data, out_batch, out_seq, out_classes);

        // Map back to original order
        for (int r = 0; r < static_cast<int>(results.size()) && beg + r < img_num; ++r) {
            int orig_idx = indices[beg + r];
            rec_res[orig_idx].text = results[r].text;
            rec_res[orig_idx].score = results[r].score;
        }
    }

    return rec_res;
}

} // namespace ocr