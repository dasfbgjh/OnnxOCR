#include "pre_process.h"

#include <cmath>
#include <algorithm>
#include <iostream>

namespace ocr {

DetPreResult det_preprocess(const cv::Mat& img, const DetPreProcessConfig& cfg) {
    DetPreResult result;
    result.src_h = static_cast<float>(img.rows);
    result.src_w = static_cast<float>(img.cols);

    float h = result.src_h;
    float w = result.src_w;
    float ratio = 1.0f;

    if (cfg.limit_type == "max") {
        if (std::max(h, w) > cfg.limit_side_len) {
            ratio = (h > w) ? (cfg.limit_side_len / h) : (cfg.limit_side_len / w);
        }
    } else if (cfg.limit_type == "min") {
        if (std::min(h, w) < cfg.limit_side_len) {
            ratio = (h < w) ? (cfg.limit_side_len / h) : (cfg.limit_side_len / w);
        }
    } else {
        ratio = cfg.limit_side_len / std::max(h, w);
    }

    int resize_h = static_cast<int>(h * ratio);
    int resize_w = static_cast<int>(w * ratio);
    resize_h = std::max(static_cast<int>(std::round(resize_h / 32.0) * 32), 32);
    resize_w = std::max(static_cast<int>(std::round(resize_w / 32.0) * 32), 32);

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(resize_w, resize_h));
    result.ratio_h = resize_h / h;
    result.ratio_w = resize_w / w;

    // Normalize: (img * scale - mean) / std, HWC -> CHW
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);
    cv::Scalar mean_s(0.485, 0.456, 0.406);
    cv::Scalar std_s(0.229, 0.224, 0.225);
    resized = (resized - mean_s) / std_s;

    // HWC -> CHW
    std::vector<cv::Mat> channels(3);
    cv::split(resized, channels);
    // Interleave: CHW
    result.image = cv::Mat(resize_h, resize_w, CV_32FC3);
    // Actually, we need a flat CHW float vector, but we'll store as Mat for now
    // and handle CHW conversion when creating the input tensor.

    // For the ONNX input, we need CHW float. Let's store the resized+normalized image
    // and do HWC->CHW conversion in the caller.
    resized.copyTo(result.image);
    return result;
}

std::vector<float> normalize_image_chw(const cv::Mat& bgr_img,
                                        float scale,
                                        const std::vector<float>& mean,
                                        const std::vector<float>& std) {
    cv::Mat float_img;
    bgr_img.convertTo(float_img, CV_32F, scale);

    cv::Scalar mean_s(mean[0], mean[1], mean[2]);
    cv::Scalar std_s(std[0], std[1], std[2]);
    float_img = (float_img - mean_s) / std_s;

    // HWC -> CHW
    std::vector<float> chw(static_cast<size_t>(float_img.total() * 3));
    std::vector<cv::Mat> channels(3);
    cv::split(float_img, channels);
    size_t total = static_cast<size_t>(float_img.total());
    for (int c = 0; c < 3; ++c) {
        std::memcpy(chw.data() + c * total, channels[c].data, total * sizeof(float));
    }
    return chw;
}

std::vector<float> rec_resize_norm_img(const cv::Mat& img,
                                        int imgC, int imgH, int imgW,
                                        float max_wh_ratio) {
    int resized_w = 0;
    float h = static_cast<float>(img.rows);
    float w = static_cast<float>(img.cols);
    float ratio = w / h;

    int actual_imgW = static_cast<int>(imgH * max_wh_ratio);
    if (actual_imgW < imgW) actual_imgW = imgW;

    if (std::ceil(imgH * ratio) > actual_imgW) {
        resized_w = actual_imgW;
    } else {
        resized_w = static_cast<int>(std::ceil(imgH * ratio));
    }

    if (resized_w > actual_imgW) {
        resized_w = actual_imgW;
    }

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(resized_w, imgH));

    resized.convertTo(resized, CV_32F, 1.0 / 255.0);
    resized -= cv::Scalar(0.5, 0.5, 0.5);
    resized /= 0.5;

    // HWC -> CHW, pad to actual_imgW
    std::vector<float> chw(static_cast<size_t>(imgC * imgH * actual_imgW), 0.0f);
    std::vector<cv::Mat> channels(3);
    cv::split(resized, channels);
    for (int c = 0; c < imgC && c < 3; ++c) {
        for (int row = 0; row < imgH; ++row) {
            const float* src = channels[c].ptr<float>(row);
            float* dst = chw.data() + (c * imgH * actual_imgW) + (row * actual_imgW);
            std::memcpy(dst, src, resized_w * sizeof(float));
        }
    }
    return chw;
}

std::vector<float> cls_resize_norm_img(const cv::Mat& img,
                                        int imgC, int imgH, int imgW) {
    float h = static_cast<float>(img.rows);
    float w = static_cast<float>(img.cols);
    float ratio = w / h;

    int resized_w;
    if (std::ceil(imgH * ratio) > imgW) {
        resized_w = imgW;
    } else {
        resized_w = static_cast<int>(std::ceil(imgH * ratio));
    }

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(resized_w, imgH));

    resized.convertTo(resized, CV_32F, 1.0 / 255.0);
    resized -= cv::Scalar(0.5, 0.5, 0.5);
    resized /= 0.5;

    // HWC -> CHW, pad to imgW
    std::vector<float> chw(static_cast<size_t>(imgC * imgH * imgW), 0.0f);
    std::vector<cv::Mat> channels(3);
    cv::split(resized, channels);
    for (int c = 0; c < imgC && c < 3; ++c) {
        for (int row = 0; row < imgH; ++row) {
            const float* src = channels[c].ptr<float>(row);
            float* dst = chw.data() + (c * imgH * imgW) + (row * imgW);
            std::memcpy(dst, src, resized_w * sizeof(float));
        }
    }
    return chw;
}

} // namespace ocr