#ifndef PRE_PROCESS_H
#define PRE_PROCESS_H

#include <opencv2/opencv.hpp>
#include <vector>

namespace ocr {

/* Configuration for detection preprocessing. */
struct DetPreProcessConfig {
    float limit_side_len = 960.0f;
    std::string limit_type = "max";  // "max" or "min"
};

/* Detection preprocessing result. */
struct DetPreResult {
    cv::Mat image;          // preprocessed image, float32 CHW
    float src_h, src_w;     // original height, width
    float ratio_h, ratio_w; // resize ratios
};

/* Detection preprocessing: resize (multiple of 32) + normalize + HWC->CHW.
   Returns a float vector (CHW) and the shape info. */
DetPreResult det_preprocess(const cv::Mat& img, const DetPreProcessConfig& cfg);

/* Normalize an image with ImageNet mean/std: (img/255 - mean) / std
   Returns a float32 CHW tensor as a flat vector. */
std::vector<float> normalize_image_chw(const cv::Mat& bgr_img,
                                        float scale = 1.0f / 255.0f,
                                        const std::vector<float>& mean = {0.485f, 0.456f, 0.406f},
                                        const std::vector<float>& std = {0.229f, 0.224f, 0.225f});

/* Recognition preprocessing: resize maintaining aspect ratio, normalize, pad to fixed width.
   max_wh_ratio: maximum width/height ratio allowed.
   imgC, imgH, imgW: target shape.
   Returns a float vector (C, H, W). */
std::vector<float> rec_resize_norm_img(const cv::Mat& img,
                                        int imgC, int imgH, int imgW,
                                        float max_wh_ratio);

/* Classification preprocessing: resize maintaining aspect ratio, normalize, pad.
   Returns a float vector (C, H, W). */
std::vector<float> cls_resize_norm_img(const cv::Mat& img,
                                        int imgC, int imgH, int imgW);

} // namespace ocr

#endif /* PRE_PROCESS_H */
