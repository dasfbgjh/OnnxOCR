#ifndef DB_POST_PROCESS_H
#define DB_POST_PROCESS_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "utils.h"

namespace ocr {

/* Configuration for DB (Differentiable Binarization) post-processing. */
struct DBPostProcessConfig {
    float thresh = 0.3f;
    float box_thresh = 0.6f;
    int   max_candidates = 1000;
    float unclip_ratio = 1.5f;
    bool  use_dilation = false;
    std::string score_mode = "fast";  // "fast" or "slow"
    std::string box_type = "quad";     // "quad" or "poly"
};

/* Shape list entry: [src_h, src_w, ratio_h, ratio_w] */
struct ShapeListEntry {
    float src_h, src_w, ratio_h, ratio_w;
};

/* Run DB post-processing on a segmentation map.
   pred: segmentation map [H, W] (float32, after slicing batch & channel)
   bitmap: binary mask [H, W] (float32 or uint8, values 0 or 1)
   dest_width, dest_height: original image dimensions
   Returns detected boxes. */
std::vector<TextBox> db_post_process(const cv::Mat& pred,
                                      const cv::Mat& bitmap,
                                      float dest_width, float dest_height,
                                      const DBPostProcessConfig& cfg);

/* The full DB post-process pipeline (handles batch & dilation). */
std::vector<TextBox> db_post_process_batch(const cv::Mat& pred_batch,
                                            const std::vector<ShapeListEntry>& shape_list,
                                            const DBPostProcessConfig& cfg);

} // namespace ocr

#endif /* DB_POST_PROCESS_H */
