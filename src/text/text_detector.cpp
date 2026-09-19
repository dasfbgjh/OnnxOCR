#include "text_detector.h"

#include <stdexcept>
#include <cmath>
#include <iostream>

namespace ocr {

TextDetector::TextDetector(const Config& cfg) {
    session_ = std::make_unique<OnnxSession>(cfg.model_path, cfg.use_gpu, cfg.gpu_id);

    pre_cfg_.limit_side_len = cfg.det_limit_side_len;
    pre_cfg_.limit_type = cfg.det_limit_type;

    post_cfg_.thresh = cfg.det_db_thresh;
    post_cfg_.box_thresh = cfg.det_db_box_thresh;
    post_cfg_.unclip_ratio = cfg.det_db_unclip_ratio;
    post_cfg_.use_dilation = cfg.use_dilation;
    post_cfg_.score_mode = cfg.det_db_score_mode;
    post_cfg_.box_type = cfg.det_box_type;
    post_cfg_.max_candidates = 1000;

    // Warm-up: run a dummy inference to initialize ONNX Runtime kernels
    auto input_shape = session_->get_input_shape();
    // Make dynamic dimensions concrete for warm-up
    for (auto& d : input_shape) {
        if (d < 1) d = 1;
    }
    if (input_shape.size() >= 4) {
        input_shape[2] = static_cast<int64_t>(cfg.det_limit_side_len);
        input_shape[3] = static_cast<int64_t>(cfg.det_limit_side_len);
    }
    std::vector<float> dummy(input_shape[1] * input_shape[2] * input_shape[3], 0.0f);
    try {
        session_->run(input_shape, dummy);
    } catch (...) {
        // Warm-up failure is non-fatal
    }
}

TextDetector::~TextDetector() = default;

std::vector<TextBox> TextDetector::detect(const cv::Mat& img) {
    // 1. Preprocess: resize + normalize + HWC->CHW
    auto pre = det_preprocess(img, pre_cfg_);

    // Build CHW float input
    int h = pre.image.rows;
    int w = pre.image.cols;
    int c = 3;
    std::vector<float> input_data(c * h * w);

    // Convert HWC to CHW
    std::vector<cv::Mat> channels;
    cv::split(pre.image, channels);
    for (int ch = 0; ch < c && ch < static_cast<int>(channels.size()); ++ch) {
        std::memcpy(input_data.data() + ch * h * w, channels[ch].data, h * w * sizeof(float));
    }

    // 2. Run inference
    std::vector<int64_t> input_shape = {1, c, h, w};
    auto outputs = session_->run(input_shape, input_data);

    // 3. Get output (segmentation map)
    auto& output = outputs[0];
    auto info = output.GetTensorTypeAndShapeInfo();
    auto shape = info.GetShape();
    const float* output_data = output.GetTensorData<float>();

    int out_h = static_cast<int>(shape[2]);
    int out_w = static_cast<int>(shape[3]);
    cv::Mat pred(out_h, out_w, CV_32F, const_cast<float*>(output_data));

    // 4. Post-process: DB
    std::vector<ShapeListEntry> shape_list = {
        {pre.src_h, pre.src_w, pre.ratio_h, pre.ratio_w}
    };

    auto boxes = db_post_process_batch(pred, shape_list, post_cfg_);

    // 5. Filter
    int img_h = img.rows;
    int img_w = img.cols;
    if (post_cfg_.box_type == "poly") {
        boxes = filter_tag_det_res_only_clip(boxes, img_h, img_w);
    } else {
        boxes = filter_tag_det_res(boxes, img_h, img_w);
    }

    return boxes;
}

} // namespace ocr