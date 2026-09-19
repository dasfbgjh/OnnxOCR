#include "doc_layout_analyzer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

namespace ocr {

constexpr int DocLayoutAnalyzer::IMG_SIZE;

DocLayoutAnalyzer::DocLayoutAnalyzer(const Config& cfg)
    : conf_thresh_(cfg.conf_thresh),
      iou_thresh_(cfg.iou_thresh) {

    session_ = std::make_unique<OnnxSession>(cfg.model_path, cfg.use_gpu, cfg.gpu_id);

    /* Load labels from model metadata ("character" key, newline-separated). */
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

    /* Fallback: pp_doclayoutv2 labels (25 classes) */
    if (labels_.empty()) {
        labels_ = {
            "abstract", "algorithm", "aside_text", "chart", "content",
            "formula", "doc_title", "figure_title", "footer", "footer",
            "footnote", "formula_number", "header", "header", "image",
            "inline_formula", "number", "paragraph_title", "reference",
            "reference_content", "seal", "table", "text", "vertical_text",
            "vision_footnote"
        };
    }

    /* Warm-up: 3 inputs (im_shape, image, scale_factor) for pp_doclayoutv2. */
    try {
        std::vector<int64_t> warmup_shape = {1, 3, IMG_SIZE, IMG_SIZE};
        std::vector<float> dummy(3 * IMG_SIZE * IMG_SIZE, 0.0f);
        std::vector<int64_t> im_shape_sz = {1, 2};
        std::vector<float> im_shape_data = {
            static_cast<float>(IMG_SIZE), static_cast<float>(IMG_SIZE)};
        std::vector<int64_t> sf_shape = {1, 2};
        std::vector<float> sf_data = {1.0f, 1.0f};
        session_->run_multi(
            {im_shape_sz, warmup_shape, sf_shape},
            {im_shape_data, dummy, sf_data});
    } catch (...) {}
}

DocLayoutAnalyzer::~DocLayoutAnalyzer() = default;

float DocLayoutAnalyzer::iou_of(const float box1[4], const float box2[4]) {
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

std::vector<int> DocLayoutAnalyzer::hard_nms(
    const std::vector<std::vector<float>>& box_scores,
    float iou_thresh) {

    int n = static_cast<int>(box_scores.size());
    if (n == 0) return {};

    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    /* Sort by score ascending (like np.argsort), so highest score is at the end */
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return box_scores[a][4] < box_scores[b][4];
    });

    std::vector<int> picked;
    while (!indices.empty()) {
        int current = indices.back();
        picked.push_back(current);
        if (indices.size() == 1) break;

        indices.pop_back();
        float current_box[4] = {box_scores[current][0], box_scores[current][1],
                                  box_scores[current][2], box_scores[current][3]};

        std::vector<int> remaining;
        for (int idx : indices) {
            float other_box[4] = {box_scores[idx][0], box_scores[idx][1],
                                    box_scores[idx][2], box_scores[idx][3]};
            if (iou_of(current_box, other_box) <= iou_thresh) {
                remaining.push_back(idx);
            }
        }
        indices = std::move(remaining);
    }
    return picked;
}

cv::Mat DocLayoutAnalyzer::letterbox(const cv::Mat& img, float& ratio,
                                      int& pad_left, int& pad_top) {
    int h = img.rows, w = img.cols;
    ratio = std::min(static_cast<float>(IMG_SIZE) / h,
                     static_cast<float>(IMG_SIZE) / w);
    /* Only scale down, not up (matches Python scaleup=False) */
    ratio = std::min(ratio, 1.0f);

    int new_w = static_cast<int>(std::round(w * ratio));
    int new_h = static_cast<int>(std::round(h * ratio));

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    int dw = IMG_SIZE - new_w;
    int dh = IMG_SIZE - new_h;
    /* Top-left aligned padding (PaddlePaddle convention):
       padding goes to bottom-right only, image starts at (0,0) */
    pad_top = 0;
    pad_left = 0;
    int pad_bottom = dh;
    int pad_right = dw;

    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, pad_top, pad_bottom, pad_left, pad_right,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    return padded;
}

void DocLayoutAnalyzer::scale_boxes(std::vector<std::vector<float>>& boxes,
                                     float ratio, int pad_left, int pad_top,
                                     int ori_h, int ori_w) {
    for (auto& box : boxes) {
        /* Remove padding */
        box[0] -= pad_left;  /* x1 */
        box[1] -= pad_top;   /* y1 */
        box[2] -= pad_left;  /* x2 */
        box[3] -= pad_top;   /* y2 */
        /* Scale back */
        box[0] /= ratio;
        box[1] /= ratio;
        box[2] /= ratio;
        box[3] /= ratio;
        /* Clip to original image bounds */
        box[0] = std::max(0.0f, std::min(box[0], static_cast<float>(ori_w)));
        box[1] = std::max(0.0f, std::min(box[1], static_cast<float>(ori_h)));
        box[2] = std::max(0.0f, std::min(box[2], static_cast<float>(ori_w)));
        box[3] = std::max(0.0f, std::min(box[3], static_cast<float>(ori_h)));
    }
}

std::vector<DocLayoutResult> DocLayoutAnalyzer::analyze(const cv::Mat& img) {
    std::vector<DocLayoutResult> results;
    if (img.empty()) return results;

    int ori_h = img.rows;
    int ori_w = img.cols;

    /* Step 1: Letterbox preprocessing */
    float ratio;
    int pad_left, pad_top;
    cv::Mat padded = letterbox(img, ratio, pad_left, pad_top);

    /* Normalize: /255, no mean/std for pp_doclayoutv2 */
    padded.convertTo(padded, CV_32F, 1.0 / 255.0);

    /* HWC -> CHW */
    std::vector<float> input_data(3 * IMG_SIZE * IMG_SIZE);
    std::vector<cv::Mat> channels(3);
    cv::split(padded, channels);
    for (int c = 0; c < 3; ++c) {
        std::memcpy(input_data.data() + c * IMG_SIZE * IMG_SIZE,
                    channels[c].data, IMG_SIZE * IMG_SIZE * sizeof(float));
    }

    /* scale_factor: [h_scale, w_scale] (original -> resized).
       PaddlePaddle convention: scale_factor = [new_h/ori_h, new_w/ori_w]
       i.e. the actual resize ratio, NOT including padding. */
    int new_h = static_cast<int>(std::round(ori_h * ratio));
    int new_w = static_cast<int>(std::round(ori_w * ratio));
    float h_scale = static_cast<float>(new_h) / ori_h;
    float w_scale = static_cast<float>(new_w) / ori_w;

    /* Step 2: Run inference with 3 inputs: im_shape, image, scale_factor */
    std::vector<int64_t> im_shape_sz = {1, 2};
    std::vector<float> im_shape_data = {
        static_cast<float>(IMG_SIZE), static_cast<float>(IMG_SIZE)};
    std::vector<int64_t> img_shape = {1, 3, IMG_SIZE, IMG_SIZE};
    std::vector<int64_t> sf_shape = {1, 2};
    std::vector<float> sf_data = {h_scale, w_scale};

    auto outputs = session_->run_multi(
        {im_shape_sz, img_shape, sf_shape},
        {im_shape_data, input_data, sf_data});

    if (outputs.size() < 2) return results;

    /* Output 0: boxes [total_boxes, 8] =
       [class_id, score, x1, y1, x2, y2, reading_order, reading_order_dup]
       Output 1: box_nums [batch_size] = number of boxes per image (int32) */
    auto& boxes_out = outputs[0];
    auto& box_nums_out = outputs[1];

    auto boxes_info = boxes_out.GetTensorTypeAndShapeInfo();
    auto boxes_shape = boxes_info.GetShape();
    const float* boxes_data = boxes_out.GetTensorData<float>();

    auto nums_info = box_nums_out.GetTensorTypeAndShapeInfo();
    auto nums_shape = nums_info.GetShape();
    /* pp_doclayoutv2 uses int32 for box count (not int64) */
    const int32_t* nums_data_i32 =
        box_nums_out.GetTensorData<int32_t>();
    const int64_t* nums_data_i64 =
        box_nums_out.GetTensorData<int64_t>();

    int num_boxes = 0;
    if (nums_shape.empty() || nums_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        num_boxes = static_cast<int>(nums_data_i32[0]);
    } else {
        num_boxes = static_cast<int>(nums_data_i64[0]);
    }
    /* Fallback: derive from boxes_shape[0] when count looks invalid */
    if (num_boxes <= 0 && boxes_shape.size() >= 1) {
        num_boxes = (boxes_shape[0] > 0) ? static_cast<int>(boxes_shape[0]) : 0;
    }

    int box_attr = 8;
    if (boxes_shape.size() >= 2 && boxes_shape[1] > 0) {
        box_attr = static_cast<int>(boxes_shape[1]);
    }

    /* Step 3: Post-process: filter by confidence, NMS, scale boxes */
    /* Each entry: [x1, y1, x2, y2, score, class_id, reading_order] */
    std::vector<std::vector<float>> valid_boxes;

    for (int i = 0; i < num_boxes; ++i) {
        const float* box = boxes_data + i * box_attr;
        float class_id = box[0];
        float score = box[1];
        float x1 = box[2], y1 = box[3], x2 = box[4], y2 = box[5];
        float reading_order = (box_attr >= 7) ? box[6] : 0.0f;

        if (score < conf_thresh_) continue;

        valid_boxes.push_back({x1, y1, x2, y2, score, class_id, reading_order});
    }

    if (valid_boxes.empty()) return results;

    /* NMS (treating all classes together with same-class threshold) */
    std::vector<std::vector<float>> nms_input;
    for (const auto& vb : valid_boxes) {
        nms_input.push_back({vb[0], vb[1], vb[2], vb[3], vb[4]});
    }
    auto keep = hard_nms(nms_input, iou_thresh_);

    /* Scale boxes back to original image coordinates */
    std::vector<std::vector<float>> kept_boxes;
    for (int idx : keep) {
        kept_boxes.push_back({
            valid_boxes[idx][0], valid_boxes[idx][1],
            valid_boxes[idx][2], valid_boxes[idx][3],
            valid_boxes[idx][4],  /* score */
            valid_boxes[idx][5],  /* class_id */
            valid_boxes[idx][6]   /* reading_order */
        });
    }

    /* Build results */
    for (int i = 0; i < static_cast<int>(kept_boxes.size()); ++i) {
        const auto& kb = kept_boxes[i];
        DocLayoutResult r;
        r.box[0] = kb[0];
        r.box[1] = kb[1];
        r.box[2] = kb[2];
        r.box[3] = kb[3];
        r.score = kb[4];
        r.class_id = static_cast<int>(kb[5]);
        r.class_name = (r.class_id < static_cast<int>(labels_.size()))
                       ? labels_[r.class_id]
                       : ("class_" + std::to_string(r.class_id));
        /* Prefer model-provided reading order when available */
        r.order = static_cast<int>(kb[6]);
        results.push_back(std::move(r));
    }

    /* Sort by the model's reading order (pp_doclayoutv2 outputs an explicit
       reading order; fall back to top-to-bottom / left-to-right if missing. */
    std::sort(results.begin(), results.end(), [](const DocLayoutResult& a, const DocLayoutResult& b) {
        if (a.order != b.order) return a.order < b.order;
        float ay = (a.box[1] + a.box[3]) / 2.0f;
        float by = (b.box[1] + b.box[3]) / 2.0f;
        if (std::abs(ay - by) > 10.0f) return ay < by;
        return a.box[0] < b.box[0];
    });

    /* Re-assign sequential order after sorting */
    for (int i = 0; i < static_cast<int>(results.size()); ++i) {
        results[i].order = i;
    }

    return results;
}

} // namespace ocr