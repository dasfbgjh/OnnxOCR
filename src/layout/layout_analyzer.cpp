#include "layout_analyzer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

namespace ocr {

constexpr int LayoutAnalyzer::STRIDES[4];

LayoutAnalyzer::LayoutAnalyzer(const Config& cfg)
    : conf_thresh_(cfg.conf_thresh),
      iou_thresh_(cfg.iou_thresh),
      img_h_(800),
      img_w_(608) {

    session_ = std::make_unique<OnnxSession>(cfg.model_path, cfg.use_gpu, cfg.gpu_id);

    /* Load labels from model metadata ("character" key, newline-separated). */
    auto meta = session_->get_metadata();
    auto it = meta.find("character");
    if (it != meta.end()) {
        /* Split by newlines */
        std::string chars = it->second;
        size_t start = 0;
        while (start < chars.size()) {
            size_t end = chars.find('\n', start);
            if (end == std::string::npos) end = chars.size();
            labels_.push_back(chars.substr(start, end - start));
            start = end + 1;
        }
    }

    /* Fallback: use default labels based on model type */
    if (labels_.empty()) {
        if (cfg.model_type == LayoutModelType::PP_LAYOUT_PUBLAYNET) {
            labels_ = {"text", "title", "list", "table", "figure"};
        } else {
            /* CDLA labels */
            labels_ = {"title", "text", "figure", "figure_caption",
                       "table", "table_caption", "header", "footer",
                       "reference", "equation"};
        }
    }

    /* Warm-up */
    try {
        std::vector<int64_t> warmup_shape = {1, 3, img_h_, img_w_};
        std::vector<float> dummy(3 * img_h_ * img_w_, 0.0f);
        session_->run(warmup_shape, dummy);
    } catch (...) {}
}

LayoutAnalyzer::~LayoutAnalyzer() = default;

void LayoutAnalyzer::softmax(std::vector<float>& x, int rows, int cols) {
    /* Softmax along columns (axis=1) */
    for (int r = 0; r < rows; ++r) {
        float* row = x.data() + r * cols;
        float max_val = row[0];
        for (int c = 1; c < cols; ++c) {
            if (row[c] > max_val) max_val = row[c];
        }
        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) {
            row[c] = std::exp(row[c] - max_val);
            sum += row[c];
        }
        for (int c = 0; c < cols; ++c) {
            row[c] /= sum;
        }
    }
}

float LayoutAnalyzer::iou_of(const float box1[4], const float box2[4]) {
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

std::vector<int> LayoutAnalyzer::hard_nms(
    const std::vector<std::vector<float>>& box_scores,
    float iou_thresh, int top_k) {
    /* box_scores[i] = [x1, y1, x2, y2, score] */
    int n = static_cast<int>(box_scores.size());
    if (n == 0) return {};

    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    /* Sort by score ascending (like np.argsort), so highest score is at the end */
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return box_scores[a][4] < box_scores[b][4];
    });

    /* Only consider top candidates (highest scores at the end) */
    int candidate_size = std::min(n, 200);
    if (n > candidate_size) {
        indices.erase(indices.begin(), indices.begin() + (n - candidate_size));
    }

    std::vector<int> picked;
    while (!indices.empty()) {
        int current = indices.back();
        picked.push_back(current);
        if (top_k > 0 && static_cast<int>(picked.size()) >= top_k) break;
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

void LayoutAnalyzer::clip_boxes(std::vector<std::vector<float>>& boxes,
                                 int img_h, int img_w) {
    for (auto& box : boxes) {
        box[0] = std::max(0.0f, std::min(box[0], static_cast<float>(img_w)));
        box[2] = std::max(0.0f, std::min(box[2], static_cast<float>(img_w)));
        box[1] = std::max(0.0f, std::min(box[1], static_cast<float>(img_h)));
        box[3] = std::max(0.0f, std::min(box[3], static_cast<float>(img_h)));
    }
}

std::vector<LayoutResult> LayoutAnalyzer::analyze(const cv::Mat& img) {
    std::vector<LayoutResult> results;
    if (img.empty()) return results;

    int ori_h = img.rows;
    int ori_w = img.cols;

    /* Step 1: Preprocess - resize to (img_h_, img_w_), normalize with ImageNet mean/std */
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(img_w_, img_h_));
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);

    cv::Scalar mean_s(0.485, 0.456, 0.406);
    cv::Scalar std_s(0.229, 0.224, 0.225);
    resized = (resized - mean_s) / std_s;

    /* HWC -> CHW */
    std::vector<float> input_data(3 * img_h_ * img_w_);
    std::vector<cv::Mat> channels(3);
    cv::split(resized, channels);
    for (int c = 0; c < 3; ++c) {
        std::memcpy(input_data.data() + c * img_h_ * img_w_,
                    channels[c].data, img_h_ * img_w_ * sizeof(float));
    }

    /* Step 2: Run inference */
    std::vector<int64_t> input_shape = {1, 3, img_h_, img_w_};
    auto outputs = session_->run(input_shape, input_data);

    /* Step 3: Post-processing
       The PP model outputs 8 tensors: 4 score maps + 4 box distribution maps.
       num_outs = len(outputs) / 2
       scores[i] = outputs[i]
       raw_boxes[i] = outputs[i + num_outs] */
    int num_outs = static_cast<int>(outputs.size()) / 2;
    if (num_outs != 4) {
        /* Unexpected number of outputs */
        return results;
    }

    /* reg_max = raw_boxes[0].shape[-1] / 4 - 1 */
    auto& first_box = outputs[num_outs];  /* first box distribution output */
    auto first_box_info = first_box.GetTensorTypeAndShapeInfo();
    auto first_box_shape = first_box_info.GetShape();
    int reg_max = static_cast<int>(first_box_shape[first_box_shape.size() - 1] / 4 - 1);

    /* Compute scale factors */
    float scale_y = static_cast<float>(img_h_) / ori_h;
    float scale_x = static_cast<float>(img_w_) / ori_w;

    /* Collect decoded boxes and scores from all strides */
    std::vector<std::vector<float>> all_boxes;  /* [x1,y1,x2,y2] */
    std::vector<std::vector<float>> all_scores; /* [class0, class1, ...] per box */
    int num_classes = 0;

    for (int s = 0; s < 4; ++s) {
        int stride = STRIDES[s];

        /* Score output: [1, num_anchors, num_classes] or [1, num_classes, h*w] */
        auto& score_out = outputs[s];
        auto score_info = score_out.GetTensorTypeAndShapeInfo();
        auto score_shape = score_info.GetShape();
        const float* score_data = score_out.GetTensorData<float>();

        /* Box output: [1, num_anchors, 4*(reg_max+1)] */
        auto& box_out = outputs[s + num_outs];
        auto box_info = box_out.GetTensorTypeAndShapeInfo();
        auto box_shape = box_out.GetTensorTypeAndShapeInfo().GetShape();
        const float* box_data = box_out.GetTensorData<float>();

        /* Determine feature map dimensions from stride (ceiling division) */
        int fm_h = (img_h_ + stride - 1) / stride;
        int fm_w = (img_w_ + stride - 1) / stride;
        int num_anchors = fm_h * fm_w;

        /* Determine layout: [1, num_anchors, num_classes] or [1, num_classes, num_anchors] */
        if (score_shape.size() == 3) {
            if (score_shape[1] == num_anchors) {
                /* [1, num_anchors, num_classes] */
                num_classes = static_cast<int>(score_shape[2]);
            } else if (score_shape[2] == num_anchors) {
                /* [1, num_classes, num_anchors] - need to handle transposition */
                num_classes = static_cast<int>(score_shape[1]);
            } else {
                /* Fallback: assume [1, num_anchors, num_classes] */
                num_classes = static_cast<int>(score_shape[2]);
            }
        }

        /* Decode box distribution: softmax over (reg_max+1) for each of 4 directions */
        int box_attr = 4 * (reg_max + 1);

        /* For each anchor position */
        for (int pos = 0; pos < num_anchors; ++pos) {
            /* Center coordinates */
            int row = pos / fm_w;
            int col = pos % fm_w;
            float ct_row = (row + 0.5f) * stride;
            float ct_col = (col + 0.5f) * stride;

            /* Decode box distribution for this position */
            const float* box_dist = box_data + pos * box_attr;

            /* For each of 4 directions (left, top, right, bottom) */
            float dist[4];
            for (int d = 0; d < 4; ++d) {
                const float* dist_ptr = box_dist + d * (reg_max + 1);
                /* Softmax */
                float max_val = dist_ptr[0];
                for (int r = 1; r <= reg_max; ++r) {
                    if (dist_ptr[r] > max_val) max_val = dist_ptr[r];
                }
                float sum_exp = 0.0f;
                std::vector<float> exp_vals(reg_max + 1);
                for (int r = 0; r <= reg_max; ++r) {
                    exp_vals[r] = std::exp(dist_ptr[r] - max_val);
                    sum_exp += exp_vals[r];
                }
                /* Expected value: sum(r * softmax[r]) */
                float expected = 0.0f;
                for (int r = 0; r <= reg_max; ++r) {
                    expected += r * (exp_vals[r] / sum_exp);
                }
                dist[d] = expected * stride;
            }

            /* Decode box: center + [-1, -1, 1, 1] * distance */
            /* x1 = ct_col - dist[0], y1 = ct_row - dist[1]
               x2 = ct_col + dist[2], y2 = ct_row + dist[3] */
            std::vector<float> box = {
                ct_col - dist[0],
                ct_row - dist[1],
                ct_col + dist[2],
                ct_row + dist[3]
            };
            all_boxes.push_back(std::move(box));

            /* Get scores for all classes at this position */
            std::vector<float> scores(num_classes);
            if (score_shape.size() == 3 && score_shape[2] == num_classes) {
                /* [1, num_anchors, num_classes] */
                const float* s = score_data + pos * num_classes;
                std::memcpy(scores.data(), s, num_classes * sizeof(float));
            } else if (score_shape.size() == 3 && score_shape[1] == num_classes) {
                /* [1, num_classes, num_anchors] - transposed */
                for (int c = 0; c < num_classes; ++c) {
                    scores[c] = score_data[c * num_anchors + pos];
                }
            } else {
                /* Fallback */
                const float* s = score_data + pos * num_classes;
                std::memcpy(scores.data(), s, num_classes * sizeof(float));
            }
            all_scores.push_back(std::move(scores));
        }
    }

    /* Step 4: Per-class filtering + NMS */
    for (int class_idx = 0; class_idx < num_classes; ++class_idx) {
        std::vector<std::vector<float>> class_box_scores;  /* [x1,y1,x2,y2,score] */

        for (size_t i = 0; i < all_boxes.size(); ++i) {
            float score = all_scores[i][class_idx];
            if (score <= conf_thresh_) continue;
            std::vector<float> bs = all_boxes[i];
            bs.push_back(score);
            class_box_scores.push_back(std::move(bs));
        }

        if (class_box_scores.empty()) continue;

        /* NMS */
        auto keep = hard_nms(class_box_scores, iou_thresh_, KEEP_TOP_K);

        for (int idx : keep) {
            /* Rescale box to original image coordinates */
            float x1 = (class_box_scores[idx][0] / scale_x);
            float y1 = (class_box_scores[idx][1] / scale_y);
            float x2 = (class_box_scores[idx][2] / scale_x);
            float y2 = (class_box_scores[idx][3] / scale_y);

            /* Clip to original image bounds */
            x1 = std::max(0.0f, std::min(x1, static_cast<float>(ori_w)));
            y1 = std::max(0.0f, std::min(y1, static_cast<float>(ori_h)));
            x2 = std::max(0.0f, std::min(x2, static_cast<float>(ori_w)));
            y2 = std::max(0.0f, std::min(y2, static_cast<float>(ori_h)));

            LayoutResult r;
            r.box[0] = x1;
            r.box[1] = y1;
            r.box[2] = x2;
            r.box[3] = y2;
            r.score = class_box_scores[idx][4];
            r.class_id = class_idx;
            r.class_name = (class_idx < static_cast<int>(labels_.size()))
                           ? labels_[class_idx]
                           : ("class_" + std::to_string(class_idx));
            results.push_back(std::move(r));
        }
    }

    /* Sort results: by class_id then by score descending */
    std::sort(results.begin(), results.end(), [](const LayoutResult& a, const LayoutResult& b) {
        if (a.class_id != b.class_id) return a.class_id < b.class_id;
        return a.score > b.score;
    });

    return results;
}

} // namespace ocr