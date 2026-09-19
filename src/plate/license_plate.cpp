#include "license_plate.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

namespace ocr {

static std::vector<std::string> split_utf8_chars(const std::string& s) {
    std::vector<std::string> chars;
    for (size_t i = 0; i < s.size(); ) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        int len = 1;
        if (c >= 0xF0) len = 4;
        else if (c >= 0xE0) len = 3;
        else if (c >= 0xC0) len = 2;
        chars.push_back(s.substr(i, len));
        i += len;
    }
    return chars;
}

const std::vector<std::string> LicensePlateRecognizer::PLATE_CHARS = split_utf8_chars(
    "#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新"
    "学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品"
);

static const float PLATE_MEAN = 0.588f;
static const float PLATE_STD = 0.193f;

LicensePlateRecognizer::LicensePlateRecognizer(const Config& cfg)
    : min_score_(cfg.min_score), iou_thresh_(cfg.iou_thresh) {

    det_session_ = std::make_unique<OnnxSession>(cfg.detect_model_path, cfg.use_gpu, cfg.gpu_id);
    rec_session_ = std::make_unique<OnnxSession>(cfg.rec_model_path, cfg.use_gpu, cfg.gpu_id);

    // Warm-up detection model
    try {
        std::vector<int64_t> warmup_shape = {1, 3, 640, 640};
        std::vector<float> dummy(3 * 640 * 640, 0.0f);
        det_session_->run(warmup_shape, dummy);
    } catch (...) {}

    // Warm-up recognition model
    try {
        std::vector<int64_t> warmup_shape = {1, 3, 48, 168};
        std::vector<float> dummy(3 * 48 * 168, 0.0f);
        rec_session_->run(warmup_shape, dummy);
    } catch (...) {}
}

LicensePlateRecognizer::~LicensePlateRecognizer() = default;

LicensePlateRecognizer::LetterboxResult LicensePlateRecognizer::letterbox(
    const cv::Mat& img, int target_h, int target_w) {
    LetterboxResult result;
    int h = img.rows;
    int w = img.cols;
    float ratio = std::min(static_cast<float>(target_h) / h,
                           static_cast<float>(target_w) / w);
    int new_h = static_cast<int>(h * ratio);
    int new_w = static_cast<int>(w * ratio);

    int top = (target_h - new_h) / 2;
    int left = (target_w - new_w) / 2;
    int bottom = target_h - new_h - top;
    int right = target_w - new_w - left;

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h));
    cv::copyMakeBorder(resized, result.img, top, bottom, left, right,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    result.ratio = ratio;
    result.pad_left = left;
    result.pad_top = top;
    return result;
}

std::vector<std::vector<float>> LicensePlateRecognizer::postprocess_detection(
    const float* output, int num_dets, int num_attrs,
    float ratio, int pad_left, int pad_top) {

    /* Each detection row: [cx, cy, w, h, obj, lmk0_x, lmk0_y, lmk1_x, lmk1_y,
       lmk2_x, lmk2_y, lmk3_x, lmk3_y, class0_score, class1_score]
       (15 attributes total, index 0-14) */

    /* Step 1: filter by objectness > min_score */
    std::vector<std::vector<float>> filtered;
    for (int i = 0; i < num_dets; ++i) {
        const float* row = output + i * num_attrs;
        float obj = row[4];
        if (obj <= min_score_) continue;

        /* Multiply class scores by objectness */
        float class0 = row[13] * obj;
        float class1 = row[14] * obj;

        /* Pick the class with higher score */
        float score;
        int label;
        if (class1 > class0) {
            score = class1;
            label = 1;
        } else {
            score = class0;
            label = 0;
        }
        if (score < min_score_) continue;

        /* Convert cx,cy,w,h -> x1,y1,x2,y2 */
        float cx = row[0], cy = row[1], w = row[2], h = row[3];
        float x1 = cx - w / 2.0f;
        float y1 = cy - h / 2.0f;
        float x2 = cx + w / 2.0f;
        float y2 = cy + h / 2.0f;

        /* Restore from letterbox: subtract padding, divide by ratio */
        x1 = (x1 - pad_left) / ratio;
        y1 = (y1 - pad_top) / ratio;
        x2 = (x2 - pad_left) / ratio;
        y2 = (y2 - pad_top) / ratio;

        /* Restore landmarks: [lmk0_x, lmk0_y, ..., lmk3_x, lmk3_y] at indices 5-12 */
        float landmarks[4][2];
        for (int k = 0; k < 4; ++k) {
            landmarks[k][0] = (row[5 + k * 2] - pad_left) / ratio;
            landmarks[k][1] = (row[6 + k * 2] - pad_top) / ratio;
        }

        /* Build result row: [x1, y1, x2, y2, score, label, lmk0_x, lmk0_y, ..., lmk3_x, lmk3_y] */
        std::vector<float> det;
        det.reserve(2 + 5 + 8); /* 15 values: x1,y1,x2,y2,score,label + 8 landmarks */
        det.push_back(x1);
        det.push_back(y1);
        det.push_back(x2);
        det.push_back(y2);
        det.push_back(score);
        det.push_back(static_cast<float>(label));
        for (int k = 0; k < 4; ++k) {
            det.push_back(landmarks[k][0]);
            det.push_back(landmarks[k][1]);
        }
        filtered.push_back(std::move(det));
    }

    return filtered;
}

std::vector<int> LicensePlateRecognizer::nms(
    const std::vector<std::vector<float>>& boxes, float iou_thresh) {
    /* boxes[i] = [x1, y1, x2, y2, score, ...] */
    int n = static_cast<int>(boxes.size());
    if (n == 0) return {};

    /* Sort indices by score descending */
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return boxes[a][4] > boxes[b][4];
    });

    std::vector<int> keep;
    std::vector<bool> suppressed(n, false);
    for (int i = 0; i < n; ++i) {
        int idx = indices[i];
        if (suppressed[idx]) continue;
        keep.push_back(idx);

        float x1 = boxes[idx][0], y1 = boxes[idx][1];
        float x2 = boxes[idx][2], y2 = boxes[idx][3];
        float area = (x2 - x1) * (y2 - y1);

        for (int j = i + 1; j < n; ++j) {
            int jdx = indices[j];
            if (suppressed[jdx]) continue;

            float jx1 = boxes[jdx][0], jy1 = boxes[jdx][1];
            float jx2 = boxes[jdx][2], jy2 = boxes[jdx][3];
            float jarea = (jx2 - jx1) * (jy2 - jy1);

            float ix1 = std::max(x1, jx1);
            float iy1 = std::max(y1, jy1);
            float ix2 = std::min(x2, jx2);
            float iy2 = std::min(y2, jy2);
            float iw = std::max(0.0f, ix2 - ix1);
            float ih = std::max(0.0f, iy2 - iy1);
            float inter = iw * ih;
            float iou = inter / (area + jarea - inter + 1e-6f);

            if (iou > iou_thresh) {
                suppressed[jdx] = true;
            }
        }
    }
    return keep;
}

std::vector<cv::Point2f> LicensePlateRecognizer::order_points(
    const std::vector<cv::Point2f>& pts) {
    std::vector<cv::Point2f> rect(4);
    if (pts.size() < 4) return rect;

    /* top-left has smallest sum, bottom-right has largest sum */
    auto cmp_sum = [](const cv::Point2f& a, const cv::Point2f& b) {
        return (a.x + a.y) < (b.x + b.y);
    };
    auto min_s = std::min_element(pts.begin(), pts.end(), cmp_sum);
    auto max_s = std::max_element(pts.begin(), pts.end(), cmp_sum);
    rect[0] = *min_s;
    rect[2] = *max_s;

    /* top-right has smallest diff (x-y), bottom-left has largest diff */
    std::vector<cv::Point2f> tmp;
    for (size_t i = 0; i < pts.size(); ++i) {
        if (&pts[i] != &*min_s && &pts[i] != &*max_s) {
            tmp.push_back(pts[i]);
        }
    }
    auto cmp_diff = [](const cv::Point2f& a, const cv::Point2f& b) {
        return (a.y - a.x) < (b.y - b.x);
    };
    auto min_d = std::min_element(tmp.begin(), tmp.end(), cmp_diff);
    auto max_d = std::max_element(tmp.begin(), tmp.end(), cmp_diff);
    rect[1] = *min_d;  /* top-right */
    rect[3] = *max_d;  /* bottom-left */
    return rect;
}

cv::Mat LicensePlateRecognizer::four_point_transform(
    const cv::Mat& image, const std::vector<cv::Point2f>& pts) {
    auto rect = order_points(pts);
    cv::Point2f tl = rect[0], tr = rect[1], br = rect[2], bl = rect[3];

    float width_a = static_cast<float>(cv::norm(br - bl));
    float width_b = static_cast<float>(cv::norm(tr - tl));
    int max_width = std::max({static_cast<int>(width_a), static_cast<int>(width_b), 1});

    float height_a = static_cast<float>(cv::norm(tr - br));
    float height_b = static_cast<float>(cv::norm(tl - bl));
    int max_height = std::max({static_cast<int>(height_a), static_cast<int>(height_b), 1});

    std::vector<cv::Point2f> dst = {
        {0, 0},
        {static_cast<float>(max_width - 1), 0},
        {static_cast<float>(max_width - 1), static_cast<float>(max_height - 1)},
        {0, static_cast<float>(max_height - 1)}
    };
    cv::Mat M = cv::getPerspectiveTransform(rect, dst);
    cv::Mat warped;
    cv::warpPerspective(image, warped, M, cv::Size(max_width, max_height));
    return warped;
}

cv::Mat LicensePlateRecognizer::split_merge(const cv::Mat& img) {
    int h = img.rows;
    cv::Mat upper = img(cv::Range(0, static_cast<int>(5.0 / 12.0 * h)), cv::Range::all());
    cv::Mat lower = img(cv::Range(static_cast<int>(1.0 / 3.0 * h), h), cv::Range::all());
    cv::resize(upper, upper, cv::Size(lower.cols, lower.rows));
    cv::Mat out;
    cv::hconcat(upper, lower, out);
    return out;
}

std::string LicensePlateRecognizer::decode_plate(const int* preds, int seq_len) {
    /* CTC-like decode: skip blank (index 0) and deduplicate consecutive same tokens */
    std::string result;
    int previous = 0;
    for (int t = 0; t < seq_len; ++t) {
        int pred = preds[t];
        if (pred != 0 && pred != previous && pred < static_cast<int>(PLATE_CHARS.size())) {
            result += PLATE_CHARS[static_cast<size_t>(pred)];
        }
        previous = pred;
    }
    return result;
}

std::string LicensePlateRecognizer::recognize_text(const cv::Mat& plate_img) {
    /* Resize to (168, 48), normalize, transpose to CHW */
    cv::Mat resized;
    cv::resize(plate_img, resized, cv::Size(168, 48));

    resized.convertTo(resized, CV_32F, 1.0 / 255.0);
    resized -= cv::Scalar(PLATE_MEAN, PLATE_MEAN, PLATE_MEAN);
    resized /= PLATE_STD;

    /* HWC -> CHW */
    std::vector<float> input_data(3 * 48 * 168);
    std::vector<cv::Mat> channels(3);
    cv::split(resized, channels);
    for (int c = 0; c < 3; ++c) {
        std::memcpy(input_data.data() + c * 48 * 168, channels[c].data,
                    48 * 168 * sizeof(float));
    }

    /* Run inference: input [1, 3, 48, 168] */
    std::vector<int64_t> input_shape = {1, 3, 48, 168};
    auto outputs = rec_session_->run(input_shape, input_data);

    /* Output shape: [1, seq_len, num_classes] */
    auto& output = outputs[0];
    auto info = output.GetTensorTypeAndShapeInfo();
    auto shape = info.GetShape();
    const float* output_data = output.GetTensorData<float>();

    int seq_len = static_cast<int>(shape[1]);
    int num_classes = static_cast<int>(shape[2]);

    /* Argmax over classes for each time step */
    std::vector<int> preds(seq_len);
    for (int t = 0; t < seq_len; ++t) {
        const float* step = output_data + t * num_classes;
        int max_id = 0;
        float max_val = step[0];
        for (int c = 1; c < num_classes; ++c) {
            if (step[c] > max_val) {
                max_val = step[c];
                max_id = c;
            }
        }
        preds[t] = max_id;
    }

    return decode_plate(preds.data(), seq_len);
}

std::vector<PlateResult> LicensePlateRecognizer::recognize(const cv::Mat& img) {
    std::vector<PlateResult> results;
    if (img.empty()) return results;

    /* Step 1: Letterbox to 640x640 */
    auto lb = letterbox(img, 640, 640);

    /* Step 2: Preprocess - BGR->RGB, HWC->CHW, normalize */
    cv::Mat rgb;
    cv::cvtColor(lb.img, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

    std::vector<float> input_data(3 * 640 * 640);
    std::vector<cv::Mat> channels(3);
    cv::split(rgb, channels);
    for (int c = 0; c < 3; ++c) {
        std::memcpy(input_data.data() + c * 640 * 640, channels[c].data,
                    640 * 640 * sizeof(float));
    }

    /* Step 3: Run detection inference */
    std::vector<int64_t> input_shape = {1, 3, 640, 640};
    auto outputs = det_session_->run(input_shape, input_data);

    /* Step 4: Get output - shape [1, num_dets, num_attrs] or [1, num_attrs, num_dets]
       num_attrs varies by model: typically 15 (2-class) or 16 (3-class with extra column).
       Layout: [cx, cy, w, h, obj, lmk0_x, lmk0_y, ..., lmk3_x, lmk3_y, class0, class1, ...] */
    auto& output = outputs[0];
    auto info = output.GetTensorTypeAndShapeInfo();
    auto shape = info.GetShape();
    const float* output_data = output.GetTensorData<float>();

    if (shape.size() != 3) {
        return results;
    }

    /* Determine output layout: [1, num_attrs, num_dets] or [1, num_dets, num_attrs]
       num_attrs is the smaller dimension (typically 15 or 16).
       If shape[1] < shape[2], layout is [1, attrs, dets] and needs transpose. */
    int num_dets;
    int num_attrs;
    const float* det_data;
    std::vector<float> transposed;

    if (shape[1] < shape[2]) {
        num_attrs = static_cast<int>(shape[1]);
        num_dets = static_cast<int>(shape[2]);
        transposed.resize(num_dets * num_attrs);
        for (int i = 0; i < num_dets; ++i) {
            for (int j = 0; j < num_attrs; ++j) {
                transposed[i * num_attrs + j] = output_data[j * num_dets + i];
            }
        }
        det_data = transposed.data();
    } else {
        num_dets = static_cast<int>(shape[1]);
        num_attrs = static_cast<int>(shape[2]);
        det_data = output_data;
    }

    auto dets = postprocess_detection(det_data, num_dets, num_attrs,
                                      lb.ratio, lb.pad_left, lb.pad_top);
    auto keep_indices = nms(dets, iou_thresh_);

    int img_h = img.rows;
    int img_w = img.cols;

    for (int idx : keep_indices) {
        const auto& det = dets[idx];
        PlateResult r;
        r.score = det[4];
        if (r.score < min_score_) continue;

        int label = static_cast<int>(det[5]);
        r.type = (label == 1) ? "double_layer" : "single_layer";

        r.box.x1 = std::max(0, std::min(img_w - 1, static_cast<int>(det[0])));
        r.box.y1 = std::max(0, std::min(img_h - 1, static_cast<int>(det[1])));
        r.box.x2 = std::max(0, std::min(img_w - 1, static_cast<int>(det[2])));
        r.box.y2 = std::max(0, std::min(img_h - 1, static_cast<int>(det[3])));

        std::vector<cv::Point2f> lmk_pts(4);
        for (int k = 0; k < 4; ++k) {
            r.landmarks[k].x = det[6 + k * 2];
            r.landmarks[k].y = det[7 + k * 2];
            lmk_pts[k] = cv::Point2f(r.landmarks[k].x, r.landmarks[k].y);
        }

        cv::Mat roi = four_point_transform(img, lmk_pts);
        if (label == 1) {
            roi = split_merge(roi);
        }

        r.plate = recognize_text(roi);
        results.push_back(std::move(r));
    }

    return results;
}

} // namespace ocr