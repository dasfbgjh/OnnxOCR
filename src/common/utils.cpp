#include "utils.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

namespace ocr {

// Thread-local error message
static thread_local std::string g_last_error;

void set_last_error(const std::string& msg) {
    g_last_error = msg;
}

std::string get_last_error() {
    return g_last_error;
}

std::vector<cv::Point2f> order_points_clockwise(const std::vector<cv::Point2f>& pts) {
    std::vector<cv::Point2f> rect(4);
    if (pts.size() < 4) return rect;

    // s = x + y; rect[0] = min(s), rect[2] = max(s)
    auto cmp_sum = [](const cv::Point2f& a, const cv::Point2f& b) {
        return (a.x + a.y) < (b.x + b.y);
    };
    auto min_s = std::min_element(pts.begin(), pts.end(), cmp_sum);
    auto max_s = std::max_element(pts.begin(), pts.end(), cmp_sum);
    rect[0] = *min_s;
    rect[2] = *max_s;

    // remaining two points
    std::vector<cv::Point2f> tmp;
    for (size_t i = 0; i < pts.size(); ++i) {
        if (&pts[i] != &*min_s && &pts[i] != &*max_s) {
            tmp.push_back(pts[i]);
        }
    }
    // diff = y - x (matches Python np.diff(axis=1)); rect[1] = min(diff), rect[3] = max(diff)
    auto cmp_diff = [](const cv::Point2f& a, const cv::Point2f& b) {
        return (a.y - a.x) < (b.y - b.x);
    };
    auto min_d = std::min_element(tmp.begin(), tmp.end(), cmp_diff);
    auto max_d = std::max_element(tmp.begin(), tmp.end(), cmp_diff);
    rect[1] = *min_d;
    rect[3] = *max_d;
    return rect;
}

cv::Point2f clip_point(cv::Point2f p, int img_h, int img_w) {
    p.x = std::max(0.0f, std::min(p.x, static_cast<float>(img_w - 1)));
    p.y = std::max(0.0f, std::min(p.y, static_cast<float>(img_h - 1)));
    return p;
}

static TextBox to_textbox(const std::vector<cv::Point2f>& pts) {
    TextBox box;
    for (int i = 0; i < 4 && i < static_cast<int>(pts.size()); ++i) {
        box.pts[i] = {pts[i].x, pts[i].y};
    }
    return box;
}

static std::vector<cv::Point2f> from_textbox(const TextBox& box) {
    std::vector<cv::Point2f> pts(4);
    for (int i = 0; i < 4; ++i) {
        pts[i] = cv::Point2f(box.pts[i].x, box.pts[i].y);
    }
    return pts;
}

std::vector<TextBox> filter_tag_det_res(const std::vector<TextBox>& boxes, int img_h, int img_w) {
    std::vector<TextBox> result;
    for (const auto& box : boxes) {
        auto pts = from_textbox(box);
        pts = order_points_clockwise(pts);
        for (auto& p : pts) p = clip_point(p, img_h, img_w);

        float w = static_cast<float>(cv::norm(pts[0] - pts[1]));
        float h = static_cast<float>(cv::norm(pts[0] - pts[3]));
        if (w <= 3.0f || h <= 3.0f) continue;
        result.push_back(to_textbox(pts));
    }
    return result;
}

std::vector<TextBox> filter_tag_det_res_only_clip(const std::vector<TextBox>& boxes, int img_h, int img_w) {
    std::vector<TextBox> result;
    for (const auto& box : boxes) {
        auto pts = from_textbox(box);
        for (auto& p : pts) p = clip_point(p, img_h, img_w);
        result.push_back(to_textbox(pts));
    }
    return result;
}

cv::Mat get_rotate_crop_image(const cv::Mat& img, const TextBox& box) {
    std::vector<cv::Point2f> pts = from_textbox(box);

    // Compute crop width = max(|p0-p1|, |p2-p3|)
    float w1 = static_cast<float>(cv::norm(pts[0] - pts[1]));
    float w2 = static_cast<float>(cv::norm(pts[2] - pts[3]));
    int crop_w = static_cast<int>(std::max(w1, w2));

    // Compute crop height = max(|p0-p3|, |p1-p2|)
    float h1 = static_cast<float>(cv::norm(pts[0] - pts[3]));
    float h2 = static_cast<float>(cv::norm(pts[1] - pts[2]));
    int crop_h = static_cast<int>(std::max(h1, h2));

    if (crop_w <= 0 || crop_h <= 0) {
        return cv::Mat();
    }

    std::vector<cv::Point2f> dst = {
        {0, 0},
        {static_cast<float>(crop_w), 0},
        {static_cast<float>(crop_w), static_cast<float>(crop_h)},
        {0, static_cast<float>(crop_h)}
    };

    cv::Mat M = cv::getPerspectiveTransform(pts, dst);
    cv::Mat dst_img;
    cv::warpPerspective(img, dst_img, M, cv::Size(crop_w, crop_h),
                        cv::INTER_CUBIC, cv::BORDER_REPLICATE);

    // If tall and narrow, rotate 90 degrees
    if (dst_img.rows * 1.0 / dst_img.cols >= 1.5) {
        cv::rotate(dst_img, dst_img, cv::ROTATE_90_COUNTERCLOCKWISE);
    }
    return dst_img;
}

cv::Mat get_minarea_rect_crop(const cv::Mat& img, const TextBox& box) {
    std::vector<cv::Point2f> pts = from_textbox(box);
    std::vector<cv::Point2i> int_pts;
    for (const auto& p : pts) int_pts.emplace_back(static_cast<int>(p.x), static_cast<int>(p.y));

    cv::RotatedRect rrect = cv::minAreaRect(int_pts);
    std::vector<cv::Point2f> box_pts(4);
    rrect.points(box_pts.data());

    // Sort by x
    std::sort(box_pts.begin(), box_pts.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
        return a.x < b.x;
    });

    // Reorder like Python code
    int idx_a, idx_b, idx_c, idx_d;
    if (box_pts[1].y > box_pts[0].y) { idx_a = 0; idx_d = 1; }
    else { idx_a = 1; idx_d = 0; }
    if (box_pts[3].y > box_pts[2].y) { idx_b = 2; idx_c = 3; }
    else { idx_b = 3; idx_c = 2; }

    std::vector<cv::Point2f> reordered = {box_pts[idx_a], box_pts[idx_b], box_pts[idx_c], box_pts[idx_d]};
    TextBox reordered_box = to_textbox(reordered);
    return get_rotate_crop_image(img, reordered_box);
}

std::vector<TextBox> sorted_boxes(const std::vector<TextBox>& boxes) {
    if (boxes.empty()) return {};

    std::vector<TextBox> sorted = boxes;

    // Sort by (top-left y, top-left x)
    std::sort(sorted.begin(), sorted.end(), [](const TextBox& a, const TextBox& b) {
        if (a.pts[0].y != b.pts[0].y) return a.pts[0].y < b.pts[0].y;
        return a.pts[0].x < b.pts[0].x;
    });

    // Bubble-like swap for same-row boxes
    int n = static_cast<int>(sorted.size());
    for (int i = 0; i < n - 1; ++i) {
        for (int j = i; j >= 0; --j) {
            float dy = sorted[j + 1].pts[0].y - sorted[j].pts[0].y;
            if (std::abs(dy) < 10.0f && sorted[j + 1].pts[0].x < sorted[j].pts[0].x) {
                std::swap(sorted[j], sorted[j + 1]);
            } else {
                break;
            }
        }
    }
    return sorted;
}

} // namespace ocr