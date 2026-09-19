#include "db_post_process.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace ocr {

/* ---- Polygon geometry helpers (replaces shapely + pyclipper) ---- */

/* Shoelace formula for polygon area. Returns absolute area. */
static float polygon_area(const std::vector<cv::Point2f>& poly) {
    float area = 0.0f;
    int n = static_cast<int>(poly.size());
    for (int i = 0; i < n; ++i) {
        int j = (i + 1) % n;
        area += poly[i].x * poly[j].y - poly[j].x * poly[i].y;
    }
    return std::abs(area) * 0.5f;
}

/* Polygon perimeter (sum of edge lengths). */
static float polygon_perimeter(const std::vector<cv::Point2f>& poly) {
    float perim = 0.0f;
    int n = static_cast<int>(poly.size());
    for (int i = 0; i < n; ++i) {
        int j = (i + 1) % n;
        float dx = poly[j].x - poly[i].x;
        float dy = poly[j].y - poly[i].y;
        perim += std::sqrt(dx * dx + dy * dy);
    }
    return perim;
}

/* Compute signed area: positive = CCW, negative = CW. */
static float signed_area(const std::vector<cv::Point2f>& poly) {
    float area = 0.0f;
    int n = static_cast<int>(poly.size());
    for (int i = 0; i < n; ++i) {
        int j = (i + 1) % n;
        area += poly[i].x * poly[j].y - poly[j].x * poly[i].y;
    }
    return area * 0.5f;
}

/* Expand a polygon outward by `distance` using edge offset + convex hull.
   This mimics pyclipper's ClipperOffset with JT_ROUND/ET_CLOSEDPOLYGON.
   Steps:
   1. Offset each edge outward by `distance` along its normal.
   2. Compute intersection of adjacent offset edges to get offset vertices.
   3. Take convex hull of the result to handle concavities and self-intersections.
   4. Approximate the hull to get a clean polygon. */
static std::vector<cv::Point2f> unclip_polygon(const std::vector<cv::Point2f>& poly, float distance) {
    if (poly.size() < 3 || distance <= 0.0f) return poly;

    std::vector<cv::Point2f> p = poly;
    if (signed_area(p) < 0) {
        std::reverse(p.begin(), p.end());
    }

    int n = static_cast<int>(p.size());

    std::vector<cv::Point2f> offset_pts;
    offset_pts.reserve(n * 2);

    for (int i = 0; i < n; ++i) {
        int prev = (i - 1 + n) % n;
        int next = (i + 1) % n;

        cv::Point2f e1 = p[i] - p[prev];
        float e1_len = static_cast<float>(cv::norm(e1));
        if (e1_len < 1e-6f) {
            offset_pts.push_back(p[i]);
            continue;
        }
        cv::Point2f n1(e1.y / e1_len, -e1.x / e1_len);

        cv::Point2f e2 = p[next] - p[i];
        float e2_len = static_cast<float>(cv::norm(e2));
        if (e2_len < 1e-6f) {
            offset_pts.push_back(p[i] + cv::Point2f(n1.x * distance, n1.y * distance));
            continue;
        }
        cv::Point2f n2(e2.y / e2_len, -e2.x / e2_len);

        cv::Point2f p1 = p[prev] + cv::Point2f(n1.x * distance, n1.y * distance);
        cv::Point2f p2 = p[i] + cv::Point2f(n2.x * distance, n2.y * distance);
        cv::Point2f d1(e1.x / e1_len, e1.y / e1_len);
        cv::Point2f d2(e2.x / e2_len, e2.y / e2_len);

        float cross_dd = d1.x * d2.y - d1.y * d2.x;
        if (std::abs(cross_dd) < 1e-10f) {
            offset_pts.push_back(p[i] + cv::Point2f(n1.x * distance, n1.y * distance));
        } else {
            cv::Point2f diff = p2 - p1;
            float cross_pd = diff.x * d2.y - diff.y * d2.x;
            float t = cross_pd / cross_dd;
            offset_pts.push_back(p1 + cv::Point2f(d1.x * t, d1.y * t));
        }

        float angle = std::acos(std::clamp(
            n1.x * n2.x + n1.y * n2.y, -1.0f, 1.0f));
        if (angle > 0.05f) {
            int arc_steps = std::max(2, static_cast<int>(angle * distance / 2.0f));
            float step_angle = angle / static_cast<float>(arc_steps);
            float cos_a = std::cos(step_angle);
            float sin_a = std::sin(step_angle);
            cv::Point2f dir = n1;
            for (int s = 1; s < arc_steps; ++s) {
                float new_dx = dir.x * cos_a - dir.y * sin_a;
                float new_dy = dir.x * sin_a + dir.y * cos_a;
                dir = cv::Point2f(new_dx, new_dy);
                offset_pts.push_back(p[i] + cv::Point2f(dir.x * distance, dir.y * distance));
            }
        }
    }

    if (offset_pts.size() < 3) return poly;

    std::vector<cv::Point2i> int_pts;
    int_pts.reserve(offset_pts.size());
    for (const auto& pt : offset_pts) {
        int_pts.emplace_back(static_cast<int>(std::round(pt.x)),
                             static_cast<int>(std::round(pt.y)));
    }

    std::vector<cv::Point2i> hull;
    cv::convexHull(int_pts, hull);

    std::vector<cv::Point2f> result;
    result.reserve(hull.size());
    for (const auto& pt : hull) {
        result.emplace_back(static_cast<float>(pt.x), static_cast<float>(pt.y));
    }

    return result;
}

/* Get the minimum bounding rotated rectangle and return the 4 corner points
   in a specific order (top-left, top-right, bottom-right, bottom-left).
   Also returns the minimum side length. */
static std::pair<std::vector<cv::Point2f>, float> get_mini_boxes(const std::vector<cv::Point2f>& contour) {
    std::vector<cv::Point2i> int_pts;
    for (const auto& p : contour) {
        int_pts.emplace_back(static_cast<int>(p.x), static_cast<int>(p.y));
    }

    cv::RotatedRect rrect = cv::minAreaRect(int_pts);
    cv::Point2f box_pts[4];
    rrect.points(box_pts);

    std::vector<cv::Point2f> pts(box_pts, box_pts + 4);
    // Sort by x-coordinate
    std::sort(pts.begin(), pts.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
        return a.x < b.x;
    });

    int idx_1, idx_2, idx_3, idx_4;
    if (pts[1].y > pts[0].y) { idx_1 = 0; idx_4 = 1; }
    else { idx_1 = 1; idx_4 = 0; }
    if (pts[3].y > pts[2].y) { idx_2 = 2; idx_3 = 3; }
    else { idx_2 = 3; idx_3 = 2; }

    std::vector<cv::Point2f> result = {pts[idx_1], pts[idx_2], pts[idx_3], pts[idx_4]};
    float min_side = std::min(rrect.size.width, rrect.size.height);
    return {result, min_side};
}

/* Also overload for cv::InputArray (contour from findContours). */
static std::pair<std::vector<cv::Point2f>, float> get_mini_boxes_from_contour(const std::vector<cv::Point>& contour) {
    cv::RotatedRect rrect = cv::minAreaRect(contour);
    cv::Point2f box_pts[4];
    rrect.points(box_pts);

    std::vector<cv::Point2f> pts(box_pts, box_pts + 4);
    std::sort(pts.begin(), pts.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
        return a.x < b.x;
    });

    int idx_1, idx_2, idx_3, idx_4;
    if (pts[1].y > pts[0].y) { idx_1 = 0; idx_4 = 1; }
    else { idx_1 = 1; idx_4 = 0; }
    if (pts[3].y > pts[2].y) { idx_2 = 2; idx_3 = 3; }
    else { idx_2 = 3; idx_3 = 2; }

    std::vector<cv::Point2f> result = {pts[idx_1], pts[idx_2], pts[idx_3], pts[idx_4]};
    float min_side = std::min(rrect.size.width, rrect.size.height);
    return {result, min_side};
}

/* Compute the mean score of the prediction map inside the box (fast mode).
   Uses fillPoly to create a mask, then computes the mean of pred inside the mask. */
static float box_score_fast(const cv::Mat& pred, const std::vector<cv::Point2f>& box) {
    int h = pred.rows;
    int w = pred.cols;

    float xmin = std::floor(std::min({box[0].x, box[1].x, box[2].x, box[3].x}));
    float xmax = std::ceil(std::max({box[0].x, box[1].x, box[2].x, box[3].x}));
    float ymin = std::floor(std::min({box[0].y, box[1].y, box[2].y, box[3].y}));
    float ymax = std::ceil(std::max({box[0].y, box[1].y, box[2].y, box[3].y}));

    xmin = std::max(0.0f, std::min(xmin, static_cast<float>(w - 1)));
    xmax = std::max(0.0f, std::min(xmax, static_cast<float>(w - 1)));
    ymin = std::max(0.0f, std::min(ymin, static_cast<float>(h - 1)));
    ymax = std::max(0.0f, std::min(ymax, static_cast<float>(h - 1)));

    int iw = static_cast<int>(xmax - xmin + 1);
    int ih = static_cast<int>(ymax - ymin + 1);
    if (iw <= 0 || ih <= 0) return 0.0f;

    cv::Mat mask = cv::Mat::zeros(ih, iw, CV_8U);
    std::vector<cv::Point> mask_pts;
    for (const auto& p : box) {
        mask_pts.emplace_back(
            cv::saturate_cast<int>(p.x - xmin),
            cv::saturate_cast<int>(p.y - ymin));
    }
    cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{mask_pts}, cv::Scalar(1));

    cv::Rect roi(static_cast<int>(xmin), static_cast<int>(ymin), iw, ih);
    cv::Mat pred_roi = pred(roi);

    // Compute mean of pred_roi under mask
    cv::Scalar mean_val = cv::mean(pred_roi, mask);
    return static_cast<float>(mean_val[0]);
}

/* Compute the mean score using the contour (slow mode). */
static float box_score_slow(const cv::Mat& pred, const std::vector<cv::Point>& contour) {
    int h = pred.rows;
    int w = pred.cols;

    std::vector<float> xs, ys;
    for (const auto& p : contour) {
        xs.push_back(static_cast<float>(p.x));
        ys.push_back(static_cast<float>(p.y));
    }
    float xmin = std::max(0.0f, *std::min_element(xs.begin(), xs.end()));
    float xmax = std::min(static_cast<float>(w - 1), *std::max_element(xs.begin(), xs.end()));
    float ymin = std::max(0.0f, *std::min_element(ys.begin(), ys.end()));
    float ymax = std::min(static_cast<float>(h - 1), *std::max_element(ys.begin(), ys.end()));

    int iw = static_cast<int>(std::ceil(xmax) - std::floor(ymin) + 1);
    int ih = static_cast<int>(std::ceil(ymax) - std::floor(ymin) + 1);
    if (iw <= 0 || ih <= 0) return 0.0f;

    int x0 = static_cast<int>(std::floor(xmin));
    int y0 = static_cast<int>(std::floor(ymin));
    int x1 = x0 + iw;
    int y1 = y0 + ih;

    cv::Mat mask = cv::Mat::zeros(ih, iw, CV_8U);
    std::vector<cv::Point> shifted;
    for (const auto& p : contour) {
        shifted.emplace_back(p.x - x0, p.y - y0);
    }
    cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{shifted}, cv::Scalar(1));

    cv::Rect roi(x0, y0, iw, ih);
    // Clamp roi to pred bounds
    roi &= cv::Rect(0, 0, w, h);
    if (roi.width <= 0 || roi.height <= 0) return 0.0f;

    cv::Mat pred_roi = pred(roi);
    cv::Mat mask_roi = mask(cv::Rect(0, 0, roi.width, roi.height));
    cv::Scalar mean_val = cv::mean(pred_roi, mask_roi);
    return static_cast<float>(mean_val[0]);
}

/* boxes_from_bitmap: extract quad boxes from the binary segmentation map. */
static std::vector<TextBox> boxes_from_bitmap(const cv::Mat& pred,
                                               const cv::Mat& bitmap,
                                               float dest_width, float dest_height,
                                               const DBPostProcessConfig& cfg) {
    int height = bitmap.rows;
    int width = bitmap.cols;

    // Find contours
    cv::Mat bitmap_u8;
    bitmap.convertTo(bitmap_u8, CV_8U, 255.0);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bitmap_u8, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    int num_contours = std::min(static_cast<int>(contours.size()), cfg.max_candidates);

    std::vector<TextBox> boxes;
    for (int i = 0; i < num_contours; ++i) {
        const auto& contour = contours[i];
        auto [points, sside] = get_mini_boxes_from_contour(contour);
        if (sside < 3.0f) continue;  // min_size = 3

        float score;
        if (cfg.score_mode == "slow") {
            score = box_score_slow(pred, contour);
        } else {
            score = box_score_fast(pred, points);
        }
        if (cfg.box_thresh > score) continue;

        // Unclip
        float area = polygon_area(points);
        float perim = polygon_perimeter(points);
        float distance = area * cfg.unclip_ratio / perim;
        auto expanded = unclip_polygon(points, distance);
        if (expanded.size() < 3) continue;

        auto [box_pts, sside2] = get_mini_boxes(expanded);
        if (sside2 < 3.0f + 2.0f) continue;  // min_size + 2

        // Scale to original image dimensions
        TextBox tb;
        for (int k = 0; k < 4; ++k) {
            float bx = std::clamp(std::round(box_pts[k].x / width * dest_width), 0.0f, dest_width);
            float by = std::clamp(std::round(box_pts[k].y / height * dest_height), 0.0f, dest_height);
            tb.pts[k] = {bx, by};
        }
        boxes.push_back(tb);
    }
    return boxes;
}

/* polygons_from_bitmap: extract polygon boxes from the binary segmentation map. */
static std::vector<TextBox> polygons_from_bitmap(const cv::Mat& pred,
                                                  const cv::Mat& bitmap,
                                                  float dest_width, float dest_height,
                                                  const DBPostProcessConfig& cfg) {
    int height = bitmap.rows;
    int width = bitmap.cols;

    cv::Mat bitmap_u8;
    bitmap.convertTo(bitmap_u8, CV_8U, 255.0);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bitmap_u8, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    int num_contours = std::min(static_cast<int>(contours.size()), cfg.max_candidates);

    std::vector<TextBox> boxes;
    for (int i = 0; i < num_contours; ++i) {
        const auto& contour = contours[i];
        double epsilon = 0.002 * cv::arcLength(contour, true);
        std::vector<cv::Point> approx;
        cv::approxPolyDP(contour, approx, epsilon, true);
        if (approx.size() < 4) continue;

        std::vector<cv::Point2f> points(approx.begin(), approx.end());
        float score = box_score_fast(pred, points);
        if (cfg.box_thresh > score) continue;

        // Unclip
        float area = polygon_area(points);
        float perim = polygon_perimeter(points);
        float distance = area * cfg.unclip_ratio / perim;
        auto expanded = unclip_polygon(points, distance);
        if (expanded.size() <= 1) continue;

        auto [box_pts, sside] = get_mini_boxes(expanded);
        if (sside < 3.0f + 2.0f) continue;

        TextBox tb;
        for (size_t k = 0; k < expanded.size() && k < 4; ++k) {
            float bx = std::clamp(std::round(expanded[k].x / width * dest_width), 0.0f, dest_width);
            float by = std::clamp(std::round(expanded[k].y / height * dest_height), 0.0f, dest_height);
            if (k < 4) tb.pts[k] = {bx, by};
        }
        boxes.push_back(tb);
    }
    return boxes;
}

std::vector<TextBox> db_post_process(const cv::Mat& pred,
                                      const cv::Mat& bitmap,
                                      float dest_width, float dest_height,
                                      const DBPostProcessConfig& cfg) {
    if (cfg.box_type == "poly") {
        return polygons_from_bitmap(pred, bitmap, dest_width, dest_height, cfg);
    } else {
        return boxes_from_bitmap(pred, bitmap, dest_width, dest_height, cfg);
    }
}

std::vector<TextBox> db_post_process_batch(const cv::Mat& pred_batch,
                                            const std::vector<ShapeListEntry>& shape_list,
                                            const DBPostProcessConfig& cfg) {
    // pred_batch: [B, 1, H, W], we take batch 0
    // For simplicity, we handle batch size 1
    if (shape_list.empty()) return {};

    // Extract pred[0, 0] = the segmentation map
    // pred_batch is expected to be [1, 1, H, W] or [H, W] (already squeezed)
    cv::Mat pred;
    if (pred_batch.dims == 4 || (pred_batch.rows > 0 && pred_batch.cols > 0 && pred_batch.channels() > 1)) {
        // Squeeze: take first batch, first channel
        // For simplicity, assume it's already [H, W]
        pred = pred_batch;
    } else {
        pred = pred_batch;
    }

    const auto& shape = shape_list[0];

    // Segmentation: pred > thresh
    cv::Mat segmentation;
    cv::threshold(pred, segmentation, cfg.thresh, 1.0, cv::THRESH_BINARY);

    cv::Mat mask = segmentation;
    if (cfg.use_dilation) {
        cv::Mat kernel = (cv::Mat_<uchar>(2, 2) << 1, 1, 1, 1);
        cv::Mat seg_u8;
        segmentation.convertTo(seg_u8, CV_8U);
        cv::dilate(seg_u8, mask, kernel);
        mask.convertTo(mask, CV_32F);
    }

    return db_post_process(pred, mask, shape.src_w, shape.src_h, cfg);
}

} // namespace ocr