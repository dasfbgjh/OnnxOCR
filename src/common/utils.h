#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

namespace ocr {

/* A 2D point. */
struct Point {
    float x;
    float y;
};

/* A text box: 4 corner points. */
struct TextBox {
    Point pts[4];
};

/* Sort text boxes from top to bottom, left to right (same as Python sorted_boxes). */
std::vector<TextBox> sorted_boxes(const std::vector<TextBox>& boxes);

/* Crop a rotated text region from the image using perspective transform.
   If the crop is taller than wide (ratio >= 1.5), rotate 90 degrees. */
cv::Mat get_rotate_crop_image(const cv::Mat& img, const TextBox& box);

/* Crop using minAreaRect (for "poly" box type). */
cv::Mat get_minarea_rect_crop(const cv::Mat& img, const TextBox& box);

/* Order 4 points clockwise starting from top-left.
   rect[0] = top-left (min sum), rect[2] = bottom-right (max sum)
   rect[1] = top-right, rect[3] = bottom-left */
std::vector<cv::Point2f> order_points_clockwise(const std::vector<cv::Point2f>& pts);

/* Clip a point to image bounds. */
cv::Point2f clip_point(cv::Point2f p, int img_h, int img_w);

/* Filter detected boxes: order points, clip, remove too-small boxes. */
std::vector<TextBox> filter_tag_det_res(const std::vector<TextBox>& boxes, int img_h, int img_w);

/* Filter: only clip, no size filtering (for "poly" box type). */
std::vector<TextBox> filter_tag_det_res_only_clip(const std::vector<TextBox>& boxes, int img_h, int img_w);

/* Set the thread-local error message. */
void set_last_error(const std::string& msg);

/* Get the thread-local error message. */
std::string get_last_error();

} // namespace ocr

#endif /* UTILS_H */
