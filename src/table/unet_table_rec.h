#ifndef UNET_TABLE_REC_H
#define UNET_TABLE_REC_H

#include <memory>
#include <string>
#include <vector>
#include "onnx_session.h"
#include <opencv2/opencv.hpp>

namespace ocr {

struct UnetTableResult {
    std::vector<std::vector<float>> cell_bboxes;
    std::vector<std::vector<float>> logic_points;
};

class UnetTableRecognizer {
public:
    struct Config {
        std::string model_path;
        bool use_gpu = false;
        int gpu_id = 0;
    };

    explicit UnetTableRecognizer(const Config& cfg);
    ~UnetTableRecognizer();

    UnetTableRecognizer(const UnetTableRecognizer&) = delete;
    UnetTableRecognizer& operator=(const UnetTableRecognizer&) = delete;

    UnetTableResult recognize(const cv::Mat& img);

private:
    std::unique_ptr<OnnxSession> session_;

    static constexpr int INP_H = 1024;
    static constexpr int INP_W = 1024;
    static constexpr float MEAN[3] = {123.675f, 116.28f, 103.53f};
    static constexpr float STD[3]  = {58.395f, 57.12f, 57.375f};

    cv::Mat preprocess(const cv::Mat& img);
    cv::Mat infer(const cv::Mat& preprocessed);
    UnetTableResult postprocess(const cv::Mat& img, const cv::Mat& pred);

    std::vector<cv::Vec4f> get_table_line(const cv::Mat& mask, int axis, int lineW);
    std::vector<cv::Vec4f> adjust_lines(const std::vector<cv::Vec4f>& lines,
                                         int alph = 50, int angle = 50);
    void final_adjust_lines(std::vector<cv::Vec4f>& rowboxes,
                            std::vector<cv::Vec4f>& colboxes);

    std::vector<std::vector<cv::Point2f>> extract_cells(const cv::Mat& line_img,
                                                         int img_w, int img_h);

    std::vector<std::vector<float>> recover_logic_points(
        const std::vector<std::vector<cv::Point2f>>& sorted_polygons,
        int row_thresh = 10, int col_thresh = 15);

    static cv::Mat resize_img(const cv::Mat& img, int target_h, int target_w);
    static void sort_polygons(std::vector<std::vector<cv::Point2f>>& polygons);
};

} // namespace ocr

#endif /* UNET_TABLE_REC_H */