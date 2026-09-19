#ifndef LICENSE_PLATE_H
#define LICENSE_PLATE_H

#include <memory>
#include <string>
#include <vector>
#include "onnx_session.h"
#include <opencv2/opencv.hpp>

namespace ocr {

struct PointF {
    float x;
    float y;
};

struct RectBox {
    int x1;
    int y1;
    int x2;
    int y2;
};

/* License plate recognition result. */
struct PlateResult {
    RectBox box;
    float score;
    std::string plate;
    std::string type;
    PointF landmarks[4];
};

/* License plate detector + recognizer.
   Wraps the plate detection ONNX model and plate recognition ONNX model.
   Detection uses a YOLO-style model (640x640 input, letterbox preprocessing).
   Recognition uses a CTC-like model (168x48 input). */
class LicensePlateRecognizer {
public:
    struct Config {
        std::string detect_model_path; /* car_plate_detect.onnx */
        std::string rec_model_path;    /* plate_rec.onnx */
        bool use_gpu = false;
        int gpu_id = 0;
        float min_score = 0.4f;  /* confidence threshold */
        float iou_thresh = 0.5f; /* NMS IoU threshold */
    };

    explicit LicensePlateRecognizer( const Config &cfg );
    ~LicensePlateRecognizer();

    LicensePlateRecognizer( const LicensePlateRecognizer & ) = delete;
    LicensePlateRecognizer &operator=( const LicensePlateRecognizer & ) = delete;

    /* Recognize license plates in a BGR image.
       Returns a list of plate results. */
    std::vector<PlateResult> recognize( const cv::Mat &img );

private:
    std::unique_ptr<OnnxSession> det_session_;
    std::unique_ptr<OnnxSession> rec_session_;
    float min_score_;
    float iou_thresh_;

    /* Letterbox preprocessing: resize keeping aspect ratio, pad to 640x640. */
    struct LetterboxResult {
        cv::Mat img;
        float ratio;
        int pad_left;
        int pad_top;
    };
    LetterboxResult letterbox( const cv::Mat &img, int target_h, int target_w );

    /* Detection post-processing: filter by confidence, xywh->xyxy, NMS, restore coords. */
    std::vector<std::vector<float>> postprocess_detection(
        const float *output, int num_dets, int num_attrs,
        float ratio, int pad_left, int pad_top );

    /* NMS for detection boxes. */
    std::vector<int> nms( const std::vector<std::vector<float>> &boxes, float iou_thresh );

    /* Order 4 points clockwise from top-left. */
    std::vector<cv::Point2f> order_points( const std::vector<cv::Point2f> &pts );

    /* Perspective transform to extract the plate region. */
    cv::Mat four_point_transform( const cv::Mat &image, const std::vector<cv::Point2f> &pts );

    /* Split and merge for double-layer plates. */
    cv::Mat split_merge( const cv::Mat &img );

    /* Recognize plate text from a cropped plate image. */
    std::string recognize_text( const cv::Mat &plate_img );

    /* Decode plate characters from model output. */
    std::string decode_plate( const int *preds, int seq_len );

    /* Character table for plate recognition (each element is a UTF-8 character). */
    static const std::vector<std::string> PLATE_CHARS;
};

} // namespace ocr

#endif /* LICENSE_PLATE_H */