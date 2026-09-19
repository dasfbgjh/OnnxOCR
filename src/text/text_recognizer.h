#ifndef TEXT_RECOGNIZER_H
#define TEXT_RECOGNIZER_H

#include <memory>
#include <string>
#include <vector>
#include "onnx_session.h"
#include "ctc_decode.h"
#include <opencv2/opencv.hpp>

namespace ocr {

/* Text recognizer using CTC-based recognition model.
   Wraps the recognition ONNX model and performs pre/post-processing. */
class TextRecognizer {
public:
    struct Config {
        std::string model_path;
        std::string char_dict_path;
        bool use_space_char = true;
        bool use_gpu = false;
        int gpu_id = 0;
        int rec_batch_num = 6;
        int rec_image_c = 3;
        int rec_image_h = 48;
        int rec_image_w = 320;
        float drop_score = 0.5f;
    };

    explicit TextRecognizer( const Config &cfg );
    ~TextRecognizer();

    TextRecognizer( const TextRecognizer & ) = delete;
    TextRecognizer &operator=( const TextRecognizer & ) = delete;

    /* Recognize text from a list of cropped text images.
       Returns a vector of (text, score) pairs, same order as input. */
    struct Result {
        std::string text;
        float score;
    };
    std::vector<Result> recognize( const std::vector<cv::Mat> &img_list );

private:
    std::unique_ptr<OnnxSession> session_;
    std::unique_ptr<CTCLabelDecode> postprocess_;
    int img_c_, img_h_, img_w_;
    int batch_num_;
    float drop_score_;
};

} // namespace ocr

#endif /* TEXT_RECOGNIZER_H */
