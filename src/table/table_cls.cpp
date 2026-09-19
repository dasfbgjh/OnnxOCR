#include "table_cls.h"

#include <cmath>
#include <numeric>

namespace ocr {

constexpr float TableClassifier::MEAN[3];
constexpr float TableClassifier::STD[3];

TableClassifier::TableClassifier( const Config &cfg )
    : cls_type_( cfg.cls_type ) {

    session_ = std::make_unique<OnnxSession>( cfg.model_path, cfg.use_gpu, cfg.gpu_id );

    try {
        std::vector<int64_t> warmup_shape = { 1, 3, INP_H, INP_W };
        std::vector<float> dummy( 3 * INP_H * INP_W, 0.0f );
        session_->run( warmup_shape, dummy );
    } catch ( ... ) {
    }
}

TableClassifier::~TableClassifier() = default;

cv::Mat TableClassifier::preprocess_paddle( const cv::Mat &img ) {
    int h = img.rows, w = img.cols;
    int short_side = std::min( h, w );
    float percent = static_cast<float>( RESIZE_SHORT ) / short_side;
    int new_w = static_cast<int>( std::round( w * percent ) );
    int new_h = static_cast<int>( std::round( h * percent ) );

    cv::Mat resized;
    cv::resize( img, resized, cv::Size( new_w, new_h ), 0, 0, cv::INTER_LANCZOS4 );

    int w_start = ( new_w - INP_W ) / 2;
    int h_start = ( new_h - INP_H ) / 2;
    cv::Rect crop( w_start, h_start, INP_W, INP_H );
    cv::Mat cropped = resized( crop ).clone();

    cv::Mat normalized;
    cropped.convertTo( normalized, CV_32F, 1.0 / 255.0 );
    cv::subtract( normalized, cv::Scalar( MEAN[0], MEAN[1], MEAN[2] ), normalized );
    cv::divide( normalized, cv::Scalar( STD[0], STD[1], STD[2] ), normalized );

    return normalized;
}

cv::Mat TableClassifier::preprocess_q( const cv::Mat &img ) {
    cv::Mat rgb;
    cv::cvtColor( img, rgb, cv::COLOR_BGR2RGB );

    cv::Mat gray;
    cv::cvtColor( rgb, gray, cv::COLOR_RGB2GRAY );

    cv::Mat gray3ch;
    std::vector<cv::Mat> channels = { gray, gray, gray };
    cv::merge( channels, gray3ch );

    cv::Mat resized;
    cv::resize( gray3ch, resized, cv::Size( INP_W, INP_H ) );

    cv::Mat normalized;
    resized.convertTo( normalized, CV_32F, 1.0 / 255.0 );
    cv::subtract( normalized, cv::Scalar( MEAN[0], MEAN[1], MEAN[2] ), normalized );
    cv::divide( normalized, cv::Scalar( STD[0], STD[1], STD[2] ), normalized );

    return normalized;
}

std::string TableClassifier::classify( const cv::Mat &img ) {
    if ( img.empty() )
        return "wireless";

    cv::Mat preprocessed;
    if ( cls_type_ == TableClsType::PADDLE_CLS ) {
        preprocessed = preprocess_paddle( img );
    } else {
        preprocessed = preprocess_q( img );
    }

    std::vector<float> input_data( 3 * INP_H * INP_W );
    std::vector<cv::Mat> ch( 3 );
    cv::split( preprocessed, ch );
    for ( int c = 0; c < 3; ++c ) {
        std::memcpy( input_data.data() + c * INP_H * INP_W,
                     ch[c].data, INP_H * INP_W * sizeof( float ) );
    }

    std::vector<int64_t> input_shape = { 1, 3, INP_H, INP_W };
    auto outputs = session_->run( input_shape, input_data );

    if ( outputs.empty() )
        return "wireless";

    auto &out = outputs[0];
    auto info = out.GetTensorTypeAndShapeInfo();
    auto shape = info.GetShape();
    const float *data = out.GetTensorData<float>();

    int batch_size = static_cast<int>( shape[0] );
    int num_classes = static_cast<int>( shape[1] );

    std::vector<float> probs( num_classes );
    if ( cls_type_ == TableClsType::Q_CLS ) {
        float max_val = *std::max_element( data, data + num_classes );
        float sum = 0.0f;
        for ( int c = 0; c < num_classes; ++c ) {
            probs[c] = std::exp( data[c] - max_val );
            sum += probs[c];
        }
        for ( int c = 0; c < num_classes; ++c )
            probs[c] /= sum;
    } else {
        std::memcpy( probs.data(), data, num_classes * sizeof( float ) );
    }

    int pred_idx = 0;
    float max_prob = probs[0];
    for ( int c = 1; c < num_classes; ++c ) {
        if ( probs[c] > max_prob ) {
            max_prob = probs[c];
            pred_idx = c;
        }
    }

    return ( pred_idx == 0 ) ? "wired" : "wireless";
}

} // namespace ocr