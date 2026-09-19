#include "table_recognizer.h"
#include "table_cls.h"
#include "unet_table_rec.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <set>

namespace ocr {

constexpr float TableRecognizer::MEAN[3];
constexpr float TableRecognizer::STD[3];

TableRecognizer::TableRecognizer( const Config &cfg )
    : beg_idx_( 0 ), end_idx_( 0 ), model_type_( cfg.model_type ) {

    if ( cfg.model_type == TableModelType::UNET_SLANET_PLUS ) {
        if ( !cfg.cls_model_path.empty() ) {
            TableClassifier::Config cls_cfg;
            cls_cfg.model_path = cfg.cls_model_path;
            cls_cfg.cls_type = TableClsType::Q_CLS;
            cls_cfg.use_gpu = cfg.use_gpu;
            cls_cfg.gpu_id = cfg.gpu_id;
            classifier_ = std::make_unique<TableClassifier>( cls_cfg );
        }

        if ( !cfg.unet_model_path.empty() ) {
            UnetTableRecognizer::Config unet_cfg;
            unet_cfg.model_path = cfg.unet_model_path;
            unet_cfg.use_gpu = cfg.use_gpu;
            unet_cfg.gpu_id = cfg.gpu_id;
            unet_rec_ = std::make_unique<UnetTableRecognizer>( unet_cfg );
        }

        if ( !cfg.model_path.empty() ) {
            session_ = std::make_unique<OnnxSession>( cfg.model_path, cfg.use_gpu, cfg.gpu_id );
            load_characters();
        }
        return;
    }

    if ( cfg.model_type == TableModelType::UNET ) {
        UnetTableRecognizer::Config unet_cfg;
        unet_cfg.model_path = cfg.model_path;
        unet_cfg.use_gpu = cfg.use_gpu;
        unet_cfg.gpu_id = cfg.gpu_id;
        unet_rec_ = std::make_unique<UnetTableRecognizer>( unet_cfg );
        return;
    }

    session_ = std::make_unique<OnnxSession>( cfg.model_path, cfg.use_gpu, cfg.gpu_id );
    load_characters();
}

void TableRecognizer::load_characters() {
    if ( !session_ )
        return;

    characters_.clear();
    auto meta = session_->get_metadata();
    auto it = meta.find( "character" );
    if ( it != meta.end() ) {
        std::string chars = it->second;
        size_t start = 0;
        while ( start < chars.size() ) {
            size_t end = chars.find( '\n', start );
            if ( end == std::string::npos )
                end = chars.size();
            characters_.push_back( chars.substr( start, end - start ) );
            start = end + 1;
        }
    }

    bool has_empty_td = false;
    for ( const auto &c : characters_ ) {
        if ( c == "<td></td>" ) {
            has_empty_td = true;
            break;
        }
    }
    if ( !has_empty_td ) {
        characters_.push_back( "<td></td>" );
    }
    characters_.erase( std::remove( characters_.begin(), characters_.end(), std::string( "<td>" ) ),
                       characters_.end() );

    characters_.insert( characters_.begin(), "sos" );
    characters_.push_back( "eos" );

    for ( size_t i = 0; i < characters_.size(); ++i ) {
        if ( characters_[i] == "sos" )
            beg_idx_ = static_cast<int>( i );
        if ( characters_[i] == "eos" )
            end_idx_ = static_cast<int>( i );
    }

    try {
        std::vector<int64_t> warmup_shape = { 1, 3, MAX_LEN, MAX_LEN };
        std::vector<float> dummy( 3 * MAX_LEN * MAX_LEN, 0.0f );
        session_->run( warmup_shape, dummy );
    } catch ( ... ) {
    }
}

TableRecognizer::~TableRecognizer() = default;

TableResult TableRecognizer::recognize( const cv::Mat &img ) {
    TableResult result;
    if ( img.empty() )
        return result;

    if ( model_type_ == TableModelType::UNET ) {
        if ( unet_rec_ ) {
            auto unet_res = unet_rec_->recognize( img );
            result.cell_bboxes = std::move( unet_res.cell_bboxes );
            result.logic_points = std::move( unet_res.logic_points );
            result.score = 0.0f;
        }
        return result;
    }

    if ( model_type_ == TableModelType::UNET_SLANET_PLUS ) {
        std::string table_type = "wireless";
        if ( classifier_ )
            table_type = classifier_->classify( img );

        if ( table_type == "wired" && unet_rec_ ) {
            auto unet_res = unet_rec_->recognize( img );
            result.cell_bboxes = std::move( unet_res.cell_bboxes );
            result.logic_points = std::move( unet_res.logic_points );
            result.score = 0.0f;
            return result;
        }
    }

    if ( !session_ )
        return result;

    int ori_h = img.rows;
    int ori_w = img.cols;

    /* Step 1: Preprocess
       - Resize: max side to 488 keeping aspect ratio
       - Normalize: (img/255 - mean) / std
       - Pad to 488x488
       - HWC -> CHW */
    int h = img.rows, w = img.cols;
    float ratio = static_cast<float>( MAX_LEN ) / std::max( h, w );
    int resize_h = static_cast<int>( h * ratio );
    int resize_w = static_cast<int>( w * ratio );

    cv::Mat resized;
    cv::resize( img, resized, cv::Size( resize_w, resize_h ) );

    resized.convertTo( resized, CV_32F, 1.0 / 255.0 );
    cv::subtract( resized, cv::Scalar( MEAN[0], MEAN[1], MEAN[2] ), resized );
    cv::divide( resized, cv::Scalar( STD[0], STD[1], STD[2] ), resized );

    /* Pad to 488x488 */
    cv::Mat padded = cv::Mat::zeros( MAX_LEN, MAX_LEN, CV_32FC3 );
    cv::Mat roi( padded, cv::Rect( 0, 0, resize_w, resize_h ) );
    resized.copyTo( roi );

    /* HWC -> CHW */
    std::vector<float> input_data( 3 * MAX_LEN * MAX_LEN );
    std::vector<cv::Mat> channels( 3 );
    cv::split( padded, channels );
    for ( int c = 0; c < 3; ++c ) {
        std::memcpy( input_data.data() + c * MAX_LEN * MAX_LEN,
                     channels[c].data, MAX_LEN * MAX_LEN * sizeof( float ) );
    }

    /* Step 2: Run inference */
    std::vector<int64_t> input_shape = { 1, 3, MAX_LEN, MAX_LEN };
    auto outputs = session_->run( input_shape, input_data );

    if ( outputs.size() < 2 )
        return result;

    /* Output 0: bbox_preds [1, N, 8]
       Output 1: structure_probs [1, N, num_classes] */
    auto &bbox_out = outputs[0];
    auto &struct_out = outputs[1];

    auto bbox_info = bbox_out.GetTensorTypeAndShapeInfo();
    auto bbox_shape = bbox_info.GetShape();
    auto struct_info = struct_out.GetTensorTypeAndShapeInfo();
    auto struct_shape = struct_info.GetShape();

    int num_tokens = static_cast<int>( bbox_shape[1] );
    int num_classes = static_cast<int>( struct_shape[2] );
    int bbox_attr = static_cast<int>( bbox_shape[2] ); /* typically 8 */

    const float *bbox_preds = bbox_out.GetTensorData<float>();
    const float *struct_probs = struct_out.GetTensorData<float>();

    /* Step 3: Decode structure and bboxes */
    std::vector<std::string> structure_list;
    std::vector<std::vector<float>> cell_bboxes;
    float mean_score = 0.0f;

    decode( bbox_preds, struct_probs, num_tokens, num_classes,
            ori_h, ori_w, structure_list, cell_bboxes, mean_score );

    if ( model_type_ == TableModelType::SLANET_PLUS ||
         model_type_ == TableModelType::UNET_SLANET_PLUS ) {
        float w_ratio = static_cast<float>( MAX_LEN ) / static_cast<float>( resize_w );
        float h_ratio = static_cast<float>( MAX_LEN ) / static_cast<float>( resize_h );
        for ( auto &cell : cell_bboxes ) {
            for ( int k = 0; k < 8; k += 2 ) {
                cell[k] *= w_ratio;
                cell[k + 1] *= h_ratio;
            }
        }
        cell_bboxes.erase(
            std::remove_if( cell_bboxes.begin(), cell_bboxes.end(),
                            []( const std::vector<float> &cell ) {
                                return std::all_of( cell.begin(), cell.end(),
                                                    []( float v ) { return v == 0.0f; } );
                            } ),
            cell_bboxes.end() );
    }

    result.cell_bboxes = cell_bboxes;
    result.logic_points = decode_logic_points( structure_list );
    result.score = mean_score;
    return result;
}

void TableRecognizer::decode(
    const float *bbox_preds, const float *struct_probs,
    int num_tokens, int num_classes,
    int ori_h, int ori_w,
    std::vector<std::string> &structure_list,
    std::vector<std::vector<float>> &cell_bboxes,
    float &mean_score ) {

    structure_list.clear();
    cell_bboxes.clear();

    /* TD tokens that trigger bbox decode */
    auto is_td_token = []( const std::string &s ) {
        return s == "<td>" || s == "<td" || s == "<td></td>";
    };

    std::vector<float> scores;
    int bbox_count = 0;

    for ( int idx = 0; idx < num_tokens; ++idx ) {
        const float *probs = struct_probs + idx * num_classes;

        /* Argmax */
        int char_idx = 0;
        float max_prob = probs[0];
        for ( int c = 1; c < num_classes; ++c ) {
            if ( probs[c] > max_prob ) {
                max_prob = probs[c];
                char_idx = c;
            }
        }

        /* Stop at eos (if not first token) */
        if ( idx > 0 && char_idx == end_idx_ )
            break;

        /* Skip beg/end tokens */
        if ( char_idx == beg_idx_ || char_idx == end_idx_ )
            continue;

        /* Get character text */
        std::string text;
        if ( char_idx >= 0 && char_idx < static_cast<int>( characters_.size() ) ) {
            text = characters_[char_idx];
        } else {
            continue;
        }

        /* Decode bbox if this is a td token */
        if ( is_td_token( text ) ) {
            const float *bbox = bbox_preds + idx * 8;
            /* bbox is normalized [0,1], scale to original image */
            std::vector<float> cell( 8 );
            for ( int k = 0; k < 8; ++k ) {
                if ( k % 2 == 0 ) {
                    cell[k] = bbox[k] * ori_w; /* x coords */
                } else {
                    cell[k] = bbox[k] * ori_h; /* y coords */
                }
            }
            cell_bboxes.push_back( std::move( cell ) );
            ++bbox_count;
        }

        structure_list.push_back( text );
        scores.push_back( max_prob );
    }

    /* Compute mean score */
    if ( !scores.empty() ) {
        float sum = 0.0f;
        for ( float s : scores )
            sum += s;
        mean_score = sum / scores.size();
    }
}

std::vector<std::vector<float>> TableRecognizer::decode_logic_points(
    const std::vector<std::string> &structure_list ) {

    std::vector<std::vector<float>> logic_points;
    int current_row = 0;
    int current_col = 0;

    /* Track occupied cells for rowspan/colspan */
    std::set<std::pair<int, int>> occupied;

    auto is_occupied = [&]( int row, int col ) {
        return occupied.count( { row, col } ) > 0;
    };
    auto mark_occupied = [&]( int row, int col, int rowspan, int colspan ) {
        for ( int r = row; r < row + rowspan; ++r ) {
            for ( int c = col; c < col + colspan; ++c ) {
                occupied.insert( { r, c } );
            }
        }
    };

    for ( size_t i = 0; i < structure_list.size(); ++i ) {
        const std::string &token = structure_list[i];

        if ( token == "<tr>" ) {
            current_col = 0;
        } else if ( token == "</tr>" ) {
            current_row++;
        } else if ( token.rfind( "<td", 0 ) == 0 ) {
            int colspan = 1, rowspan = 1;

            if ( token != "<td></td>" ) {
                /* Parse attributes from subsequent tokens until ">" */
                size_t j = i + 1;
                while ( j < structure_list.size() && structure_list[j].rfind( ">", 0 ) != 0 ) {
                    const std::string &attr = structure_list[j];
                    if ( attr.find( "colspan=" ) != std::string::npos ) {
                        /* Extract number */
                        size_t eq = attr.find( '=' );
                        if ( eq != std::string::npos ) {
                            colspan = std::atoi( attr.c_str() + eq + 1 );
                        }
                    } else if ( attr.find( "rowspan=" ) != std::string::npos ) {
                        size_t eq = attr.find( '=' );
                        if ( eq != std::string::npos ) {
                            rowspan = std::atoi( attr.c_str() + eq + 1 );
                        }
                    }
                    ++j;
                }
                i = j;
            }

            /* Find next unoccupied column */
            while ( is_occupied( current_row, current_col ) ) {
                current_col++;
            }

            int r_start = current_row;
            int r_end = current_row + rowspan - 1;
            int col_start = current_col;
            int col_end = current_col + colspan - 1;

            logic_points.push_back( { static_cast<float>( r_start ),
                                      static_cast<float>( r_end ),
                                      static_cast<float>( col_start ),
                                      static_cast<float>( col_end ) } );

            mark_occupied( r_start, col_start, rowspan, colspan );
            current_col += colspan;
        }
    }

    return logic_points;
}

} // namespace ocr