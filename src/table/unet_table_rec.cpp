#include "unet_table_rec.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

namespace ocr {

constexpr float UnetTableRecognizer::MEAN[3];
constexpr float UnetTableRecognizer::STD[3];

UnetTableRecognizer::UnetTableRecognizer( const Config &cfg ) {
    session_ = std::make_unique<OnnxSession>( cfg.model_path, cfg.use_gpu, cfg.gpu_id );
}

UnetTableRecognizer::~UnetTableRecognizer() = default;

cv::Mat UnetTableRecognizer::resize_img( const cv::Mat &img, int target_h, int target_w ) {
    int h = img.rows, w = img.cols;
    float scale_h = static_cast<float>( target_h ) / h;
    float scale_w = static_cast<float>( target_w ) / w;
    float scale = std::min( scale_h, scale_w );

    int new_h = static_cast<int>( h * scale );
    int new_w = static_cast<int>( w * scale );

    cv::Mat resized;
    int interp = ( scale < 1.0f ) ? cv::INTER_AREA : cv::INTER_CUBIC;
    cv::resize( img, resized, cv::Size( new_w, new_h ), 0, 0, interp );

    cv::Mat padded = cv::Mat::zeros( target_h, target_w, img.type() );
    int y_off = ( target_h - new_h ) / 2;
    int x_off = ( target_w - new_w ) / 2;
    cv::Mat roi( padded, cv::Rect( x_off, y_off, new_w, new_h ) );
    resized.copyTo( roi );

    return padded;
}

cv::Mat UnetTableRecognizer::preprocess( const cv::Mat &img ) {
    int h = img.rows, w = img.cols;
    float scale_h = static_cast<float>( INP_H ) / h;
    float scale_w = static_cast<float>( INP_W ) / w;
    float scale = std::min( scale_h, scale_w );

    int new_h = static_cast<int>( h * scale + 0.5f );
    int new_w = static_cast<int>( w * scale + 0.5f );

    cv::Mat resized;
    int interp = ( scale < 1.0f ) ? cv::INTER_AREA : cv::INTER_CUBIC;
    cv::resize( img, resized, cv::Size( new_w, new_h ), 0, 0, interp );

    cv::Mat rgb;
    cv::cvtColor( resized, rgb, cv::COLOR_BGR2RGB );

    cv::Mat flt;
    rgb.convertTo( flt, CV_32F );

    cv::subtract( flt, cv::Scalar( MEAN[0], MEAN[1], MEAN[2] ), flt );
    cv::divide( flt, cv::Scalar( STD[0], STD[1], STD[2] ), flt );

    return flt;
}

cv::Mat UnetTableRecognizer::infer( const cv::Mat &preprocessed ) {
    int ph = preprocessed.rows;
    int pw = preprocessed.cols;

    std::vector<float> input_data( 3 * ph * pw );
    std::vector<cv::Mat> channels( 3 );
    cv::split( preprocessed, channels );
    for ( int c = 0; c < 3; ++c ) {
        std::memcpy( input_data.data() + c * ph * pw,
                     channels[c].data, ph * pw * sizeof( float ) );
    }

    std::vector<int64_t> input_shape = { 1, 3, ph, pw };
    auto outputs = session_->run( input_shape, input_data );

    if ( outputs.empty() )
        return cv::Mat();

    auto &out = outputs[0];
    auto info = out.GetTensorTypeAndShapeInfo();
    auto shape = info.GetShape();
    auto elem_type = info.GetElementType();

    const float *data_f = nullptr;
    const int64_t *data_i64 = nullptr;
    bool is_int64 = ( elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 );

    if ( is_int64 ) {
        data_i64 = out.GetTensorData<int64_t>();
    } else {
        data_f = out.GetTensorData<float>();
    }

    int out_h = static_cast<int>( shape[2] );
    int out_w = static_cast<int>( shape[3] );

    cv::Mat pred( out_h, out_w, CV_8UC1 );
    for ( int i = 0; i < out_h * out_w; ++i ) {
        int cls;
        if ( is_int64 ) {
            cls = static_cast<int>( data_i64[i] );
        } else {
            cls = static_cast<int>( std::round( data_f[i] ) );
        }
        pred.data[i] = static_cast<uchar>( std::clamp( cls, 0, 255 ) );
    }

    return pred;
}

/* ---- get_table_line: match Python skimage.measure.label + regionprops ---- */
std::vector<cv::Vec4f> UnetTableRecognizer::get_table_line(
    const cv::Mat &mask, int axis, int lineW ) {

    std::vector<cv::Vec4f> lines;

    cv::Mat bin;
    cv::threshold( mask, bin, 0, 255, cv::THRESH_BINARY );

    cv::Mat labels, stats, centroids;
    int n = cv::connectedComponentsWithStats( bin, labels, stats, centroids, 8 );

    for ( int i = 1; i < n; ++i ) {
        int bx = stats.at<int>( i, cv::CC_STAT_LEFT );
        int by = stats.at<int>( i, cv::CC_STAT_TOP );
        int bw = stats.at<int>( i, cv::CC_STAT_WIDTH );
        int bh = stats.at<int>( i, cv::CC_STAT_HEIGHT );

        if ( axis == 0 ) {
            if ( bw <= lineW )
                continue;
        } else {
            if ( bh <= lineW )
                continue;
        }

        cv::Mat component_mask = ( labels == i );
        std::vector<cv::Point> pts_vec;
        cv::findNonZero( component_mask, pts_vec );
        if ( pts_vec.size() < 2 )
            continue;

        cv::RotatedRect rrect = cv::minAreaRect( pts_vec );
        cv::Point2f box_pts[4];
        rrect.points( box_pts );

        std::vector<cv::Point2f> sorted4( 4 );
        for ( int k = 0; k < 4; ++k )
            sorted4[k] = box_pts[k];

        std::sort( sorted4.begin(), sorted4.end(),
                   []( const cv::Point2f &a, const cv::Point2f &b ) {
                       return a.x < b.x;
                   } );

        cv::Point2f left0 = sorted4[0], left1 = sorted4[1];
        cv::Point2f right0 = sorted4[2], right1 = sorted4[3];

        if ( left0.y > left1.y )
            std::swap( left0, left1 );
        cv::Point2f tl = left0, bl = left1;

        float d0 = std::hypot( tl.x - right0.x, tl.y - right0.y );
        float d1 = std::hypot( tl.x - right1.x, tl.y - right1.y );
        cv::Point2f br = ( d0 > d1 ) ? right0 : right1;
        cv::Point2f tr = ( d0 > d1 ) ? right1 : right0;

        float cx = ( tl.x + tr.x + br.x + bl.x ) / 4.0f;
        float cy = ( tl.y + tr.y + br.y + bl.y ) / 4.0f;
        float w = ( std::hypot( tr.x - tl.x, tr.y - tl.y ) +
                    std::hypot( br.x - bl.x, br.y - bl.y ) ) /
                  2.0f;
        float h = ( std::hypot( bl.x - tl.x, bl.y - tl.y ) +
                    std::hypot( br.x - tr.x, br.y - tr.y ) ) /
                  2.0f;

        float xmin, ymin, xmax, ymax;
        if ( w < h ) {
            xmin = ( tl.x + tr.x ) / 2.0f;
            xmax = ( bl.x + br.x ) / 2.0f;
            ymin = ( tl.y + tr.y ) / 2.0f;
            ymax = ( bl.y + br.y ) / 2.0f;
        } else {
            xmin = ( tl.x + bl.x ) / 2.0f;
            xmax = ( tr.x + br.x ) / 2.0f;
            ymin = ( tl.y + bl.y ) / 2.0f;
            ymax = ( tr.y + br.y ) / 2.0f;
        }

        lines.push_back( cv::Vec4f( xmin, ymin, xmax, ymax ) );
    }

    return lines;
}

/* ---- adjust_lines: find additional lines by proximity of endpoints ---- */
std::vector<cv::Vec4f> UnetTableRecognizer::adjust_lines(
    const std::vector<cv::Vec4f> &lines, int alph, int angle ) {

    std::vector<cv::Vec4f> new_lines;
    int n = static_cast<int>( lines.size() );

    for ( int i = 0; i < n; ++i ) {
        float x1 = lines[i][0], y1 = lines[i][1];
        float x2 = lines[i][2], y2 = lines[i][3];
        float cx1 = ( x1 + x2 ) / 2.0f, cy1 = ( y1 + y2 ) / 2.0f;

        for ( int j = 0; j < n; ++j ) {
            if ( i == j )
                continue;
            float x3 = lines[j][0], y3 = lines[j][1];
            float x4 = lines[j][2], y4 = lines[j][3];
            float cx2 = ( x3 + x4 ) / 2.0f, cy2 = ( y3 + y4 ) / 2.0f;

            bool overlap = ( ( x3 < cx1 && cx1 < x4 ) || ( y3 < cy1 && cy1 < y4 ) ) ||
                           ( ( x1 < cx2 && cx2 < x2 ) || ( y1 < cy2 && cy2 < y2 ) );
            if ( overlap )
                continue;

            auto try_add = [&]( float px1, float py1, float px2, float py2 ) {
                float r = std::hypot( px1 - px2, py1 - py2 );
                float k = std::abs( ( py2 - py1 ) / ( px2 - px1 + 1e-10f ) );
                float a = std::atan( k ) * 180.0f / static_cast<float>( CV_PI );
                if ( r < alph && a < angle ) {
                    new_lines.push_back( cv::Vec4f( px1, py1, px2, py2 ) );
                }
            };

            try_add( x1, y1, x3, y3 );
            try_add( x1, y1, x4, y4 );
            try_add( x2, y2, x3, y3 );
            try_add( x2, y2, x4, y4 );
        }
    }

    return new_lines;
}

/* ---- line_to_line: extend line to intersection with another line ---- */
static cv::Vec4f line_to_line( cv::Vec4f pts1, const cv::Vec4f &pts2,
                               float alpha = 20.0f, float angle = 30.0f ) {
    float x1 = pts1[0], y1 = pts1[1], x2 = pts1[2], y2 = pts1[3];
    float ox1 = pts2[0], oy1 = pts2[1], ox2 = pts2[2], oy2 = pts2[3];

    float A1 = y2 - y1, B1 = x1 - x2, C1 = x2 * y1 - x1 * y2;
    float A2 = oy2 - oy1, B2 = ox1 - ox2, C2 = ox2 * oy1 - ox1 * oy2;

    float flag1 = A2 * x1 + B2 * y1 + C2;
    float flag2 = A2 * x2 + B2 * y2 + C2;

    if ( ( flag1 > 0 && flag2 > 0 ) || ( flag1 < 0 && flag2 < 0 ) ) {
        float denom = A1 * B2 - A2 * B1;
        if ( std::fabs( denom ) > 1e-6f ) {
            float px = ( B1 * C2 - B2 * C1 ) / denom;
            float py = ( A2 * C1 - A1 * C2 ) / denom;

            float r0 = std::hypot( px - x1, py - y1 );
            float r1 = std::hypot( px - x2, py - y2 );

            if ( std::min( r0, r1 ) < alpha ) {
                if ( r0 < r1 ) {
                    float k = std::abs( ( y2 - py ) / ( x2 - px + 1e-10f ) );
                    float a = std::atan( k ) * 180.0f / static_cast<float>( CV_PI );
                    if ( a < angle || std::fabs( 90.0f - a ) < angle )
                        pts1 = cv::Vec4f( px, py, x2, y2 );
                } else {
                    float k = std::abs( ( y1 - py ) / ( x1 - px + 1e-10f ) );
                    float a = std::atan( k ) * 180.0f / static_cast<float>( CV_PI );
                    if ( a < angle || std::fabs( 90.0f - a ) < angle )
                        pts1 = cv::Vec4f( x1, y1, px, py );
                }
            }
        }
    }
    return pts1;
}

/* ---- final_adjust_lines: extend lines to intersection points ---- */
void UnetTableRecognizer::final_adjust_lines(
    std::vector<cv::Vec4f> &rowboxes, std::vector<cv::Vec4f> &colboxes ) {

    int nrow = static_cast<int>( rowboxes.size() );
    int ncol = static_cast<int>( colboxes.size() );

    for ( int i = 0; i < nrow; ++i ) {
        for ( int j = 0; j < ncol; ++j ) {
            rowboxes[i] = line_to_line( rowboxes[i], colboxes[j], 20.0f, 30.0f );
            colboxes[j] = line_to_line( colboxes[j], rowboxes[i], 20.0f, 30.0f );
        }
    }
}

/* ---- extract_cells: connected components on inverse of line image ---- */
std::vector<std::vector<cv::Point2f>> UnetTableRecognizer::extract_cells(
    const cv::Mat &line_img, int img_w, int img_h ) {

    cv::Mat inv;
    cv::bitwise_not( line_img, inv );

    cv::Mat labels, stats, centroids;
    int n = cv::connectedComponentsWithStats( inv, labels, stats, centroids, 8 );

    std::vector<std::vector<cv::Point2f>> cells;
    float half_area = 0.5f * static_cast<float>( img_w ) * static_cast<float>( img_h );

    for ( int i = 1; i < n; ++i ) {
        int x = stats.at<int>( i, cv::CC_STAT_LEFT );
        int y = stats.at<int>( i, cv::CC_STAT_TOP );
        int w = stats.at<int>( i, cv::CC_STAT_WIDTH );
        int h = stats.at<int>( i, cv::CC_STAT_HEIGHT );

        cv::Mat component_mask = ( labels == i );
        std::vector<cv::Point> pts_vec;
        cv::findNonZero( component_mask, pts_vec );
        if ( pts_vec.size() < 2 )
            continue;

        cv::RotatedRect rrect = cv::minAreaRect( pts_vec );
        float rw = rrect.size.width;
        float rh = rrect.size.height;
        float rect_area = rw * rh;

        if ( rect_area >= half_area )
            continue;
        if ( rw < 15.0f || rh < 15.0f )
            continue;

        cv::Point2f verts[4];
        rrect.points( verts );

        std::sort( verts, verts + 4,
                   []( const cv::Point2f &a, const cv::Point2f &b ) {
                       return a.x < b.x;
                   } );

        cv::Point2f left0 = verts[0], left1 = verts[1];
        cv::Point2f right0 = verts[2], right1 = verts[3];
        if ( left0.y > left1.y )
            std::swap( left0, left1 );
        cv::Point2f tl = left0, bl = left1;
        float d0 = std::hypot( tl.x - right0.x, tl.y - right0.y );
        float d1 = std::hypot( tl.x - right1.x, tl.y - right1.y );
        cv::Point2f br = ( d0 > d1 ) ? right0 : right1;
        cv::Point2f tr = ( d0 > d1 ) ? right1 : right0;

        cells.push_back( { tl, tr, br, bl } );
    }

    return cells;
}

void UnetTableRecognizer::sort_polygons(
    std::vector<std::vector<cv::Point2f>> &polygons ) {

    std::sort( polygons.begin(), polygons.end(),
               []( const std::vector<cv::Point2f> &a, const std::vector<cv::Point2f> &b ) {
                   float ay = ( a[0].y + a[1].y + a[2].y + a[3].y ) / 4.0f;
                   float by = ( b[0].y + b[1].y + b[2].y + b[3].y ) / 4.0f;
                   float ax = ( a[0].x + a[1].x + a[2].x + a[3].x ) / 4.0f;
                   float bx = ( b[0].x + b[1].x + b[2].x + b[3].x ) / 4.0f;
                   if ( std::fabs( ay - by ) > 10.0f )
                       return ay < by;
                   return ax < bx;
               } );
}

std::vector<std::vector<float>> UnetTableRecognizer::recover_logic_points(
    const std::vector<std::vector<cv::Point2f>> &sorted_polygons,
    int row_thresh, int col_thresh ) {

    int n = static_cast<int>( sorted_polygons.size() );
    if ( n == 0 )
        return {};

    std::vector<float> cy( n ), cx( n );
    for ( int i = 0; i < n; ++i ) {
        cy[i] = ( sorted_polygons[i][0].y + sorted_polygons[i][1].y +
                  sorted_polygons[i][2].y + sorted_polygons[i][3].y ) /
                4.0f;
        cx[i] = ( sorted_polygons[i][0].x + sorted_polygons[i][1].x +
                  sorted_polygons[i][2].x + sorted_polygons[i][3].x ) /
                4.0f;
    }

    std::vector<int> row_id( n );
    int cur_row = 0;
    row_id[0] = 0;
    for ( int i = 1; i < n; ++i ) {
        if ( cy[i] - cy[i - 1] > row_thresh )
            ++cur_row;
        row_id[i] = cur_row;
    }
    int num_rows = cur_row + 1;

    std::vector<std::vector<int>> row_groups( num_rows );
    for ( int i = 0; i < n; ++i )
        row_groups[row_id[i]].push_back( i );

    std::vector<int> longest_row;
    for ( auto &rg : row_groups )
        if ( rg.size() > longest_row.size() )
            longest_row = rg;

    std::vector<float> col_boundaries;
    for ( int idx : longest_row ) {
        float left = std::min( { sorted_polygons[idx][0].x, sorted_polygons[idx][1].x,
                                 sorted_polygons[idx][2].x, sorted_polygons[idx][3].x } );
        col_boundaries.push_back( left );
    }
    std::sort( col_boundaries.begin(), col_boundaries.end() );
    int num_cols = static_cast<int>( col_boundaries.size() );

    auto find_col = [&]( float x_center ) -> int {
        int best = 0;
        float best_dist = std::fabs( x_center - col_boundaries[0] );
        for ( int c = 1; c < num_cols; ++c ) {
            float d = std::fabs( x_center - col_boundaries[c] );
            if ( d < best_dist ) {
                best_dist = d;
                best = c;
            }
        }
        return best;
    };

    std::vector<std::vector<float>> logic_points( n );
    for ( int i = 0; i < n; ++i ) {
        int r = row_id[i];
        int c = find_col( cx[i] );
        logic_points[i] = { static_cast<float>( r ),
                            static_cast<float>( r ),
                            static_cast<float>( c ),
                            static_cast<float>( c ) };
    }

    return logic_points;
}

UnetTableResult UnetTableRecognizer::postprocess(
    const cv::Mat &img, const cv::Mat &pred ) {

    UnetTableResult result;
    if ( pred.empty() )
        return result;

    int ori_h = img.rows;
    int ori_w = img.cols;

    cv::Mat hpred = pred.clone();
    cv::Mat vpred = pred.clone();

    for ( int y = 0; y < pred.rows; ++y ) {
        for ( int x = 0; x < pred.cols; ++x ) {
            uchar v = pred.at<uchar>( y, x );
            if ( v == 2 ) {
                hpred.at<uchar>( y, x ) = 0;
            } else if ( v == 1 ) {
                vpred.at<uchar>( y, x ) = 0;
            } else {
                hpred.at<uchar>( y, x ) = 0;
                vpred.at<uchar>( y, x ) = 0;
            }
        }
    }

    cv::resize( hpred, hpred, cv::Size( ori_w, ori_h ), 0, 0, cv::INTER_NEAREST );
    cv::resize( vpred, vpred, cv::Size( ori_w, ori_h ), 0, 0, cv::INTER_NEAREST );

    int h = pred.rows, w = pred.cols;
    int hors_k = static_cast<int>( std::sqrt( static_cast<double>( w ) ) * 1.2 );
    int vert_k = static_cast<int>( std::sqrt( static_cast<double>( h ) ) * 1.2 );

    cv::Mat hkernel = cv::getStructuringElement( cv::MORPH_RECT, cv::Size( hors_k, 1 ) );
    cv::Mat vkernel = cv::getStructuringElement( cv::MORPH_RECT, cv::Size( 1, vert_k ) );

    cv::morphologyEx( vpred, vpred, cv::MORPH_CLOSE, vkernel );
    cv::morphologyEx( hpred, hpred, cv::MORPH_CLOSE, hkernel );

    auto rowboxes = get_table_line( hpred, 0, 50 );
    auto colboxes = get_table_line( vpred, 1, 30 );

    auto extra_rows = adjust_lines( rowboxes, 100, 50 );
    auto extra_cols = adjust_lines( colboxes, 15, 50 );
    rowboxes.insert( rowboxes.end(), extra_rows.begin(), extra_rows.end() );
    colboxes.insert( colboxes.end(), extra_cols.begin(), extra_cols.end() );

    final_adjust_lines( rowboxes, colboxes );

    cv::Mat line_img = cv::Mat::zeros( ori_h, ori_w, CV_8UC1 );
    for ( auto &l : rowboxes ) {
        cv::line( line_img,
                  cv::Point( static_cast<int>( l[0] ), static_cast<int>( l[1] ) ),
                  cv::Point( static_cast<int>( l[2] ), static_cast<int>( l[3] ) ),
                  255, 2, cv::LINE_8 );
    }
    for ( auto &l : colboxes ) {
        cv::line( line_img,
                  cv::Point( static_cast<int>( l[0] ), static_cast<int>( l[1] ) ),
                  cv::Point( static_cast<int>( l[2] ), static_cast<int>( l[3] ) ),
                  255, 2, cv::LINE_8 );
    }

    auto cells = extract_cells( line_img, ori_w, ori_h );
    if ( cells.empty() )
        return result;

    sort_polygons( cells );

    for ( auto &quad : cells ) {
        std::vector<float> bbox( 8 );
        bbox[0] = quad[0].x;
        bbox[1] = quad[0].y;
        bbox[2] = quad[1].x;
        bbox[3] = quad[1].y;
        bbox[4] = quad[2].x;
        bbox[5] = quad[2].y;
        bbox[6] = quad[3].x;
        bbox[7] = quad[3].y;
        result.cell_bboxes.push_back( std::move( bbox ) );
    }

    result.logic_points = recover_logic_points( cells );

    return result;
}

UnetTableResult UnetTableRecognizer::recognize( const cv::Mat &img ) {
    UnetTableResult result;
    if ( img.empty() )
        return result;

    cv::Mat preprocessed = preprocess( img );
    cv::Mat pred = infer( preprocessed );
    return postprocess( img, pred );
}

} // namespace ocr