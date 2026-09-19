#include "ocr.h"

#include "text_system.h"
#include "license_plate.h"
#include "table_recognizer.h"
#include "doclayout_yolo_analyzer.h"
#include "utils.h"

#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

/* Opaque handle wraps a TextSystem instance. */
struct OcrHandle {
    std::unique_ptr<ocr::TextSystem> system;
};

/* Opaque handle wraps a LicensePlateRecognizer instance. */
struct PlateHandle {
    std::unique_ptr<ocr::LicensePlateRecognizer> recognizer;
};

/* Opaque handle wraps a TableRecognizer instance. */
struct TableHandle {
    std::unique_ptr<ocr::TableRecognizer> recognizer;
};

/* Opaque handle wraps a DocLayoutYOLOAnalyzer instance. */
struct DocLayoutYOLOHandle {
    std::unique_ptr<ocr::DocLayoutYOLOAnalyzer> analyzer;
};

/* ---- Error handling ---- */

static thread_local std::string g_error;

static void set_error( const std::string &msg ) {
    g_error = msg;
    ocr::set_last_error( msg );
}

OCR_API const char *last_error( void ) {
    if ( g_error.empty() )
        return nullptr;
    return g_error.c_str();
}

/* ---- Config conversion ---- */

static ocr::TextSystemConfig make_config( const OcrConfig *cfg ) {
    ocr::TextSystemConfig ts_cfg;

    // Detection
    ts_cfg.det_cfg.model_path = cfg->det_model_path ? cfg->det_model_path : "";
    ts_cfg.det_cfg.use_gpu = cfg->use_gpu != 0;
    ts_cfg.det_cfg.gpu_id = cfg->gpu_id;
    ts_cfg.det_cfg.det_limit_side_len = cfg->det_limit_side_len > 0 ? cfg->det_limit_side_len : 960.0f;
    ts_cfg.det_cfg.det_limit_type = cfg->det_limit_type ? cfg->det_limit_type : "max";
    ts_cfg.det_cfg.det_db_thresh = cfg->det_db_thresh;
    ts_cfg.det_cfg.det_db_box_thresh = cfg->det_db_box_thresh;
    ts_cfg.det_cfg.det_db_unclip_ratio = cfg->det_db_unclip_ratio;
    ts_cfg.det_cfg.use_dilation = cfg->use_dilation != 0;
    ts_cfg.det_cfg.det_db_score_mode = cfg->det_db_score_mode ? cfg->det_db_score_mode : "fast";
    ts_cfg.det_cfg.det_box_type = cfg->det_box_type ? cfg->det_box_type : "quad";

    // Recognition
    ts_cfg.rec_cfg.model_path = cfg->rec_model_path ? cfg->rec_model_path : "";
    ts_cfg.rec_cfg.char_dict_path = cfg->rec_char_dict_path ? cfg->rec_char_dict_path : "";
    ts_cfg.rec_cfg.use_space_char = cfg->use_space_char != 0;
    ts_cfg.rec_cfg.use_gpu = cfg->use_gpu != 0;
    ts_cfg.rec_cfg.gpu_id = cfg->gpu_id;
    ts_cfg.rec_cfg.rec_batch_num = cfg->rec_batch_num > 0 ? cfg->rec_batch_num : 6;
    ts_cfg.rec_cfg.rec_image_c = cfg->rec_image_c > 0 ? cfg->rec_image_c : 3;
    ts_cfg.rec_cfg.rec_image_h = cfg->rec_image_h > 0 ? cfg->rec_image_h : 48;
    ts_cfg.rec_cfg.rec_image_w = cfg->rec_image_w > 0 ? cfg->rec_image_w : 320;
    ts_cfg.rec_cfg.drop_score = cfg->drop_score;
    ts_cfg.drop_score = cfg->drop_score > 0 ? cfg->drop_score : 0.5f;

    // Classification / Orientation
    if ( cfg->cls_model_path && cfg->cls_model_path[0] != '\0' ) {
        ts_cfg.use_angle_cls = true;
        bool use_orientation = ( cfg->cls_model_type == CLS_MODEL_ORIENTATION );
        ts_cfg.use_rapid_orientation = use_orientation;
        if ( use_orientation ) {
            ts_cfg.orientation_cfg.model_path = cfg->cls_model_path;
            ts_cfg.orientation_cfg.use_gpu = cfg->use_gpu != 0;
            ts_cfg.orientation_cfg.gpu_id = cfg->gpu_id;
            ts_cfg.orientation_cfg.batch_num = cfg->cls_batch_num > 0 ? cfg->cls_batch_num : 6;
            ts_cfg.orientation_cfg.cls_thresh = cfg->cls_thresh > 0 ? cfg->cls_thresh : 0.9f;
        } else {
            ts_cfg.cls_cfg.model_path = cfg->cls_model_path;
            ts_cfg.cls_cfg.use_gpu = cfg->use_gpu != 0;
            ts_cfg.cls_cfg.gpu_id = cfg->gpu_id;
            ts_cfg.cls_cfg.cls_batch_num = cfg->cls_batch_num > 0 ? cfg->cls_batch_num : 6;
            ts_cfg.cls_cfg.cls_image_c = cfg->cls_image_c > 0 ? cfg->cls_image_c : 3;
            ts_cfg.cls_cfg.cls_image_h = cfg->cls_image_h > 0 ? cfg->cls_image_h : 48;
            ts_cfg.cls_cfg.cls_image_w = cfg->cls_image_w > 0 ? cfg->cls_image_w : 192;
            ts_cfg.cls_cfg.cls_thresh = cfg->cls_thresh;
        }
    } else {
        ts_cfg.use_angle_cls = false;
        ts_cfg.use_rapid_orientation = false;
    }

    return ts_cfg;
}

/* ---- C API implementation ---- */

OCR_API OcrHandle *ocr_create( const OcrConfig *config ) {
    g_error.clear();
    if ( !config ) {
        set_error( "config is NULL" );
        return nullptr;
    }
    if ( !config->det_model_path || !config->rec_model_path || !config->rec_char_dict_path ) {
        set_error( "det_model_path, rec_model_path, and rec_char_dict_path are required" );
        return nullptr;
    }

    try {
        auto cfg = make_config( config );
        auto handle = std::make_unique<OcrHandle>();
        handle->system = std::make_unique<ocr::TextSystem>( cfg );
        return handle.release();
    } catch ( const std::exception &e ) {
        set_error( e.what() );
        return nullptr;
    } catch ( ... ) {
        set_error( "unknown error during OCR engine creation" );
        return nullptr;
    }
}

OCR_API void ocr_destroy( OcrHandle *handle ) {
    delete handle;
}

static cv::Mat crop_image( const cv::Mat &img, int x1, int y1, int x2, int y2 ) {
    int cx1 = ( x1 < 0 ) ? 0 : std::max( 0, x1 );
    int cy1 = ( y1 < 0 ) ? 0 : std::max( 0, y1 );
    int cx2 = ( x2 < 0 ) ? img.cols : std::min( img.cols, x2 );
    int cy2 = ( y2 < 0 ) ? img.rows : std::min( img.rows, y2 );
    if ( cx2 <= cx1 || cy2 <= cy1 )
        return cv::Mat();
    return img( cv::Rect( cx1, cy1, cx2 - cx1, cy2 - cy1 ) ).clone();
}

static int run_ocr_impl( OcrHandle *handle, const cv::Mat &img, OcrResultList *out_results ) {
    if ( !handle || !handle->system || !out_results ) {
        set_error( "invalid arguments to ocr_run" );
        return -1;
    }

    if ( img.empty() ) {
        set_error( "image is empty" );
        return -1;
    }

    try {
        auto results = handle->system->run( img, true );

        // Allocate result array
        out_results->count = static_cast<int>( results.size() );
        out_results->items = nullptr;

        if ( out_results->count > 0 ) {
            out_results->items = static_cast<OcrResult *>(
                std::calloc( out_results->count, sizeof( OcrResult ) ) );
            if ( !out_results->items ) {
                set_error( "memory allocation failed" );
                return -1;
            }

            for ( int i = 0; i < out_results->count; ++i ) {
                const auto &r = results[i];
                // Copy box points
                for ( int k = 0; k < 4; ++k ) {
                    out_results->items[i].box[k].x = r.box.pts[k].x;
                    out_results->items[i].box[k].y = r.box.pts[k].y;
                }
                // Copy text (using strdup so caller can free with ocr_free_results)
                out_results->items[i].text = _strdup( r.text.c_str() );
                out_results->items[i].score = r.score;
            }
        }

        return 0;
    } catch ( const std::exception &e ) {
        set_error( e.what() );
        return -1;
    } catch ( ... ) {
        set_error( "unknown error during OCR" );
        return -1;
    }
}

OCR_API int ocr_run( OcrHandle *handle,
                     const unsigned char *image_data,
                     int width, int height,
                     int crop_x1, int crop_y1, int crop_x2, int crop_y2,
                     OcrResultList *out_results ) {
    g_error.clear();
    if ( !image_data || width <= 0 || height <= 0 ) {
        set_error( "invalid image dimensions or data" );
        return -1;
    }

    cv::Mat img( height, width, CV_8UC3, const_cast<unsigned char *>( image_data ) );
    img = img.clone();

    cv::Mat cropped = crop_image( img, crop_x1, crop_y1, crop_x2, crop_y2 );
    if ( cropped.empty() ) {
        set_error( "crop region is empty" );
        return -1;
    }

    return run_ocr_impl( handle, cropped, out_results );
}

OCR_API int ocr_run_file( OcrHandle *handle,
                          const char *image_path,
                          int crop_x1, int crop_y1, int crop_x2, int crop_y2,
                          OcrResultList *out_results ) {
    g_error.clear();
    if ( !image_path ) {
        set_error( "image_path is NULL" );
        return -1;
    }

    cv::Mat img = cv::imread( image_path, cv::IMREAD_COLOR );
    if ( img.empty() ) {
        set_error( std::string( "failed to read image: " ) + image_path );
        return -1;
    }

    cv::Mat cropped = crop_image( img, crop_x1, crop_y1, crop_x2, crop_y2 );
    if ( cropped.empty() ) {
        set_error( "crop region is empty" );
        return -1;
    }

    return run_ocr_impl( handle, cropped, out_results );
}

OCR_API void ocr_free_results( OcrResultList *results ) {
    if ( !results || !results->items )
        return;

    for ( int i = 0; i < results->count; ++i ) {
        if ( results->items[i].text ) {
            std::free( const_cast<char *>( results->items[i].text ) );
        }
    }
    std::free( results->items );
    results->items = nullptr;
    results->count = 0;
}

/* ===== License Plate Recognition API ===== */

OCR_API PlateHandle *plate_create( const PlateConfig *config ) {
    g_error.clear();
    if ( !config ) {
        set_error( "plate config is NULL" );
        return nullptr;
    }
    if ( !config->detect_model_path || !config->rec_model_path ) {
        set_error( "detect_model_path and rec_model_path are required" );
        return nullptr;
    }

    try {
        ocr::LicensePlateRecognizer::Config cfg;
        cfg.detect_model_path = config->detect_model_path;
        cfg.rec_model_path = config->rec_model_path;
        cfg.use_gpu = config->use_gpu != 0;
        cfg.gpu_id = config->gpu_id;
        cfg.min_score = config->min_score > 0 ? config->min_score : 0.4f;
        cfg.iou_thresh = config->iou_thresh > 0 ? config->iou_thresh : 0.5f;

        auto handle = std::make_unique<PlateHandle>();
        handle->recognizer = std::make_unique<ocr::LicensePlateRecognizer>( cfg );
        return handle.release();
    } catch ( const std::exception &e ) {
        set_error( e.what() );
        return nullptr;
    } catch ( ... ) {
        set_error( "unknown error during plate recognizer creation" );
        return nullptr;
    }
}

OCR_API void plate_destroy( PlateHandle *handle ) {
    delete handle;
}

static int run_plate_impl( PlateHandle *handle, const cv::Mat &img,
                           PlateResultList *out_results ) {
    if ( !handle || !handle->recognizer || !out_results ) {
        set_error( "invalid arguments to plate_run" );
        return -1;
    }
    if ( img.empty() ) {
        set_error( "image is empty" );
        return -1;
    }

    try {
        auto results = handle->recognizer->recognize( img );

        out_results->count = static_cast<int>( results.size() );
        out_results->items = nullptr;

        if ( out_results->count > 0 ) {
            out_results->items = static_cast<PlateResult *>(
                std::calloc( out_results->count, sizeof( PlateResult ) ) );
            if ( !out_results->items ) {
                set_error( "memory allocation failed" );
                return -1;
            }

            for ( int i = 0; i < out_results->count; ++i ) {
                const auto &r = results[i];
                out_results->items[i].box.x1 = r.box.x1;
                out_results->items[i].box.y1 = r.box.y1;
                out_results->items[i].box.x2 = r.box.x2;
                out_results->items[i].box.y2 = r.box.y2;
                out_results->items[i].score = r.score;
                out_results->items[i].plate = _strdup( r.plate.c_str() );
                out_results->items[i].type = _strdup( r.type.c_str() );
                for ( int k = 0; k < 4; ++k ) {
                    out_results->items[i].landmarks[k].x = r.landmarks[k].x;
                    out_results->items[i].landmarks[k].y = r.landmarks[k].y;
                }
            }
        }
        return 0;
    } catch ( const std::exception &e ) {
        set_error( e.what() );
        return -1;
    } catch ( ... ) {
        set_error( "unknown error during plate recognition" );
        return -1;
    }
}

OCR_API int plate_run( PlateHandle *handle,
                       const unsigned char *image_data,
                       int width, int height,
                       PlateResultList *out_results ) {
    g_error.clear();
    if ( !image_data || width <= 0 || height <= 0 ) {
        set_error( "invalid image dimensions or data" );
        return -1;
    }
    cv::Mat img( height, width, CV_8UC3, const_cast<unsigned char *>( image_data ) );
    img = img.clone();
    return run_plate_impl( handle, img, out_results );
}

OCR_API int plate_run_file( PlateHandle *handle,
                            const char *image_path,
                            PlateResultList *out_results ) {
    g_error.clear();
    if ( !image_path ) {
        set_error( "image_path is NULL" );
        return -1;
    }
    cv::Mat img = cv::imread( image_path, cv::IMREAD_COLOR );
    if ( img.empty() ) {
        set_error( std::string( "failed to read image: " ) + image_path );
        return -1;
    }
    return run_plate_impl( handle, img, out_results );
}

OCR_API void plate_free_results( PlateResultList *results ) {
    if ( !results || !results->items )
        return;
    for ( int i = 0; i < results->count; ++i ) {
        if ( results->items[i].plate ) {
            std::free( const_cast<char *>( results->items[i].plate ) );
        }
        if ( results->items[i].type ) {
            std::free( const_cast<char *>( results->items[i].type ) );
        }
    }
    std::free( results->items );
    results->items = nullptr;
    results->count = 0;
}

/* ===== Table Recognition API ===== */

static ocr::TableModelType to_table_model_type( int type ) {
    switch ( type ) {
    case TABLE_MODEL_UNET:
        return ocr::TableModelType::UNET;
    case TABLE_MODEL_UNET_SLANET_PLUS:
        return ocr::TableModelType::UNET_SLANET_PLUS;
    default:
        return ocr::TableModelType::SLANET_PLUS;
    }
}

OCR_API TableHandle *table_create( const TableConfig *config ) {
    g_error.clear();
    if ( !config ) {
        set_error( "table config is NULL" );
        return nullptr;
    }
    if ( !config->model_path ) {
        set_error( "model_path is required" );
        return nullptr;
    }

    try {
        ocr::TableRecognizer::Config cfg;
        cfg.model_path = config->model_path;
        cfg.model_type = to_table_model_type( config->model_type );
        cfg.use_gpu = config->use_gpu != 0;
        cfg.gpu_id = config->gpu_id;
        if ( config->cls_model_path )
            cfg.cls_model_path = config->cls_model_path;
        if ( config->unet_model_path )
            cfg.unet_model_path = config->unet_model_path;

        auto handle = std::make_unique<TableHandle>();
        handle->recognizer = std::make_unique<ocr::TableRecognizer>( cfg );
        return handle.release();
    } catch ( const std::exception &e ) {
        set_error( e.what() );
        return nullptr;
    } catch ( ... ) {
        set_error( "unknown error during table recognizer creation" );
        return nullptr;
    }
}

OCR_API void table_destroy( TableHandle *handle ) {
    delete handle;
}

static int run_table_impl( TableHandle *handle, const cv::Mat &img,
                           TableResult *out_result ) {
    if ( !handle || !handle->recognizer || !out_result ) {
        set_error( "invalid arguments to table_run" );
        return -1;
    }
    if ( img.empty() ) {
        set_error( "image is empty" );
        return -1;
    }

    try {
        auto result = handle->recognizer->recognize( img );

        out_result->score = result.score;
        out_result->cell_count = static_cast<int>( result.cell_bboxes.size() );
        out_result->cells = nullptr;
        out_result->logic_points = nullptr;

        if ( out_result->cell_count > 0 ) {
            out_result->cells = static_cast<TableCell *>(
                std::calloc( out_result->cell_count, sizeof( TableCell ) ) );
            if ( !out_result->cells ) {
                set_error( "memory allocation failed" );
                return -1;
            }
            for ( int i = 0; i < out_result->cell_count; ++i ) {
                for ( int k = 0; k < 4 && k * 2 + 1 < static_cast<int>( result.cell_bboxes[i].size() ); ++k ) {
                    out_result->cells[i].bbox[k].x = result.cell_bboxes[i][k * 2];
                    out_result->cells[i].bbox[k].y = result.cell_bboxes[i][k * 2 + 1];
                }
            }

            out_result->logic_points = static_cast<RectBox *>(
                std::calloc( out_result->cell_count, sizeof( RectBox ) ) );
            if ( !out_result->logic_points ) {
                std::free( out_result->cells );
                out_result->cells = nullptr;
                set_error( "memory allocation failed" );
                return -1;
            }
            for ( int i = 0; i < out_result->cell_count; ++i ) {
                if ( i < static_cast<int>( result.logic_points.size() ) &&
                     result.logic_points[i].size() >= 4 ) {
                    out_result->logic_points[i].x1 = static_cast<int>( result.logic_points[i][0] );
                    out_result->logic_points[i].y1 = static_cast<int>( result.logic_points[i][1] );
                    out_result->logic_points[i].x2 = static_cast<int>( result.logic_points[i][2] );
                    out_result->logic_points[i].y2 = static_cast<int>( result.logic_points[i][3] );
                }
            }
        }
        return 0;
    } catch ( const std::exception &e ) {
        set_error( e.what() );
        return -1;
    } catch ( ... ) {
        set_error( "unknown error during table recognition" );
        return -1;
    }
}

OCR_API int table_run( TableHandle *handle,
                       const unsigned char *image_data,
                       int width, int height,
                       TableResult *out_result ) {
    g_error.clear();
    if ( !image_data || width <= 0 || height <= 0 ) {
        set_error( "invalid image dimensions or data" );
        return -1;
    }
    cv::Mat img( height, width, CV_8UC3, const_cast<unsigned char *>( image_data ) );
    img = img.clone();
    return run_table_impl( handle, img, out_result );
}

OCR_API int table_run_file( TableHandle *handle,
                            const char *image_path,
                            TableResult *out_result ) {
    g_error.clear();
    if ( !image_path ) {
        set_error( "image_path is NULL" );
        return -1;
    }
    cv::Mat img = cv::imread( image_path, cv::IMREAD_COLOR );
    if ( img.empty() ) {
        set_error( std::string( "failed to read image: " ) + image_path );
        return -1;
    }
    return run_table_impl( handle, img, out_result );
}

OCR_API void table_free_result( TableResult *result ) {
    if ( !result )
        return;
    if ( result->cells ) {
        std::free( result->cells );
        result->cells = nullptr;
    }
    if ( result->logic_points ) {
        std::free( result->logic_points );
        result->logic_points = nullptr;
    }
    result->cell_count = 0;
}

/* ===== Layout Analysis Unified API ===== */

static ocr::DocLayoutYOLOModelType to_doclayout_yolo_model_type( int type ) {
    switch ( type ) {
    case PP_LAYOUT_CDLA:
        return ocr::DocLayoutYOLOModelType::PP_LAYOUT_CDLA;
    case PP_LAYOUT_PUBLAYNET:
        return ocr::DocLayoutYOLOModelType::PP_LAYOUT_PUBLAYNET;
    case PP_DOCLAYOUT_V2:
        return ocr::DocLayoutYOLOModelType::PP_DOCLAYOUT_V2;
    case YOLO_LAYOUT_PAPER:
        return ocr::DocLayoutYOLOModelType::YOLOV8_PAPER;
    case YOLO_LAYOUT_REPORT:
        return ocr::DocLayoutYOLOModelType::YOLOV8_REPORT;
    case YOLO_LAYOUT_PUBLAYNET:
        return ocr::DocLayoutYOLOModelType::YOLOV8_PUBLAYNET;
    case YOLO_LAYOUT_GENERAL6:
        return ocr::DocLayoutYOLOModelType::YOLOV8_GENERAL6;
    case DOCLAYOUT_YOLO_D4LA:
        return ocr::DocLayoutYOLOModelType::D4LA;
    case DOCLAYOUT_YOLO_DOCSYNTH:
        return ocr::DocLayoutYOLOModelType::DOCSYNTH;
    case DOCLAYOUT_YOLO_DOCSTRUCTBENCH:
    default:
        return ocr::DocLayoutYOLOModelType::DOCSTRUCTBENCH;
    }
}

static bool is_yolov8_layout_type( int type ) {
    return type >= YOLO_LAYOUT_PAPER && type <= YOLO_LAYOUT_GENERAL6;
}

static bool is_doclayout_yolo_type( int type ) {
    return type >= DOCLAYOUT_YOLO_DOCSTRUCTBENCH;
}

static bool is_pp_layout_type( int type ) {
    return type == PP_LAYOUT_CDLA || type == PP_LAYOUT_PUBLAYNET;
}

OCR_API DocLayoutYOLOHandle *doclayout_yolo_create( const DocLayoutYOLOConfig *config ) {
    g_error.clear();
    if ( !config ) {
        set_error( "doclayout yolo config is NULL" );
        return nullptr;
    }
    if ( !config->model_path ) {
        set_error( "model_path is required" );
        return nullptr;
    }

    try {
        ocr::DocLayoutYOLOAnalyzer::Config cfg;
        cfg.model_path = config->model_path;
        cfg.model_type = to_doclayout_yolo_model_type( config->model_type );
        cfg.use_gpu = config->use_gpu != 0;
        cfg.gpu_id = config->gpu_id;
        if ( config->conf_thresh > 0 ) {
            cfg.conf_thresh = config->conf_thresh;
        } else {
            cfg.conf_thresh = is_doclayout_yolo_type( config->model_type ) ? 0.2f : 0.5f;
        }
        cfg.iou_thresh = config->iou_thresh > 0 ? config->iou_thresh : 0.5f;

        auto handle = std::make_unique<DocLayoutYOLOHandle>();
        handle->analyzer = std::make_unique<ocr::DocLayoutYOLOAnalyzer>( cfg );
        return handle.release();
    } catch ( const std::exception &e ) {
        set_error( e.what() );
        return nullptr;
    } catch ( ... ) {
        set_error( "unknown error during doclayout yolo analyzer creation" );
        return nullptr;
    }
}

OCR_API void doclayout_yolo_destroy( DocLayoutYOLOHandle *handle ) {
    delete handle;
}

static int run_doclayout_yolo_impl( DocLayoutYOLOHandle *handle, const cv::Mat &img,
                                    DocLayoutItemList *out_results ) {
    if ( !handle || !handle->analyzer || !out_results ) {
        set_error( "invalid arguments to doclayout_yolo_run" );
        return -1;
    }
    if ( img.empty() ) {
        set_error( "image is empty" );
        return -1;
    }

    try {
        auto results = handle->analyzer->analyze( img );

        out_results->count = static_cast<int>( results.size() );
        out_results->items = nullptr;

        if ( out_results->count > 0 ) {
            out_results->items = static_cast<DocLayoutItem *>(
                std::calloc( out_results->count, sizeof( DocLayoutItem ) ) );
            if ( !out_results->items ) {
                set_error( "memory allocation failed" );
                return -1;
            }
            for ( int i = 0; i < out_results->count; ++i ) {
                const auto &r = results[i];
                for ( int k = 0; k < 4; ++k ) {
                    out_results->items[i].box[k] = r.box[k];
                }
                out_results->items[i].score = r.score;
                out_results->items[i].class_id = r.class_id;
                out_results->items[i].class_name = _strdup( r.class_name.c_str() );
                out_results->items[i].order = r.order;
            }
        }
        return 0;
    } catch ( const std::exception &e ) {
        set_error( e.what() );
        return -1;
    } catch ( ... ) {
        set_error( "unknown error during doclayout yolo analysis" );
        return -1;
    }
}

OCR_API int doclayout_yolo_run( DocLayoutYOLOHandle *handle,
                                const unsigned char *image_data,
                                int width, int height,
                                DocLayoutItemList *out_results ) {
    g_error.clear();
    if ( !image_data || width <= 0 || height <= 0 ) {
        set_error( "invalid image dimensions or data" );
        return -1;
    }
    cv::Mat img( height, width, CV_8UC3, const_cast<unsigned char *>( image_data ) );
    img = img.clone();
    return run_doclayout_yolo_impl( handle, img, out_results );
}

OCR_API int doclayout_yolo_run_file( DocLayoutYOLOHandle *handle,
                                     const char *image_path,
                                     DocLayoutItemList *out_results ) {
    g_error.clear();
    if ( !image_path ) {
        set_error( "image_path is NULL" );
        return -1;
    }
    cv::Mat img = cv::imread( image_path, cv::IMREAD_COLOR );
    if ( img.empty() ) {
        set_error( std::string( "failed to read image: " ) + image_path );
        return -1;
    }
    return run_doclayout_yolo_impl( handle, img, out_results );
}

OCR_API void doc_layout_free_results( DocLayoutItemList *results ) {
    if ( !results || !results->items )
        return;
    for ( int i = 0; i < results->count; ++i ) {
        if ( results->items[i].class_name ) {
            std::free( const_cast<char *>( results->items[i].class_name ) );
        }
    }
    std::free( results->items );
    results->items = nullptr;
    results->count = 0;
}

/* ===== Image Show API ===== */

static int image_show_impl( const cv::Mat &img,
                            const RectBox *rects, int rect_count,
                            const char *win_name,
                            int color_b, int color_g, int color_r,
                            int thickness, int wait_ms ) {
    if ( img.empty() ) {
        set_error( "image is empty" );
        return -1;
    }

    cv::Mat display = img.clone();
    cv::Scalar color( color_b, color_g, color_r );

    for ( int i = 0; i < rect_count; ++i ) {
        cv::Point pt1( rects[i].x1, rects[i].y1 );
        cv::Point pt2( rects[i].x2, rects[i].y2 );
        cv::Rect roi( pt1, pt2 );
        if ( roi.x >= 0 && roi.y >= 0 &&
             roi.x + roi.width <= display.cols &&
             roi.y + roi.height <= display.rows &&
             roi.width > 0 && roi.height > 0 ) {
            cv::rectangle( display, pt1, pt2, color, thickness, cv::LINE_AA );
        }
    }

    std::string name = win_name ? win_name : "Image";
    cv::namedWindow( name, cv::WINDOW_NORMAL );
    cv::imshow( name, display );

    if ( wait_ms >= 0 ) {
        cv::waitKey( wait_ms );
    }

    return 0;
}

OCR_API int image_show_rects( const unsigned char *image_data,
                              int width, int height,
                              const RectBox *rects, int rect_count,
                              const char *win_name,
                              int color_b, int color_g, int color_r,
                              int thickness, int wait_ms ) {
    g_error.clear();
    if ( !image_data || width <= 0 || height <= 0 ) {
        set_error( "invalid image dimensions or data" );
        return -1;
    }
    cv::Mat img( height, width, CV_8UC3, const_cast<unsigned char *>( image_data ) );
    img = img.clone();
    return image_show_impl( img, rects, rect_count, win_name,
                            color_b, color_g, color_r, thickness, wait_ms );
}

OCR_API int image_show_rects_file( const char *image_path,
                                   const RectBox *rects, int rect_count,
                                   const char *win_name,
                                   int color_b, int color_g, int color_r,
                                   int thickness, int wait_ms ) {
    g_error.clear();
    if ( !image_path ) {
        set_error( "image_path is NULL" );
        return -1;
    }
    cv::Mat img = cv::imread( image_path, cv::IMREAD_COLOR );
    if ( img.empty() ) {
        set_error( std::string( "failed to read image: " ) + image_path );
        return -1;
    }
    return image_show_impl( img, rects, rect_count, win_name,
                            color_b, color_g, color_r, thickness, wait_ms );
}