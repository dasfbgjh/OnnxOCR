#include <iostream>
#include <vector>
#include "ocr.h"

void run_table( const char *img, const char *model_path, int model_type,
                const char *unet_path = nullptr, const char *cls_path = nullptr,
                const char *tag = "TABLE" ) {
    printf( "\n============================================================\n" );
    printf( "C++ Table: %s\n", tag );
    printf( "Model: %s\n", model_path );
    printf( "Image: %s\n", img );

    TableConfig cfg = {};
    cfg.model_path = model_path;
    cfg.model_type = model_type;
    cfg.unet_model_path = unet_path;
    cfg.cls_model_path = cls_path;

    TableHandle *handle = table_create( &cfg );
    if ( !handle ) {
        printf( "Failed to create table recognizer\n" );
        return;
    }

    TableResult result = {};
    table_run_file( handle, img, &result );
    printf( "Cells: %d, Score: %.3f\n", result.cell_count, result.score );

    for ( int i = 0; i < result.cell_count; ++i ) {
        const PointF *pts = result.cells[i].bbox;
        RectBox lp = result.logic_points[i];
        printf( "  [%d] bbox=(%.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f)  logic=(%d, %d, %d, %d)\n",
                i, pts[0].x, pts[0].y, pts[1].x, pts[1].y,
                pts[2].x, pts[2].y, pts[3].x, pts[3].y,
                lp.x1, lp.y1, lp.x2, lp.y2 );
    }

    std::vector<RectBox> cell_rects( result.cell_count );
    for ( int i = 0; i < result.cell_count; ++i ) {
        const PointF *pts = result.cells[i].bbox;
        int min_x = static_cast<int>( pts[0].x ), max_x = static_cast<int>( pts[0].x );
        int min_y = static_cast<int>( pts[0].y ), max_y = static_cast<int>( pts[0].y );
        for ( int k = 1; k < 4; ++k ) {
            int x = static_cast<int>( pts[k].x );
            int y = static_cast<int>( pts[k].y );
            if ( x < min_x )
                min_x = x;
            if ( x > max_x )
                max_x = x;
            if ( y < min_y )
                min_y = y;
            if ( y > max_y )
                max_y = y;
        }
        cell_rects[i].x1 = min_x;
        cell_rects[i].y1 = min_y;
        cell_rects[i].x2 = max_x;
        cell_rects[i].y2 = max_y;
    }
    image_show_rects_file( img,
                           cell_rects.data(), static_cast<int>( cell_rects.size() ),
                           tag, 0, 255, 0, 2, 0 );

    table_free_result( &result );
    table_destroy( handle );
}

void table() {
    const char *img = "D:/Downloads/t1.png";
    const char *slanet_path = "D:/project/cmake/OnnxOCR/models/table/slanet_plus/slanet-plus.onnx";
    const char *unet_path = "D:/project/cmake/OnnxOCR/models/table/unet/unet.onnx";
    const char *cls_path = "D:/project/cmake/OnnxOCR/models/table/cls/cls.onnx";

    run_table( img, slanet_path, TABLE_MODEL_SLANET_PLUS, nullptr, nullptr, "SLANET_PLUS (wireless)" );

    run_table( img, unet_path, TABLE_MODEL_UNET, nullptr, nullptr, "UNET (wired)" );

    run_table( img, slanet_path, TABLE_MODEL_UNET_SLANET_PLUS, unet_path, cls_path, "AUTO (classify + select)" );
}

void run_layout( const char *img, const char *model_path, DocLayoutYOLOModelType model_type, const char *tag ) {
    printf( "\n============================================================\n" );
    printf( "C++ Layout: %s\n", tag );
    printf( "Model: %s\n", model_path );
    printf( "Image: %s\n", img );

    DocLayoutYOLOConfig cfg = {};
    cfg.model_path = model_path;
    cfg.model_type = model_type;

    DocLayoutYOLOHandle *handle = doclayout_yolo_create( &cfg );
    if ( !handle ) {
        printf( "Failed to create layout analyzer\n" );
        return;
    }

    DocLayoutItemList results = {};
    if ( doclayout_yolo_run_file( handle, img, &results ) != 0 ) {
        printf( "Layout analysis failed\n" );
        doclayout_yolo_destroy( handle );
        return;
    }

    printf( "Layout items: %d\n", results.count );
    for ( int i = 0; i < results.count; ++i ) {
        const DocLayoutItem &item = results.items[i];
        printf( "  [%d] %-20s box=(%.1f, %.1f, %.1f, %.1f)  score=%.3f  order=%d\n",
                item.class_id, item.class_name,
                item.box[0], item.box[1], item.box[2], item.box[3],
                item.score, item.order );
    }

    std::vector<RectBox> rects( results.count );
    for ( int i = 0; i < results.count; ++i ) {
        rects[i].x1 = static_cast<int>( results.items[i].box[0] );
        rects[i].y1 = static_cast<int>( results.items[i].box[1] );
        rects[i].x2 = static_cast<int>( results.items[i].box[2] );
        rects[i].y2 = static_cast<int>( results.items[i].box[3] );
    }
    image_show_rects_file( img,
                           rects.data(), static_cast<int>( rects.size() ),
                           tag, 0, 255, 0, 2, 0 );

    doc_layout_free_results( &results );
    doclayout_yolo_destroy( handle );
}

void layout() {
    const char *img = "D:/Downloads/t2.png";

    run_layout( img,
                "D:/project/cmake/OnnxOCR/models/layout/pp_layout_cdla/layout_cdla.onnx",
                PP_LAYOUT_CDLA, "PP_LAYOUT_CDLA" );

    run_layout( img,
                "D:/project/cmake/OnnxOCR/models/layout/pp_layout_publaynet/layout_publaynet.onnx",
                PP_LAYOUT_PUBLAYNET, "PP_LAYOUT_PUBLAYNET" );

    run_layout( img,
                "D:/project/cmake/OnnxOCR/models/layout/pp_doclayout_v2/doclayout_pp_v2.onnx",
                PP_DOCLAYOUT_V2, "PP_DOCLAYOUT_V2" );
}

int main( int argc, char *argv[] ) {
    table();
    // layout();
    return 0;
}