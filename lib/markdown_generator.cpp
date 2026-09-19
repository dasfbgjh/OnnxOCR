#include "markdown_generator.h"

#include <algorithm>
#include <cstring>

namespace ocr {

MarkdownGenerator::MarkdownGenerator( OcrDynamicLoader &loader, const Config &cfg )
    : loader_( loader ),
      table_enabled_( !cfg.table_model_path.empty() ),
      include_images_( cfg.include_images ),
      assets_dir_( cfg.assets_dir ) {

    /* 创建OCR实例 */
    OcrConfig ocr_cfg = {};
    ocr_cfg.det_model_path = cfg.det_model_path.c_str();
    ocr_cfg.rec_model_path = cfg.rec_model_path.c_str();
    ocr_cfg.rec_char_dict_path = cfg.rec_char_dict_path.c_str();
    ocr_cfg.cls_model_path = cfg.cls_model_path.empty() ? nullptr : cfg.cls_model_path.c_str();
    ocr_cfg.cls_model_type = cfg.cls_model_type;
    ocr_cfg.use_gpu = cfg.use_gpu;
    ocr_cfg.gpu_id = cfg.gpu_id;
    ocr_handle_ = loader_.create_or_throw( &ocr_cfg );

    /* 创建版面分析实例 */
    DocLayoutYOLOConfig layout_cfg = {};
    layout_cfg.model_path = cfg.layout_model_path.c_str();
    layout_cfg.model_type = cfg.layout_model_type;
    layout_cfg.use_gpu = cfg.use_gpu;
    layout_cfg.gpu_id = cfg.gpu_id;
    layout_cfg.conf_thresh = cfg.layout_conf_thresh;
    layout_cfg.iou_thresh = cfg.layout_iou_thresh;
    layout_handle_ = loader_.doclayout_yolo_create_or_throw( &layout_cfg );

    /* 创建表格识别实例（可选） */
    if ( table_enabled_ ) {
        TableConfig table_cfg = {};
        table_cfg.model_path = cfg.table_model_path.c_str();
        table_cfg.model_type = cfg.table_model_type;
        table_cfg.cls_model_path = cfg.table_cls_model_path.empty()
                                       ? nullptr
                                       : cfg.table_cls_model_path.c_str();
        table_cfg.unet_model_path = cfg.table_unet_model_path.empty()
                                        ? nullptr
                                        : cfg.table_unet_model_path.c_str();
        table_cfg.use_gpu = cfg.use_gpu;
        table_cfg.gpu_id = cfg.gpu_id;
        table_handle_ = loader_.table_create_or_throw( &table_cfg );
    }
}

MarkdownGenerator::~MarkdownGenerator() {
    if ( table_handle_ )
        loader_.table_destroy( table_handle_ );
    if ( layout_handle_ )
        loader_.doclayout_yolo_destroy( layout_handle_ );
    if ( ocr_handle_ )
        loader_.destroy( ocr_handle_ );
}

/* ===== convert_file：文件路径模式 ===== */

std::string MarkdownGenerator::convert_file( const char *image_path ) {
    /* Step 1: 文档版面分析 */
    DocLayoutResultGuard layout_results( loader_ );
    int rc = loader_.doclayout_yolo_run_file( layout_handle_, image_path, layout_results.get() );
    if ( rc != 0 || layout_results.get()->count == 0 ) {
        /* 降级：对整页做OCR */
        OcrResultGuard ocr_results( loader_ );
        loader_.run_file( ocr_handle_, image_path, -1, -1, -1, -1, ocr_results.get() );
        std::ostringstream ss;
        for ( int i = 0; i < ocr_results.get()->count; ++i ) {
            const char *txt = ocr_results.get()->items[i].text;
            if ( txt && txt[0] != '\0' )
                ss << txt << "\n";
        }
        return ss.str();
    }

    /* Step 2: 按阅读顺序遍历每个版面块，生成Markdown */
    std::ostringstream markdown;
    const DocLayoutItemList *list = layout_results.get();

    for ( int i = 0; i < list->count; ++i ) {
        const DocLayoutItem &item = list->items[i];
        const char *cls = item.class_name;
        if ( !cls )
            continue;

        int x1 = static_cast<int>( item.box[0] );
        int y1 = static_cast<int>( item.box[1] );
        int x2 = static_cast<int>( item.box[2] );
        int y2 = static_cast<int>( item.box[3] );

        std::string block_md;

        if ( is_table_block( cls ) ) {
            /* 表格区域：使用表格识别获取二维网格 */
            if ( table_enabled_ && table_handle_ ) {
                auto grid = recognize_table_file( image_path, x1, y1, x2, y2 );
                if ( !grid.empty() ) {
                    std::ostringstream ss;
                    for ( size_t r = 0; r < grid.size(); ++r ) {
                        ss << "|";
                        for ( size_t c = 0; c < grid[r].size(); ++c ) {
                            ss << " " << grid[r][c] << " |";
                        }
                        ss << "\n";
                        if ( r == 0 ) {
                            ss << "|";
                            for ( size_t c = 0; c < grid[0].size(); ++c ) {
                                ss << " --- |";
                            }
                            ss << "\n";
                        }
                    }
                    block_md = ss.str();
                }
            }
            /* 降级：如果表格识别失败或未启用，用OCR按行识别 */
            if ( block_md.empty() ) {
                auto lines = ocr_file_crop_lines( image_path, x1, y1, x2, y2 );
                if ( !lines.empty() ) {
                    std::ostringstream ss;
                    for ( const auto &line : lines ) {
                        ss << line << "\n";
                    }
                    block_md = ss.str();
                }
            }
        } else if ( is_title_block( cls ) ) {
            std::string text = ocr_file_crop( image_path, x1, y1, x2, y2 );
            if ( !text.empty() ) {
                int level = get_title_level( cls );
                block_md = std::string( level, '#' ) + " " + text;
            }
        } else if ( is_image_block( cls ) ) {
            if ( include_images_ ) {
                block_md = std::string( "![image](" ) + assets_dir_ + "/block_" +
                           std::to_string( item.order ) + ".png)";
            }
        } else if ( std::string( cls ) == "formula" ||
                    std::string( cls ) == "inline_formula" ||
                    std::string( cls ) == "formula_number" ) {
            continue;
        } else {
            /* 文本块：OCR */
            auto lines = ocr_file_crop_lines( image_path, x1, y1, x2, y2 );
            if ( !lines.empty() ) {
                std::ostringstream ss;
                for ( const auto &line : lines ) {
                    ss << line << "\n";
                }
                block_md = ss.str();
                if ( !block_md.empty() && block_md.back() == '\n' )
                    block_md.pop_back();
            }
        }

        if ( !block_md.empty() ) {
            markdown << block_md << "\n\n";
        }
    }

    return markdown.str();
}

/* ===== convert：内存图像模式 ===== */

std::string MarkdownGenerator::convert( const unsigned char *image_data,
                                        int width, int height ) {
    /* Step 1: 文档版面分析 */
    DocLayoutResultGuard layout_results( loader_ );
    int rc = loader_.doclayout_yolo_run( layout_handle_, image_data, width, height,
                                     layout_results.get() );
    if ( rc != 0 || layout_results.get()->count == 0 ) {
        /* 降级：对整页做OCR */
        OcrResultGuard ocr_results( loader_ );
        loader_.run( ocr_handle_, image_data, width, height,
                     -1, -1, -1, -1, ocr_results.get() );
        std::ostringstream ss;
        for ( int i = 0; i < ocr_results.get()->count; ++i ) {
            const char *txt = ocr_results.get()->items[i].text;
            if ( txt && txt[0] != '\0' )
                ss << txt << "\n";
        }
        return ss.str();
    }

    /* Step 2: 按阅读顺序遍历每个版面块，生成Markdown */
    std::ostringstream markdown;
    const DocLayoutItemList *list = layout_results.get();

    for ( int i = 0; i < list->count; ++i ) {
        const DocLayoutItem &item = list->items[i];
        const char *cls = item.class_name;
        if ( !cls )
            continue;

        int x1 = static_cast<int>( item.box[0] );
        int y1 = static_cast<int>( item.box[1] );
        int x2 = static_cast<int>( item.box[2] );
        int y2 = static_cast<int>( item.box[3] );

        std::string block_md;

        if ( is_table_block( cls ) ) {
            if ( table_enabled_ && table_handle_ ) {
                auto grid = recognize_table( image_data, width, height, x1, y1, x2, y2 );
                if ( !grid.empty() ) {
                    std::ostringstream ss;
                    for ( size_t r = 0; r < grid.size(); ++r ) {
                        ss << "|";
                        for ( size_t c = 0; c < grid[r].size(); ++c ) {
                            ss << " " << grid[r][c] << " |";
                        }
                        ss << "\n";
                        if ( r == 0 ) {
                            ss << "|";
                            for ( size_t c = 0; c < grid[0].size(); ++c ) {
                                ss << " --- |";
                            }
                            ss << "\n";
                        }
                    }
                    block_md = ss.str();
                }
            }
            if ( block_md.empty() ) {
                auto lines = ocr_crop_lines( image_data, width, height, x1, y1, x2, y2 );
                if ( !lines.empty() ) {
                    std::ostringstream ss;
                    for ( const auto &line : lines ) {
                        ss << line << "\n";
                    }
                    block_md = ss.str();
                }
            }
        } else if ( is_title_block( cls ) ) {
            std::string text = ocr_crop( image_data, width, height, x1, y1, x2, y2 );
            if ( !text.empty() ) {
                int level = get_title_level( cls );
                block_md = std::string( level, '#' ) + " " + text;
            }
        } else if ( is_image_block( cls ) ) {
            if ( include_images_ ) {
                block_md = std::string( "![image](" ) + assets_dir_ + "/block_" +
                           std::to_string( item.order ) + ".png)";
            }
        } else if ( std::string( cls ) == "formula" ||
                    std::string( cls ) == "inline_formula" ||
                    std::string( cls ) == "formula_number" ) {
            continue;
        } else {
            auto lines = ocr_crop_lines( image_data, width, height, x1, y1, x2, y2 );
            if ( !lines.empty() ) {
                std::ostringstream ss;
                for ( const auto &line : lines ) {
                    ss << line << "\n";
                }
                block_md = ss.str();
                if ( !block_md.empty() && block_md.back() == '\n' )
                    block_md.pop_back();
            }
        }

        if ( !block_md.empty() ) {
            markdown << block_md << "\n\n";
        }
    }

    return markdown.str();
}

/* ===== 内存图像辅助方法 ===== */

std::string MarkdownGenerator::ocr_crop( const unsigned char *image_data,
                                         int width, int height,
                                         int x1, int y1, int x2, int y2 ) {
    OcrResultGuard results( loader_ );
    int rc = loader_.run( ocr_handle_, image_data, width, height,
                          x1, y1, x2, y2, results.get() );
    if ( rc != 0 )
        return "";

    std::ostringstream ss;
    for ( int i = 0; i < results.get()->count; ++i ) {
        const char *txt = results.get()->items[i].text;
        if ( txt && txt[0] != '\0' ) {
            ss << txt << " ";
        }
    }
    std::string result = ss.str();
    if ( !result.empty() && result.back() == ' ' )
        result.pop_back();
    return result;
}

std::vector<std::string> MarkdownGenerator::ocr_crop_lines(
    const unsigned char *image_data,
    int width, int height,
    int x1, int y1, int x2, int y2 ) {
    OcrResultGuard results( loader_ );
    int rc = loader_.run( ocr_handle_, image_data, width, height,
                          x1, y1, x2, y2, results.get() );
    if ( rc != 0 )
        return {};

    std::vector<std::string> lines;
    for ( int i = 0; i < results.get()->count; ++i ) {
        const char *txt = results.get()->items[i].text;
        if ( txt && txt[0] != '\0' ) {
            lines.push_back( txt );
        }
    }
    return lines;
}

/* ===== 文件模式辅助方法 ===== */

std::string MarkdownGenerator::ocr_file_crop( const char *image_path,
                                              int x1, int y1, int x2, int y2 ) {
    OcrResultGuard results( loader_ );
    int rc = loader_.run_file( ocr_handle_, image_path, x1, y1, x2, y2, results.get() );
    if ( rc != 0 )
        return "";

    std::ostringstream ss;
    for ( int i = 0; i < results.get()->count; ++i ) {
        const char *txt = results.get()->items[i].text;
        if ( txt && txt[0] != '\0' ) {
            ss << txt << " ";
        }
    }
    std::string result = ss.str();
    if ( !result.empty() && result.back() == ' ' )
        result.pop_back();
    return result;
}

std::vector<std::string> MarkdownGenerator::ocr_file_crop_lines(
    const char *image_path,
    int x1, int y1, int x2, int y2 ) {
    OcrResultGuard results( loader_ );
    int rc = loader_.run_file( ocr_handle_, image_path, x1, y1, x2, y2, results.get() );
    if ( rc != 0 )
        return {};

    std::vector<std::string> lines;
    for ( int i = 0; i < results.get()->count; ++i ) {
        const char *txt = results.get()->items[i].text;
        if ( txt && txt[0] != '\0' ) {
            lines.push_back( txt );
        }
    }
    return lines;
}

/* ===== 表格识别方法 ===== */

std::vector<std::vector<std::string>> MarkdownGenerator::recognize_table_file(
    const char *image_path,
    int crop_x1, int crop_y1,
    int crop_x2, int crop_y2,
    std::vector<RectBox> *rects ) {

    if ( !table_enabled_ || !table_handle_ )
        return {};

    /* Step 1: 表格结构识别 */
    TableResultGuard table_result( loader_ );
    int rc = loader_.table_run_file( table_handle_, image_path, table_result.get() );
    if ( rc != 0 || table_result.get()->cell_count == 0 )
        return {};

    /* Step 2: 构建二维网格并OCR填充 */
    return build_table_grid_file( *table_result.get(), image_path,
                                  crop_x1, crop_y1, rects );
}

std::vector<std::vector<std::string>> MarkdownGenerator::recognize_table(
    const unsigned char *image_data,
    int width, int height,
    int crop_x1, int crop_y1,
    int crop_x2, int crop_y2,
    std::vector<RectBox> *rects ) {

    if ( !table_enabled_ || !table_handle_ )
        return {};

    TableResultGuard table_result( loader_ );
    int rc = loader_.table_run( table_handle_, image_data, width, height,
                                table_result.get() );
    if ( rc != 0 || table_result.get()->cell_count == 0 )
        return {};

    return build_table_grid( *table_result.get(), image_data, width, height,
                             crop_x1, crop_y1, rects );
}

std::vector<std::vector<std::string>> MarkdownGenerator::build_table_grid(
    const TableResult &table_result,
    const unsigned char *image_data,
    int width, int height,
    int offset_x, int offset_y,
    std::vector<RectBox> *rects ) {

    if ( table_result.cell_count <= 0 || !table_result.logic_points )
        return {};

    /* Step 1: 确定表格行列数 */
    int max_row = 0, max_col = 0;
    for ( int i = 0; i < table_result.cell_count; ++i ) {
        const RectBox &lp = table_result.logic_points[i];
        if ( lp.y1 + 1 > max_row )
            max_row = lp.y1 + 1;
        if ( lp.y2 + 1 > max_col )
            max_col = lp.y2 + 1;
    }
    if ( max_row <= 0 || max_col <= 0 )
        return {};

    /* Step 2: 初始化二维网格 */
    std::vector<std::vector<std::string>> grid(
        max_row, std::vector<std::string>( max_col, "" ) );

    /* Step 3: 对每个单元格做OCR并填入网格 */
    for ( int i = 0; i < table_result.cell_count; ++i ) {
        const TableCell &cell = table_result.cells[i];
        const RectBox &lp = table_result.logic_points[i];

        /* 单元格bbox: 4个角点PointF
           取外接矩形作为裁剪区域 */
        float min_x = cell.bbox[0].x, min_y = cell.bbox[0].y;
        float max_x = cell.bbox[0].x, max_y = cell.bbox[0].y;
        for ( int k = 1; k < 4; ++k ) {
            float cx = cell.bbox[k].x;
            float cy = cell.bbox[k].y;
            if ( cx < min_x )
                min_x = cx;
            if ( cx > max_x )
                max_x = cx;
            if ( cy < min_y )
                min_y = cy;
            if ( cy > max_y )
                max_y = cy;
        }

        /* 转换为原图坐标（加上偏移量） */
        int x1 = static_cast<int>( min_x ) + offset_x;
        int y1 = static_cast<int>( min_y ) + offset_y;
        int x2 = static_cast<int>( max_x ) + offset_x;
        int y2 = static_cast<int>( max_y ) + offset_y;

        /* 对单元格区域做OCR */
        std::string text = ocr_crop( image_data, width, height, x1, y1, x2, y2 );
        if ( rects )
            rects->push_back( { x1, y1, x2, y2 } );

        /* 填入网格（处理跨行跨列：所有覆盖位置都填入相同文本） */
        for ( int r = lp.x1; r <= lp.y1 && r < max_row; ++r ) {
            for ( int c = lp.x2; c <= lp.y2 && c < max_col; ++c ) {
                grid[r][c] = text;
            }
        }
    }

    return grid;
}

std::vector<std::vector<std::string>> MarkdownGenerator::build_table_grid_file(
    const TableResult &table_result,
    const char *image_path,
    int offset_x, int offset_y,
    std::vector<RectBox> *rects ) {

    if ( table_result.cell_count <= 0 || !table_result.logic_points )
        return {};

    int max_row = 0, max_col = 0;
    for ( int i = 0; i < table_result.cell_count; ++i ) {
        const RectBox &lp = table_result.logic_points[i];
        if ( lp.y1 + 1 > max_row )
            max_row = lp.y1 + 1;
        if ( lp.y2 + 1 > max_col )
            max_col = lp.y2 + 1;
    }
    if ( max_row <= 0 || max_col <= 0 )
        return {};

    std::vector<std::vector<std::string>> grid(
        max_row, std::vector<std::string>( max_col, "" ) );

    for ( int i = 0; i < table_result.cell_count; ++i ) {
        const TableCell &cell = table_result.cells[i];
        const RectBox &lp = table_result.logic_points[i];

        float min_x = cell.bbox[0].x, min_y = cell.bbox[0].y;
        float max_x = cell.bbox[0].x, max_y = cell.bbox[0].y;
        for ( int k = 1; k < 4; ++k ) {
            float cx = cell.bbox[k].x;
            float cy = cell.bbox[k].y;
            if ( cx < min_x )
                min_x = cx;
            if ( cx > max_x )
                max_x = cx;
            if ( cy < min_y )
                min_y = cy;
            if ( cy > max_y )
                max_y = cy;
        }

        int x1 = static_cast<int>( min_x ) + offset_x;
        int y1 = static_cast<int>( min_y ) + offset_y;
        int x2 = static_cast<int>( max_x ) + offset_x;
        int y2 = static_cast<int>( max_y ) + offset_y;

        std::string text = ocr_file_crop( image_path, x1, y1, x2, y2 );
        if ( rects )
            rects->push_back( { x1, y1, x2, y2 } );

        for ( int r = lp.x1; r <= lp.y1 && r < max_row; ++r ) {
            for ( int c = lp.x2; c <= lp.y2 && c < max_col; ++c ) {
                grid[r][c] = text;
            }
        }
    }
    return grid;
}

/* ===== 类别判断 ===== */

bool MarkdownGenerator::is_text_block( const std::string &class_name ) {
    return class_name == "text" || class_name == "vertical_text" ||
           class_name == "content" || class_name == "aside_text" ||
           class_name == "abstract" || class_name == "reference" ||
           class_name == "reference_content" || class_name == "footnote" ||
           class_name == "algorithm" || class_name == "number" ||
           class_name == "vision_footnote" || class_name == "header" ||
           class_name == "footer";
}

bool MarkdownGenerator::is_title_block( const std::string &class_name ) {
    return class_name == "doc_title" || class_name == "paragraph_title" ||
           class_name == "figure_title" || class_name == "table_caption" ||
           class_name == "chart_title";
}

bool MarkdownGenerator::is_table_block( const std::string &class_name ) {
    return class_name == "table";
}

bool MarkdownGenerator::is_image_block( const std::string &class_name ) {
    return class_name == "image" || class_name == "chart" ||
           class_name == "seal" || class_name == "figure";
}

int MarkdownGenerator::get_title_level( const std::string &class_name ) {
    if ( class_name == "doc_title" )
        return 1;
    if ( class_name == "paragraph_title" || class_name == "figure_title" ||
         class_name == "table_caption" || class_name == "chart_title" )
        return 2;
    return 3;
}

} // namespace ocr