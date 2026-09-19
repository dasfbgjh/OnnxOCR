#ifndef MARKDOWN_GENERATOR_H
#define MARKDOWN_GENERATOR_H

#include "ocr_dynamic_loader.h"

#include <memory>
#include <string>
#include <sstream>
#include <vector>

namespace ocr {

/* 基于OcrDynamicLoader的Markdown生成器。
   组合文档版面分析 + OCR + 表格识别，将文档图像转换为Markdown文本。
   所有功能通过动态加载的OCR库实现，无需直接依赖C++模型类。 */
class MarkdownGenerator {
public:
    struct Config {
        /* 版面分析模型 - 必填 */
        std::string layout_model_path;
        int layout_model_type = 0;  /* DocLayoutYOLOModelType枚举值，默认PP_DOCLAYOUT_V2 */

        /* OCR模型 - 必填 */
        std::string det_model_path;
        std::string rec_model_path;
        std::string rec_char_dict_path;
        std::string cls_model_path;
        int cls_model_type = 0;  /* ClsModelType枚举值，默认CLS_MODEL_ANGLE */

        /* 表格模型 - 可选，空则禁用表格识别 */
        std::string table_model_path;
        int table_model_type = 0;
        std::string table_cls_model_path;    /* 表格分类器（仅UNET_SLANET_PLUS模式使用） */
        std::string table_unet_model_path;   /* UNet模型（仅UNET_SLANET_PLUS模式使用） */

        /* GPU设置 */
        int use_gpu = 0;
        int gpu_id = 0;

        /* 版面阈值 */
        float layout_conf_thresh = 0.5f;
        float layout_iou_thresh = 0.5f;

        /* 是否在markdown中包含图片占位符 */
        bool include_images = false;
        std::string assets_dir = "assets";
    };

    explicit MarkdownGenerator( OcrDynamicLoader &loader, const Config &cfg );
    ~MarkdownGenerator();

    MarkdownGenerator( const MarkdownGenerator & ) = delete;
    MarkdownGenerator &operator=( const MarkdownGenerator & ) = delete;

    /* 将图像文件转换为Markdown文本 */
    std::string convert_file( const char *image_path );

    /* 将BGR uint8图像（HWC布局）转换为Markdown文本 */
    std::string convert( const unsigned char *image_data, int width, int height );

    /* 表格识别：对图像文件中的表格区域识别，返回二维字符串数组。
       外层vector为行，内层vector为列。
       crop_x1/y1/x2/y2: 裁剪区域，-1表示使用默认值（0,0,宽,高）。
       如果未启用表格识别（table_model_path为空），返回空数组。 */
    std::vector<std::vector<std::string>> recognize_table_file(
        const char *image_path,
        int crop_x1 = -1, int crop_y1 = -1,
        int crop_x2 = -1, int crop_y2 = -1,
        std::vector<RectBox> *rects = nullptr );

    /* 表格识别：对内存图像中的表格区域识别，返回二维字符串数组。
       外层vector为行，内层vector为列。 */
    std::vector<std::vector<std::string>> recognize_table(
        const unsigned char *image_data,
        int width, int height,
        int crop_x1 = -1, int crop_y1 = -1,
        int crop_x2 = -1, int crop_y2 = -1,
        std::vector<RectBox> *rects = nullptr );

private:
    OcrDynamicLoader &loader_;

    OcrHandle *ocr_handle_ = nullptr;
    DocLayoutYOLOHandle *layout_handle_ = nullptr;
    TableHandle *table_handle_ = nullptr;
    bool table_enabled_ = false;
    bool include_images_ = false;
    std::string assets_dir_;

    /* ---- 表格识别内部实现 ---- */

    /* 从TableResult构建二维字符串数组。
       先用表格结构获取单元格位置，再对每个单元格做OCR填充文字。 */
    std::vector<std::vector<std::string>>
    build_table_grid( const TableResult &table_result,
                      const unsigned char *image_data, int width, int height,
                      int offset_x, int offset_y,
                      std::vector<RectBox> *rects = nullptr );

    std::vector<std::vector<std::string>>
    build_table_grid_file( const TableResult &table_result,
                           const char *image_path,
                           int offset_x, int offset_y,
                           std::vector<RectBox> *rects = nullptr );

    /* ---- 内存图像模式的辅助方法 ---- */

    std::string ocr_crop( const unsigned char *image_data,
                          int width, int height,
                          int x1, int y1, int x2, int y2 );

    std::vector<std::string> ocr_crop_lines( const unsigned char *image_data,
                                             int width, int height,
                                             int x1, int y1, int x2, int y2 );

    /* ---- 文件模式的辅助方法 ---- */

    std::string ocr_file_crop( const char *image_path,
                               int x1, int y1, int x2, int y2 );

    std::vector<std::string> ocr_file_crop_lines( const char *image_path,
                                                  int x1, int y1, int x2, int y2 );

    /* ---- 类别判断 ---- */

    static bool is_text_block( const std::string &class_name );
    static bool is_title_block( const std::string &class_name );
    static bool is_table_block( const std::string &class_name );
    static bool is_image_block( const std::string &class_name );
    static int get_title_level( const std::string &class_name );
};

} // namespace ocr

#endif /* MARKDOWN_GENERATOR_H */