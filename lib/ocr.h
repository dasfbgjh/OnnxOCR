#ifndef OCR_H
#define OCR_H

#ifdef __cplusplus
extern "C" {
#endif

#if defined( _WIN32 ) || defined( __CYGWIN__ )
#ifdef OCR_EXPORTS
#define OCR_API __declspec( dllexport )
#else
#define OCR_API __declspec( dllimport )
#endif
#else
#define OCR_API __attribute__( ( visibility( "default" ) ) )
#endif

/* 坐标点 */
typedef struct {
    float x;
    float y;
} PointF;

/* 矩形区域：左上角(x1,y1)和右下角(x2,y2)。 */
typedef struct {
    int x1;
    int y1;
    int x2;
    int y2;
} RectBox;

/* 获取最后的错误信息（线程局部）。无错误时返回NULL。 */
OCR_API const char *last_error( void );

/* ===================== 字符识别API ===================== */

/* OCR引擎实例的不透明句柄。 */
typedef struct OcrHandle OcrHandle;

/* 单个文本区域结果：4个角点（从左上角顺时针）、识别文本（UTF-8）和置信度分数[0,1]。 */
typedef struct {
    PointF box[4];    /* 4个角点，从左上角顺时针 */
    const char *text; /* UTF-8字符串，由库拥有（在下一次ocr_run/ocr_free_results之前有效） */
    float score;      /* 识别置信度 */
} OcrResult;

/* OCR结果列表。 */
typedef struct {
    OcrResult *items;
    int count;
} OcrResultList;

/* 文本方向分类模型类型。 */
typedef enum {
    CLS_MODEL_ANGLE = 0,       /* 2类角度分类（0°/180°），输入48×192，/255归一化 */
    CLS_MODEL_ORIENTATION = 1, /* 4类方向分类（0°/90°/180°/270°），输入224×224，ImageNet归一化 */
} ClsModelType;

/* 创建OCR引擎的配置。
   字符串字段在内部被复制；调用方保留所有权。
   将cls_model_path设为NULL可禁用角度分类。 */
typedef struct {
    const char *det_model_path;     /* 检测ONNX模型路径（必填） */
    const char *rec_model_path;     /* 识别ONNX模型路径（必填） */
    const char *rec_char_dict_path; /* 字符字典路径（必填） */
    const char *cls_model_path;     /* 方向分类ONNX模型路径（NULL = 禁用） */
    int cls_model_type;             /* ClsModelType枚举值，默认CLS_MODEL_ANGLE */

    int use_gpu; /* 0 = CPU，1 = CUDA */
    int gpu_id;  /* GPU设备ID */

    /* 检测参数 */
    float det_limit_side_len;      /* 输入图像缩放限制边长，默认960。图像最长边缩放到此值，控制检测精度与速度 */
    const char *det_limit_type;    /* 缩放限制方式，默认"max"。"max": 限制最长边；"min": 限制最短边 */
    float det_db_thresh;           /* DB二值化分割阈值，默认0.3。控制概率图二值化，值越大检测框越少 */
    float det_db_box_thresh;       /* DB框置信度阈值，默认0.6。过滤低置信度框，值越大保留框越少 */
    float det_db_unclip_ratio;     /* DB框扩展比例，默认1.5。对检测框做膨胀扩展，值越大框越宽松 */
    int use_dilation;              /* 是否对概率图做形态学膨胀，默认0（禁用）。启用(1)可改善密集文字检测效果 */
    const char *det_db_score_mode; /* 框置信度计算方式，默认"fast"。"fast": 取框内最小概率；"slow": 取框内平均概率 */
    const char *det_box_type;      /* 输出框类型，默认"quad"。"quad": 四边形(4点)；"poly": 多边形(>4点) */

    /* 识别参数 */
    int rec_batch_num;  /* 识别批处理大小，默认6。每次推理处理的图片数，增大可提升吞吐但占更多显存 */
    int rec_image_c;    /* 识别输入图像通道数，默认3（RGB） */
    int rec_image_h;    /* 识别输入图像高度，默认48。模型要求的固定高度，影响文字行特征提取 */
    int rec_image_w;    /* 识别输入图像宽度，默认320。模型要求的固定宽度，宽文本会被缩放到此值 */
    int use_space_char; /* 是否在字典中包含空格字符，默认1（启用） */
    float drop_score;   /* 识别结果过滤阈值，默认0.5。置信度低于此值的识别结果将被丢弃 */

    /* 分类参数（cls_model_path非NULL时生效） */
    int cls_batch_num; /* 分类批处理大小，默认6。每次推理处理的图片数 */
    int cls_image_c;   /* 分类输入图像通道数，默认3（RGB），仅ANGLE模式使用 */
    int cls_image_h;   /* 分类输入图像高度，默认48，仅ANGLE模式使用 */
    int cls_image_w;   /* 分类输入图像宽度，默认192，仅ANGLE模式使用 */
    float cls_thresh;  /* 分类置信度阈值，默认0.9。高于此阈值才执行方向矫正，低于则保持原方向 */
} OcrConfig;

/* 创建OCR引擎实例。出错时返回NULL。
   使用ocr_last_error()获取错误信息。 */
OCR_API OcrHandle *ocr_create( const OcrConfig *config );

/* 销毁OCR引擎实例并释放所有资源。 */
OCR_API void ocr_destroy( OcrHandle *handle );

/* 在BGR uint8图像上运行OCR（HWC布局）。
   image_data: 像素数据指针（H * W * 3字节，BGR顺序）
   width, height: 图像尺寸
   crop_x1, crop_y1, crop_x2, crop_y2: 裁剪区域（-1表示使用默认值：
     x1/y1默认为0，x2/y2默认为图片宽高）
   out_results: 输出结果列表（调用方须调用ocr_free_results）
   返回0表示成功，非0表示错误。 */
OCR_API int ocr_run( OcrHandle *handle,
                     const unsigned char *image_data,
                     int width, int height,
                     int crop_x1, int crop_y1, int crop_x2, int crop_y2,
                     OcrResultList *out_results );

/* 在图像文件上运行OCR（使用OpenCV解码）。
   image_path: 图像文件路径
   crop_x1, crop_y1, crop_x2, crop_y2: 裁剪区域（-1表示使用默认值：
     x1/y1默认为0，x2/y2默认为图片宽高）
   out_results: 输出结果列表
   返回0表示成功，非0表示错误。 */
OCR_API int ocr_run_file( OcrHandle *handle,
                          const char *image_path,
                          int crop_x1, int crop_y1, int crop_x2, int crop_y2,
                          OcrResultList *out_results );

/* 释放先前由ocr_run/ocr_run_file返回的结果列表。 */
OCR_API void ocr_free_results( OcrResultList *results );

/* ===================== 车牌识别API ===================== */

/* 单个车牌识别结果。 */
typedef struct {
    RectBox box;         /* 检测框，原始图像坐标 */
    float score;         /* 检测置信度 */
    const char *plate;   /* 识别的车牌文本（UTF-8，由库拥有） */
    const char *type;    /* "single_layer"或"double_layer"（由库拥有） */
    PointF landmarks[4]; /* 4个角点关键点 */
} PlateResult;

/* 车牌结果列表。 */
typedef struct {
    PlateResult *items;
    int count;
} PlateResultList;

/* 创建车牌识别器的配置。 */
typedef struct {
    const char *detect_model_path; /* car_plate_detect.onnx（必填） */
    const char *rec_model_path;    /* plate_rec.onnx（必填） */
    int use_gpu;                   /* 0 = CPU，1 = CUDA */
    int gpu_id;                    /* GPU设备ID */
    float min_score;               /* 置信度阈值，默认0.4，值越低→保留更多框（召回率高但可能有误检）；值越高→只保留高置信度框（精度高但可能漏检） */
    float iou_thresh;              /* IOU阈值，默认0.5 值越低→对重叠更敏感，更容易抑制重叠框（减少重复检测）；值越高→允许更多重叠框共存（可能产生重复检测）*/
} PlateConfig;

/* 车牌识别器实例的不透明句柄。 */
typedef struct PlateHandle PlateHandle;

/* 创建车牌识别器实例。出错时返回NULL。 */
OCR_API PlateHandle *plate_create( const PlateConfig *config );

/* 销毁车牌识别器实例。 */
OCR_API void plate_destroy( PlateHandle *handle );

/* 在BGR uint8图像中识别车牌（HWC布局）。
   返回0表示成功，非0表示错误。 */
OCR_API int plate_run( PlateHandle *handle,
                       const unsigned char *image_data,
                       int width, int height,
                       PlateResultList *out_results );

/* 在图像文件中识别车牌。
   返回0表示成功，非0表示错误。 */
OCR_API int plate_run_file( PlateHandle *handle,
                            const char *image_path,
                            PlateResultList *out_results );

/* 释放车牌结果列表。 */
OCR_API void plate_free_results( PlateResultList *results );

/* ===================== 表格识别API ===================== */

/* 带边界框的单个表格单元格。 */
typedef struct {
    PointF bbox[4]; /* 4个角点，原始图像坐标 */
} TableCell;

/* 表格识别结果。 */
typedef struct {
    TableCell *cells;      /* 单元格边界框数组 */
    RectBox *logic_points; /* 单元格逻辑位置数组（与cells一一对应），x1=row_start,y1=row_end,x2=col_start,y2=col_end */
    int cell_count;        /* 单元格数量 */
    float score;           /* 平均结构置信度 */
} TableResult;

/* 表格模型类型选择器。
   SLANet系列模型（含ppstructure_zh/en等）均使用SLANET_PLUS，
   通过model_path指定不同的模型文件即可。 */
typedef enum {
    TABLE_MODEL_SLANET_PLUS = 0,      /* SLANet系列模型（无线表格） */
    TABLE_MODEL_UNET = 1,             /* unet.onnx（有线/边框表格） */
    TABLE_MODEL_UNET_SLANET_PLUS = 2, /* 组合模式：先分类再选模型 */
} TableModelType;

/* 创建表格识别器的配置。 */
typedef struct {
    const char *model_path;      /* 表格结构ONNX模型路径（必填）。
                                    SLANET_PLUS/UNET模式：对应模型路径；
                                    UNET_SLANET_PLUS模式：SLANet+模型路径 */
    int model_type;              /* TableModelType枚举值 */
    const char *unet_model_path; /* UNet模型路径（仅UNET_SLANET_PLUS模式使用） */
    const char *cls_model_path;  /* 表格分类器模型路径（仅UNET_SLANET_PLUS模式使用） */
    int use_gpu;                 /* 0 = CPU，1 = CUDA */
    int gpu_id;                  /* GPU设备ID */
} TableConfig;

/* 表格识别器实例的不透明句柄。 */
typedef struct TableHandle TableHandle;

/* 创建表格识别器实例。出错时返回NULL。 */
OCR_API TableHandle *table_create( const TableConfig *config );

/* 销毁表格识别器实例。 */
OCR_API void table_destroy( TableHandle *handle );

/* 在BGR uint8图像中识别表格结构（HWC布局）。
   返回0表示成功，非0表示错误。 */
OCR_API int table_run( TableHandle *handle,
                       const unsigned char *image_data,
                       int width, int height,
                       TableResult *out_result );

/* 在图像文件中识别表格结构。
   返回0表示成功，非0表示错误。 */
OCR_API int table_run_file( TableHandle *handle,
                            const char *image_path,
                            TableResult *out_result );

/* 释放表格结果。 */
OCR_API void table_free_result( TableResult *result );

/* ===================== 版面分析统一API ===================== */

/* 单个版面分析结果。 */
typedef struct {
    float box[4];           /* [x1, y1, x2, y2] 原始图像坐标 */
    float score;            /* 置信度分数 */
    int class_id;           /* 类别索引 */
    const char *class_name; /* 类别名称（由库拥有） */
    int order;              /* 阅读顺序索引 */
} DocLayoutItem;

/* 版面结果列表。 */
typedef struct {
    DocLayoutItem *items;
    int count;
} DocLayoutItemList;

/* 版面分析模型类型选择器
PP-Structure CDLA 类别列表
title, text, figure, figure_caption, table, table_caption, header, footer, reference, equation

PP-Structure PublayNet 类别列表
text, title, list, table, figure

pp_doclayoutv2 类别列表
abstract, algorithm, aside_text, chart, content, formula, doc_title, figure_title, footer, footnote, formula_number, header, image, inline_formula, number, paragraph_title, reference, reference_content, seal, table, text, vertical_text, vision_footnote 等

YOLO_LAYOUT_PAPER 类别列表
title, text, figure, table, caption, equation, reference

YOLO_LAYOUT_REPORT 类别列表
title, text, figure, table, header, footer

YOLO_LAYOUT_PUBLAYNET 类别列表
text, title, list, table, figure

YOLO_LAYOUT_GENERAL6 类别列表
text, title, figure, table, caption, equation

DocLayout YOLO 类别列表
caption, footnote, formula, list-item, page-footer, page-header, picture, section-header, table, text, title
*/
typedef enum {
    /* PP-Structure系列：ImageNet归一化，DFL解码，默认conf_thresh=0.5 */
    PP_LAYOUT_CDLA = 0,      /* 800×608，中文文档，10类 */
    PP_LAYOUT_PUBLAYNET = 1, /* 800×608，英文文档，5类 */

    /* 3输入张量，letterbox(scaleup=False)，默认conf_thresh=0.5 */
    PP_DOCLAYOUT_V2 = 2, /* 800×800，25类*/

    /* YOLOv8系列：直接resize（不保持宽高比），默认conf_thresh=0.5 */
    YOLO_LAYOUT_PAPER = 3,     /* 640×640，7类 */
    YOLO_LAYOUT_REPORT = 4,    /* 640×640，6类 */
    YOLO_LAYOUT_PUBLAYNET = 5, /* 640×640，5类 */
    YOLO_LAYOUT_GENERAL6 = 6,  /* 640×640，6类 */

    /* DocLayout YOLO系列：letterbox预处理（保持宽高比），默认conf_thresh=0.2 */
    DOCLAYOUT_YOLO_DOCSTRUCTBENCH = 7, /* 1024×1024，11类 */
    DOCLAYOUT_YOLO_D4LA = 8,           /* 1600×1600，11类 */
    DOCLAYOUT_YOLO_DOCSYNTH = 9,       /* 1120×1120，11类 */
} DocLayoutYOLOModelType;

/* 创建版面分析器的配置。 */
typedef struct {
    const char *model_path;
    int model_type; /* DocLayoutYOLOModelType枚举值 */
    int use_gpu;
    int gpu_id;
    float conf_thresh; /* pp_doclayoutv2/YOLOv8默认0.5，DocLayout YOLO默认0.2 */
    float iou_thresh;  /* 默认0.5 */
} DocLayoutYOLOConfig;

/* 版面分析器实例的不透明句柄。 */
typedef struct DocLayoutYOLOHandle DocLayoutYOLOHandle;

/* 创建版面分析器实例。出错时返回NULL。 */
OCR_API DocLayoutYOLOHandle *doclayout_yolo_create( const DocLayoutYOLOConfig *config );

/* 销毁版面分析器实例。 */
OCR_API void doclayout_yolo_destroy( DocLayoutYOLOHandle *handle );

/* 在BGR uint8图像中分析版面（HWC布局）。
   返回0表示成功，非0表示错误。 */
OCR_API int doclayout_yolo_run( DocLayoutYOLOHandle *handle,
                                const unsigned char *image_data,
                                int width, int height,
                                DocLayoutItemList *out_results );

/* 在图像文件中分析版面。
   返回0表示成功，非0表示错误。 */
OCR_API int doclayout_yolo_run_file( DocLayoutYOLOHandle *handle,
                                     const char *image_path,
                                     DocLayoutItemList *out_results );

/* 释放版面结果列表。 */
OCR_API void doc_layout_free_results( DocLayoutItemList *results );

/* ===================== 图像显示辅助API ===================== */

/* 在窗口中显示图像，并在其上绘制矩形框。
   image_data: BGR uint8图像（HWC布局）。
   width/height: 图像宽高。
   rects: 矩形数组。
   rect_count: 矩形数量。
   win_name: 窗口标题（NULL则使用默认"Image"）。
   color_b/g/r: 矩形框颜色（BGR），默认绿色(0,255,0)。
   thickness: 矩形框线宽，默认2。
   wait_ms: 显示等待毫秒数，0=无限等待按键，-1=不等待直接关闭。
   返回0表示成功，非0表示错误。 */
OCR_API int image_show_rects( const unsigned char *image_data,
                              int width, int height,
                              const RectBox *rects, int rect_count,
                              const char *win_name,
                              int color_b, int color_g, int color_r,
                              int thickness, int wait_ms );

/* 在窗口中显示图像文件，并在其上绘制矩形框。
   参数含义同image_show_rects，image_path替代内存图像。 */
OCR_API int image_show_rects_file( const char *image_path,
                                   const RectBox *rects, int rect_count,
                                   const char *win_name,
                                   int color_b, int color_g, int color_r,
                                   int thickness, int wait_ms );

#ifdef __cplusplus
}
#endif

#endif /* OCR_H */