# OnnxOCR — C++ ONNX OCR Library

基于 ONNX Runtime + OpenCV 的多功能文档智能 C++ 库，提供纯 C 导出接口。将 Python 版 [OnnxOCR](https://github.com/jingsongliujing/OnnxOCR) 的核心功能用 C++ 重新实现，编译为跨平台共享库（Windows: `OnnxOCR.dll`，Linux: `libOnnxOCR.so`，macOS: `libOnnxOCR.dylib`）。

## 功能概览

| 功能模块 | 说明 | C API 前缀 |
|---------|------|-----------|
| **文本识别 (OCR)** | 文本检测 → 角度分类 → 文本识别完整管线 | `ocr_` |
| **车牌识别** | 车牌检测 + 车牌文本识别 | `plate_` |
| **版面分析（统一）** | PP-Structure / pp_doclayoutv2 / YOLOv8 / DocLayout YOLO 版面检测 | `doclayout_yolo_` |
| **表格识别** | SLANet+/PP-Structure/UNet 表格结构识别 | `table_` |
| **Markdown 生成** | 版面分析 + OCR + 表格识别 → Markdown | `MarkdownGenerator` |

## 特性

- **纯 C 导出接口**：可在 C/C++、C#、Python ctypes、Java JNI 等环境调用
- **动态加载支持**：内置 `OcrDynamicLoader` 类，运行时加载库无需导入库
- **RAII 资源管理**：`OcrHandleGuard`/`OcrResultGuard` 等守卫类自动释放资源
- **GPU 加速**：可选 CUDA 后端（所有模块均支持）
- **PaddleOCR 兼容**：支持 PPOCRv4 / PPOCRv5 模型
- **多版面分析引擎**：PP-Structure / pp_doclayoutv2 / YOLOv8 / DocLayout YOLO（统一接口）
- **多表格识别引擎**：SLANet+（无线表格）/ UNet（有线表格）/ 自动分类模式
- **图像显示辅助**：`image_show_rects` / `image_show_rects_file` 可视化调试

## 目录结构

```
OCR/
├── CMakeLists.txt                      # 构建配置
├── README.md
├── lib/                                # 公开头文件与辅助库
│   ├── ocr.h                           # C API 定义（所有模块）
│   ├── ocr_dynamic_loader.h            # 动态加载器 + RAII 守卫
│   ├── ocr_dynamic_loader.cpp          # 动态加载器实现
│   ├── markdown_generator.h            # Markdown 生成器
│   └── markdown_generator.cpp          # Markdown 生成器实现
├── src/                                # 源码（按功能模块组织）
│   ├── ocr.cpp                         # C API 统一实现入口
│   ├── common/                         # 公共组件
│   │   ├── onnx_session.h/.cpp         #   ONNX Runtime 会话封装
│   │   ├── pre_process.h/.cpp          #   图像预处理（resize/normalize/CHW）
│   │   ├── db_post_process.h/.cpp      #   DB 后处理（轮廓提取、多边形扩展）
│   │   ├── ctc_decode.h/.cpp           #   CTC 标签解码
│   │   └── utils.h/.cpp               #   框排序、透视裁剪等工具
│   ├── text/                           # 文本识别模块
│   │   ├── text_detector.h/.cpp        #   DB 文本检测器
│   │   ├── text_recognizer.h/.cpp      #   CTC 文本识别器
│   │   ├── text_classifier.h/.cpp      #   角度分类器（0°/180°）
│   │   ├── rapid_orientation.h/.cpp    #   4方向分类器（0°/90°/180°/270°，内部使用）
│   │   ├── text_system.h/.cpp          #   OCR 管线编排
│   │   └── db_post_process.h/.cpp      #   DB 后处理
│   ├── plate/                          # 车牌识别模块
│   │   └── license_plate.h/.cpp        #   车牌检测 + 识别
│   ├── table/                          # 表格识别模块
│   │   ├── table_recognizer.h/.cpp     #   SLANet+ 表格结构识别
│   │   ├── table_cls.h/.cpp            #   表格分类器（有线/无线）
│   │   └── unet_table_rec.h/.cpp       #   UNet 有线表格识别
│   └── layout/                         # 版面分析模块
│       ├── layout_analyzer.h/.cpp      #   PP-Structure 版面分析（内部使用）
│       ├── doc_layout_analyzer.h/.cpp  #   pp_doclayoutv2 文档版面分析（内部使用）
│       └── doclayout_yolo_analyzer.h/.cpp # 版面分析统一入口（PP-Structure+pp_doclayoutv2+YOLOv8+DocLayout YOLO）
├── models/                             # 模型文件目录
│   ├── text/                           # 文本识别模型
│   ├── text_cls/                       # 文本分类器模型
│   ├── plate/                          # 车牌识别模型
│   ├── table/                          # 表格识别模型
│   └── layout/                         # 版面分析模型
└── 3rdparty/                           # 第三方依赖
    ├── onnxruntime-1.21.0/             #   ONNX Runtime 预编译库
    └── opencv-static/                  #   OpenCV 预编译静态库
```

## 构建

### 前置要求

- CMake ≥ 3.10
- C++17 编译器（Windows: MSVC 2019+，Linux: GCC 7+，macOS: Clang 10+）

### 构建步骤

Windows：

```bash
cmake -B build -G "Visual Studio 16 2019" -A x64
cmake --build build --config Release --parallel
```

Linux / macOS：

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

构建产物：

| 平台    | 输出 |
|---------|------|
| Windows | `build/bin/Release/OnnxOCR.dll` + `build/lib/Release/OnnxOCR.lib` |
| Linux   | `build/bin/libOnnxOCR.so` |
| macOS   | `build/bin/libOnnxOCR.dylib` |

## 快速开始

### 文本识别 (OCR)

```cpp
#include "ocr.h"

int main() {
    OcrConfig config = {};
    config.det_model_path       = "models/text/ppocr-ch-mobile-v4/det.onnx";
    config.rec_model_path       = "models/text/ppocr-ch-mobile-v4/rec.onnx";
    config.rec_char_dict_path   = "models/text/ppocr-ch-mobile-v4/keys.txt";
    config.cls_model_path       = "models/text_cls/angle/paddle-v4.onnx";

    OcrHandle* handle = ocr_create(&config);
    if (!handle) { printf("Error: %s\n", ocr_last_error()); return 1; }

    OcrResultList results = {};
    if (ocr_run_file(handle, "test.jpg", -1,-1,-1,-1, &results) == 0) {
        for (int i = 0; i < results.count; ++i)
            printf("[%d] text='%s' score=%.3f\n", i, results.items[i].text, results.items[i].score);
        ocr_free_results(&results);
    }
    ocr_destroy(handle);
    return 0;
}
```

### 车牌识别

```cpp
PlateConfig cfg = {};
cfg.detect_model_path = "models/car_plate/det.onnx";
cfg.rec_model_path    = "models/car_plate/rec.onnx";
cfg.min_score         = 0.4f;

PlateHandle* handle = plate_create(&cfg);
PlateResultList results = {};
plate_run_file(handle, "car.jpg", &results);
for (int i = 0; i < results.count; ++i)
    printf("Plate: %s (%s) score=%.3f\n", results.items[i].plate, results.items[i].type, results.items[i].score);
plate_free_results(&results);
plate_destroy(handle);
```

### 版面分析

PP-Structure、pp_doclayoutv2、YOLOv8 版面和 DocLayout YOLO 版面已合并为统一接口，通过 `model_type` 区分。

```cpp
// PP-Structure CDLA（中文文档）
DocLayoutYOLOConfig cfg = {};
cfg.model_path = "models/layout/pp_layout_cdla/layout_cdla.onnx";
cfg.model_type = PP_LAYOUT_CDLA;

// pp_doclayoutv2（含阅读顺序）
cfg.model_path = "models/layout/pp_doclayout_v2/doclayout_pp_v2.onnx";
cfg.model_type = PP_DOCLAYOUT_V2;

// YOLOv8-paper 版面分析
cfg.model_path = "models/yolov8_layout_paper.onnx";
cfg.model_type = YOLO_LAYOUT_PAPER;

// DocLayout YOLO 版面分析
cfg.model_path = "models/doclayout_yolo.onnx";
cfg.model_type = DOCLAYOUT_YOLO_DOCSTRUCTBENCH;
cfg.conf_thresh = 0.2f;

DocLayoutYOLOHandle* handle = doclayout_yolo_create(&cfg);
DocLayoutItemList results = {};
doclayout_yolo_run_file(handle, "doc.jpg", &results);
for (int i = 0; i < results.count; ++i)
    printf("[%d] %s order=%d (%.3f)\n",
           results.items[i].class_id, results.items[i].class_name,
           results.items[i].order, results.items[i].score);
doc_layout_free_results(&results);
doclayout_yolo_destroy(handle);
```

### 表格识别

```cpp
// SLANet+ 无线表格
TableConfig cfg = {};
cfg.model_path = "models/table/slanet_plus/slanet_plus.onnx";
cfg.model_type = TABLE_MODEL_SLANET_PLUS;

// UNet 有线表格
cfg.model_path = "models/table/unet/unet.onnx";
cfg.model_type = TABLE_MODEL_UNET;

// 自动分类模式（先分类再选模型）
cfg.model_type = TABLE_MODEL_UNET_SLANET_PLUS;
cfg.model_path      = "models/table/slanet_plus/slanet_plus.onnx";  // SLANet+模型
cfg.unet_model_path = "models/table/unet/unet.onnx";
cfg.cls_model_path  = "models/table/cls/cls.onnx";

TableHandle* handle = table_create(&cfg);
TableResult result = {};
table_run_file(handle, "table.jpg", &result);
printf("Cells: %d, Score: %.3f\n", result.cell_count, result.score);
table_free_result(&result);
table_destroy(handle);
```

### 动态加载方式（无需导入库）

```cpp
#include "ocr_dynamic_loader.h"

ocr::OcrDynamicLoader loader("OnnxOCR.dll");
ocr::OcrHandleGuard handle(loader, loader.create_or_throw(&config));
ocr::OcrResultGuard results(loader);
loader.run_file(handle.get(), "test.jpg", -1,-1,-1,-1, results.get());
```

### 从内存运行（无需图片文件）

```c
OcrResultList results;
ocr_run(handle, bgr_buffer, width, height, -1,-1,-1,-1, &results);
ocr_free_results(&results);
```

## API 参考

完整接口定义见 [lib/ocr.h](lib/ocr.h)，动态加载器见 [lib/ocr_dynamic_loader.h](lib/ocr_dynamic_loader.h)。

### 核心 API 一览

| 函数 | 说明 |
|------|------|
| **文本识别** | |
| `ocr_create` / `ocr_destroy` | 创建/销毁 OCR 引擎 |
| `ocr_run` / `ocr_run_file` | 对内存图像/文件执行 OCR |
| `ocr_free_results` | 释放结果列表 |
| **车牌识别** | |
| `plate_create` / `plate_destroy` | 创建/销毁车牌识别器 |
| `plate_run` / `plate_run_file` | 识别车牌 |
| `plate_free_results` | 释放结果列表 |
| **版面分析** | |
| `doclayout_yolo_create` / `doclayout_yolo_destroy` | 创建/销毁版面分析器 |
| `doclayout_yolo_run` / `doclayout_yolo_run_file` | 分析文档版面 |
| `doc_layout_free_results` | 释放结果列表 |
| **表格识别** | |
| `table_create` / `table_destroy` | 创建/销毁表格识别器 |
| `table_run` / `table_run_file` | 识别表格结构 |
| `table_free_result` | 释放结果 |
| **辅助** | |
| `ocr_last_error` | 获取最近错误信息（线程局部） |
| `image_show_rects` / `image_show_rects_file` | 可视化调试 |

### OcrConfig 字段

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `det_model_path` | string | — | 检测 ONNX 模型路径（必填） |
| `rec_model_path` | string | — | 识别 ONNX 模型路径（必填） |
| `rec_char_dict_path` | string | — | 字符字典路径（必填） |
| `cls_model_path` | string | NULL | 方向分类模型路径（NULL 禁用） |
| `cls_model_type` | int | 0 | `ClsModelType`枚举值：0=2类角度(0°/180°)，1=4类方向(0°/90°/180°/270°) |
| `use_gpu` | int | 0 | 0=CPU, 1=CUDA |
| `gpu_id` | int | 0 | GPU 设备号 |
| `det_limit_side_len` | float | 960 | 检测图像长边限制 |
| `det_limit_type` | string | "max" | "max" 或 "min" |
| `det_db_thresh` | float | 0.3 | DB 二值化阈值 |
| `det_db_box_thresh` | float | 0.6 | 文本框平均分阈值 |
| `det_db_unclip_ratio` | float | 1.5 | 多边形扩展比例 |
| `use_dilation` | int | 0 | 是否对分割图膨胀 |
| `det_db_score_mode` | string | "fast" | "fast" 或 "slow" |
| `det_box_type` | string | "quad" | "quad" 或 "poly" |
| `rec_batch_num` | int | 6 | 识别批大小 |
| `rec_image_h` / `rec_image_w` | int | 48 / 320 | 识别输入图像尺寸 |
| `drop_score` | float | 0.5 | 丢弃低于此分数的结果 |
| `cls_thresh` | float | 0.9 | 分类置信度阈值 |

### 结果结构

```c
// 基础几何类型
typedef struct { float x; float y; } PointF;              // 二维点
typedef struct { int x1, y1, x2, y2; } RectBox;           // 矩形区域

// OCR 结果：4角点 + 文本 + 置信度
typedef struct { PointF box[4]; const char* text; float score; } OcrResult;

// 车牌结果：框 + 置信度 + 车牌号 + 类型 + 关键点
typedef struct { RectBox box; float score; const char* plate; const char* type; PointF landmarks[4]; } PlateResult;

// 版面结果：框 + 置信度 + 类别 + 阅读顺序
typedef struct { float box[4]; float score; int class_id; const char* class_name; int order; } DocLayoutItem;

// 表格结果：单元格 + 逻辑位置
typedef struct { PointF bbox[4]; } TableCell;              // 4个角点
// logic_points 为 RectBox 数组：x1=row_start, y1=row_end, x2=col_start, y2=col_end
```

## 技术细节

### 文本识别管线

```
输入图像 (BGR)
  → TextDetector (DB算法, 检测文本框)
  → crop + TextClassifier/RapidOrientation (角度矫正)
  → TextRecognizer (CTC解码, 输出文本)
```

- **TextDetector**：DB (Differentiable Binarization) 算法，输入经 resize（长边≤960，32对齐）+ ImageNet 归一化
- **TextClassifier**：2类分类（0°/180°），输入 48×192
- **RapidOrientationClassifier**：4类方向分类（0°/90°/180°/270°），输入 224×224 center crop
- **TextRecognizer**：CTC 解码，输入 48×320（保持宽高比缩放+填充）

### 版面分析引擎对比

| 引擎 | 模型 | 输入尺寸 | 预处理 | 特点 |
|------|------|---------|--------|------|
| PP-Structure | CDLA/PublayNet | 800×608 | ImageNet 归一化 | DFL解码 + 多尺度NMS |
| pp_doclayoutv2 | doclayout_pp_v2 | 800×800 | Letterbox + /255 | 含阅读顺序，25类 |
| YOLOv8 | YOLOv8-paper/report/... | 640×640 | 直接resize + /255 | 轻量级，多种类别集 |
| DocLayout YOLO | docstructbench/d4la/... | 1024/1120/1600 | Letterbox + /255 | 高精度，11类 |


### 表格识别引擎对比

| 引擎 | 模型 | 输入尺寸 | 适用场景 |
|------|------|---------|---------|
| TableRecognizer (SLANet+) | slanet_plus.onnx | 488×488 | 无线表格（最高精度） |
| TableRecognizer (PP-Structure) | ch/en_ppstructure_*.onnx | 488×488 | 无线表格（中/英文） |
| UnetTableRecognizer | unet.onnx | 1024×1024 | 有线/边框表格 |
| 自动分类模式 | cls.onnx + UNet + SLANet+ | — | 自动判断有线/无线 |

## 许可证

遵循 OnnxOCR 项目许可证。