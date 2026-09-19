#ifndef OCR_DYNAMIC_LOADER_H
#define OCR_DYNAMIC_LOADER_H

#include "ocr.h"

#include <memory>
#include <string>
#include <stdexcept>

namespace ocr {

/* 动态加载失败时抛出的异常。 */
class DynamicLoaderError : public std::runtime_error {
public:
    explicit DynamicLoaderError(const std::string& msg) : std::runtime_error(msg) {}
};

/* 在运行时动态加载OCR共享库并解析所有导出的C符号。
   这使得应用程序无需在构建时链接导入库（OnnxOCR.lib）即可使用OCR引擎 -
   DLL在运行时通过LoadLibrary/dlopen发现并加载。

   用法：
     OcrDynamicLoader loader("OnnxOCR.dll");            // 或 "libOnnxOCR.so"
     OcrHandle* h = loader.create(config);
     OcrResultList results;
     loader.run_file(h, "test.jpg", &results);
     loader.free_results(&results);
     loader.destroy(h);

   加载器在其生命周期内保持库的加载状态。销毁加载器会使通过它创建的所有句柄失效。 */
class OcrDynamicLoader {
public:
    /* 从`lib_path`加载OCR共享库。
       如果lib_path为空，则尝试默认名称：
         Windows: "OnnxOCR.dll", "OnnxOCR"
         Linux:   "libOnnxOCR.so"
         macOS:   "libOnnxOCR.dylib"
       失败时抛出DynamicLoaderError。 */
    explicit OcrDynamicLoader(const std::string& lib_path = "");
    ~OcrDynamicLoader();

    OcrDynamicLoader(const OcrDynamicLoader&) = delete;
    OcrDynamicLoader& operator=(const OcrDynamicLoader&) = delete;

    /* 库是否成功加载且所有符号已解析。 */
    bool is_loaded() const noexcept { return lib_handle_ != nullptr; }

    /* ---- OCR API ---- */

    /* 创建OCR引擎实例。出错时返回nullptr。 */
    OcrHandle* create(const OcrConfig* config) {
        return ocr_create_fn_(config);
    }

    /* 销毁OCR引擎实例。 */
    void destroy(OcrHandle* handle) {
        ocr_destroy_fn_(handle);
    }

    /* 在BGR uint8图像上运行OCR（HWC布局）。成功返回0。
       crop_x1/y1/x2/y2: 裁剪区域，-1表示使用默认值（0,0,宽,高）。 */
    int run(OcrHandle* handle,
            const unsigned char* image_data,
            int width, int height,
            int crop_x1, int crop_y1, int crop_x2, int crop_y2,
            OcrResultList* out_results) {
        return ocr_run_fn_(handle, image_data, width, height,
                           crop_x1, crop_y1, crop_x2, crop_y2, out_results);
    }

    /* 在图像文件上运行OCR。成功返回0。
       crop_x1/y1/x2/y2: 裁剪区域，-1表示使用默认值（0,0,宽,高）。 */
    int run_file(OcrHandle* handle,
                 const char* image_path,
                 int crop_x1, int crop_y1, int crop_x2, int crop_y2,
                 OcrResultList* out_results) {
        return ocr_run_file_fn_(handle, image_path,
                                crop_x1, crop_y1, crop_x2, crop_y2, out_results);
    }

    /* 释放先前由run/run_file返回的结果列表。 */
    void free_results(OcrResultList* results) {
        ocr_free_results_fn_(results);
    }

    /* 获取最后的错误信息（线程局部）。无错误时返回nullptr。 */
    const char* last_error() const {
        return last_error_fn_();
    }

    /* 便捷辅助函数，出错时抛出异常。创建实例，如果创建失败则抛出DynamicLoaderError。 */
    OcrHandle* create_or_throw(const OcrConfig* config) {
        OcrHandle* h = ocr_create_fn_(config);
        if (!h) {
            const char* err = last_error_fn_();
            throw DynamicLoaderError(err ? err : "ocr_create返回NULL");
        }
        return h;
    }

    /* ---- 车牌识别API ---- */

    /* 创建车牌识别器实例。出错时返回nullptr。 */
    PlateHandle* plate_create(const PlateConfig* config) {
        return plate_create_fn_(config);
    }

    /* 销毁车牌识别器实例。 */
    void plate_destroy(PlateHandle* handle) {
        plate_destroy_fn_(handle);
    }

    /* 在BGR uint8图像中识别车牌（HWC布局）。成功返回0。 */
    int plate_run(PlateHandle* handle,
                  const unsigned char* image_data,
                  int width, int height,
                  PlateResultList* out_results) {
        return plate_run_fn_(handle, image_data, width, height, out_results);
    }

    /* 在图像文件中识别车牌。成功返回0。 */
    int plate_run_file(PlateHandle* handle,
                       const char* image_path,
                       PlateResultList* out_results) {
        return plate_run_file_fn_(handle, image_path, out_results);
    }

    /* 释放车牌结果列表。 */
    void plate_free_results(PlateResultList* results) {
        plate_free_results_fn_(results);
    }

    /* 便捷辅助函数，出错时抛出异常。 */
    PlateHandle* plate_create_or_throw(const PlateConfig* config) {
        PlateHandle* h = plate_create_fn_(config);
        if (!h) {
            const char* err = last_error_fn_();
            throw DynamicLoaderError(err ? err : "plate_create返回NULL");
        }
        return h;
    }

    /* ---- 表格识别API ---- */

    /* 创建表格识别器实例。出错时返回nullptr。 */
    TableHandle* table_create(const TableConfig* config) {
        return table_create_fn_(config);
    }

    /* 销毁表格识别器实例。 */
    void table_destroy(TableHandle* handle) {
        table_destroy_fn_(handle);
    }

    /* 在BGR uint8图像中识别表格结构（HWC布局）。成功返回0。 */
    int table_run(TableHandle* handle,
                  const unsigned char* image_data,
                  int width, int height,
                  TableResult* out_result) {
        return table_run_fn_(handle, image_data, width, height, out_result);
    }

    /* 在图像文件中识别表格结构。成功返回0。 */
    int table_run_file(TableHandle* handle,
                       const char* image_path,
                       TableResult* out_result) {
        return table_run_file_fn_(handle, image_path, out_result);
    }

    /* 释放表格结果。 */
    void table_free_result(TableResult* result) {
        table_free_result_fn_(result);
    }

    /* 便捷辅助函数，出错时抛出异常。 */
    TableHandle* table_create_or_throw(const TableConfig* config) {
        TableHandle* h = table_create_fn_(config);
        if (!h) {
            const char* err = last_error_fn_();
            throw DynamicLoaderError(err ? err : "table_create返回NULL");
        }
        return h;
    }

    /* ---- 版面分析统一API ---- */

    DocLayoutYOLOHandle* doclayout_yolo_create(const DocLayoutYOLOConfig* config) {
        return doclayout_yolo_create_fn_(config);
    }

    void doclayout_yolo_destroy(DocLayoutYOLOHandle* handle) {
        doclayout_yolo_destroy_fn_(handle);
    }

    int doclayout_yolo_run(DocLayoutYOLOHandle* handle,
                           const unsigned char* image_data,
                           int width, int height,
                           DocLayoutItemList* out_results) {
        return doclayout_yolo_run_fn_(handle, image_data, width, height, out_results);
    }

    int doclayout_yolo_run_file(DocLayoutYOLOHandle* handle,
                                const char* image_path,
                                DocLayoutItemList* out_results) {
        return doclayout_yolo_run_file_fn_(handle, image_path, out_results);
    }

    DocLayoutYOLOHandle* doclayout_yolo_create_or_throw(const DocLayoutYOLOConfig* config) {
        DocLayoutYOLOHandle* h = doclayout_yolo_create_fn_(config);
        if (!h) {
            const char* err = last_error_fn_();
            throw DynamicLoaderError(err ? err : "doclayout_yolo_create返回NULL");
        }
        return h;
    }

    void doc_layout_free_results(DocLayoutItemList* results) {
        doc_layout_free_results_fn_(results);
    }

    /* ---- 图像显示辅助API ---- */

    /* 在窗口中显示图像并绘制矩形框。 */
    int image_show_rects(const unsigned char* image_data,
                         int width, int height,
                         const RectBox* rects, int rect_count,
                         const char* win_name,
                         int color_b, int color_g, int color_r,
                         int thickness, int wait_ms) {
        return image_show_rects_fn_(image_data, width, height,
                                    rects, rect_count, win_name,
                                    color_b, color_g, color_r,
                                    thickness, wait_ms);
    }

    /* 在窗口中显示图像文件并绘制矩形框。 */
    int image_show_rects_file(const char* image_path,
                              const RectBox* rects, int rect_count,
                              const char* win_name,
                              int color_b, int color_g, int color_r,
                              int thickness, int wait_ms) {
        return image_show_rects_file_fn_(image_path, rects, rect_count, win_name,
                                         color_b, color_g, color_r,
                                         thickness, wait_ms);
    }

private:
    // ---- OCR函数指针类型 ----
    using OcrCreateFn      = OcrHandle* (*)(const OcrConfig*);
    using OcrDestroyFn     = void (*)(OcrHandle*);
    using OcrRunFn         = int (*)(OcrHandle*, const unsigned char*, int, int, int, int, int, int, OcrResultList*);
    using OcrRunFileFn     = int (*)(OcrHandle*, const char*, int, int, int, int, OcrResultList*);
    using OcrFreeResultsFn = void (*)(OcrResultList*);
    using OcrLastErrorFn   = const char* (*)();

    // ---- 车牌识别函数指针类型 ----
    using PlateCreateFn      = PlateHandle* (*)(const PlateConfig*);
    using PlateDestroyFn     = void (*)(PlateHandle*);
    using PlateRunFn         = int (*)(PlateHandle*, const unsigned char*, int, int, PlateResultList*);
    using PlateRunFileFn     = int (*)(PlateHandle*, const char*, PlateResultList*);
    using PlateFreeResultsFn = void (*)(PlateResultList*);

    // ---- 表格识别函数指针类型 ----
    using TableCreateFn     = TableHandle* (*)(const TableConfig*);
    using TableDestroyFn    = void (*)(TableHandle*);
    using TableRunFn        = int (*)(TableHandle*, const unsigned char*, int, int, TableResult*);
    using TableRunFileFn    = int (*)(TableHandle*, const char*, TableResult*);
    using TableFreeResultFn = void (*)(TableResult*);

    // ---- 版面分析统一函数指针类型 ----
    using DocLayoutYOLOCreateFn    = DocLayoutYOLOHandle* (*)(const DocLayoutYOLOConfig*);
    using DocLayoutYOLODestroyFn   = void (*)(DocLayoutYOLOHandle*);
    using DocLayoutYOLORunFn       = int (*)(DocLayoutYOLOHandle*, const unsigned char*, int, int, DocLayoutItemList*);
    using DocLayoutYOLORunFileFn   = int (*)(DocLayoutYOLOHandle*, const char*, DocLayoutItemList*);
    using DocLayoutFreeResultsFn   = void (*)(DocLayoutItemList*);

    // ---- 图像显示函数指针类型 ----
    using ImageShowRectsFn     = int (*)(const unsigned char*, int, int,
                                         const RectBox*, int, const char*,
                                         int, int, int, int, int);
    using ImageShowRectsFileFn = int (*)(const char*,
                                         const RectBox*, int, const char*,
                                         int, int, int, int, int);

    void resolve_symbols();

    // 平台相关的库句柄（Windows上为HMODULE，Unix上为void*）
    void* lib_handle_ = nullptr;

    // ---- OCR已解析符号 ----
    OcrCreateFn      ocr_create_fn_      = nullptr;
    OcrDestroyFn     ocr_destroy_fn_     = nullptr;
    OcrRunFn         ocr_run_fn_         = nullptr;
    OcrRunFileFn     ocr_run_file_fn_    = nullptr;
    OcrFreeResultsFn ocr_free_results_fn_ = nullptr;
    OcrLastErrorFn   last_error_fn_  = nullptr;

    // ---- 车牌识别已解析符号 ----
    PlateCreateFn      plate_create_fn_      = nullptr;
    PlateDestroyFn     plate_destroy_fn_     = nullptr;
    PlateRunFn         plate_run_fn_         = nullptr;
    PlateRunFileFn     plate_run_file_fn_    = nullptr;
    PlateFreeResultsFn plate_free_results_fn_ = nullptr;

    // ---- 表格识别已解析符号 ----
    TableCreateFn     table_create_fn_     = nullptr;
    TableDestroyFn    table_destroy_fn_    = nullptr;
    TableRunFn        table_run_fn_        = nullptr;
    TableRunFileFn    table_run_file_fn_   = nullptr;
    TableFreeResultFn table_free_result_fn_ = nullptr;

    // ---- 版面分析统一已解析符号 ----
    DocLayoutYOLOCreateFn    doclayout_yolo_create_fn_    = nullptr;
    DocLayoutYOLODestroyFn   doclayout_yolo_destroy_fn_   = nullptr;
    DocLayoutYOLORunFn       doclayout_yolo_run_fn_       = nullptr;
    DocLayoutYOLORunFileFn   doclayout_yolo_run_file_fn_  = nullptr;
    DocLayoutFreeResultsFn   doc_layout_free_results_fn_  = nullptr;

    // ---- 图像显示已解析符号 ----
    ImageShowRectsFn     image_show_rects_fn_     = nullptr;
    ImageShowRectsFileFn image_show_rects_file_fn_ = nullptr;
};

/* RAII包装器，拥有通过加载器创建的OcrHandle。
   析构时自动调用destroy()。 */
class OcrHandleGuard {
public:
    OcrHandleGuard(OcrDynamicLoader& loader, OcrHandle* handle)
        : loader_(&loader), handle_(handle) {}

    ~OcrHandleGuard() {
        if (handle_ && loader_) {
            loader_->destroy(handle_);
        }
    }

    OcrHandleGuard(const OcrHandleGuard&) = delete;
    OcrHandleGuard& operator=(const OcrHandleGuard&) = delete;

    OcrHandle* get() const noexcept { return handle_; }
    OcrHandle* release() noexcept {
        OcrHandle* h = handle_;
        handle_ = nullptr;
        return h;
    }

    explicit operator bool() const noexcept { return handle_ != nullptr; }

private:
    OcrDynamicLoader* loader_;
    OcrHandle* handle_;
};

/* RAII包装器，拥有OcrResultList。
   析构时自动调用free_results()。 */
class OcrResultGuard {
public:
    OcrResultGuard(OcrDynamicLoader& loader) : loader_(&loader) {}
    ~OcrResultGuard() {
        if (loader_) loader_->free_results(&list_);
    }

    OcrResultGuard(const OcrResultGuard&) = delete;
    OcrResultGuard& operator=(const OcrResultGuard&) = delete;

    OcrResultList* get() noexcept { return &list_; }
    const OcrResultList* get() const noexcept { return &list_; }

    int count() const noexcept { return list_.count; }
    const OcrResult& at(int i) const { return list_.items[i]; }
    const OcrResult& operator[](int i) const { return list_.items[i]; }

private:
    OcrDynamicLoader* loader_;
    OcrResultList list_ = {nullptr, 0};
};

/* RAII包装器，拥有通过加载器创建的PlateHandle。
   析构时自动调用plate_destroy()。 */
class PlateHandleGuard {
public:
    PlateHandleGuard(OcrDynamicLoader& loader, PlateHandle* handle)
        : loader_(&loader), handle_(handle) {}

    ~PlateHandleGuard() {
        if (handle_ && loader_) {
            loader_->plate_destroy(handle_);
        }
    }

    PlateHandleGuard(const PlateHandleGuard&) = delete;
    PlateHandleGuard& operator=(const PlateHandleGuard&) = delete;

    PlateHandle* get() const noexcept { return handle_; }
    PlateHandle* release() noexcept {
        PlateHandle* h = handle_;
        handle_ = nullptr;
        return h;
    }

    explicit operator bool() const noexcept { return handle_ != nullptr; }

private:
    OcrDynamicLoader* loader_;
    PlateHandle* handle_;
};

/* RAII包装器，拥有PlateResultList。
   析构时自动调用plate_free_results()。 */
class PlateResultGuard {
public:
    PlateResultGuard(OcrDynamicLoader& loader) : loader_(&loader) {}
    ~PlateResultGuard() {
        if (loader_) loader_->plate_free_results(&list_);
    }

    PlateResultGuard(const PlateResultGuard&) = delete;
    PlateResultGuard& operator=(const PlateResultGuard&) = delete;

    PlateResultList* get() noexcept { return &list_; }
    const PlateResultList* get() const noexcept { return &list_; }

    int count() const noexcept { return list_.count; }
    const PlateResult& at(int i) const { return list_.items[i]; }
    const PlateResult& operator[](int i) const { return list_.items[i]; }

private:
    OcrDynamicLoader* loader_;
    PlateResultList list_ = {nullptr, 0};
};

/* RAII包装器，拥有通过加载器创建的TableHandle。
   析构时自动调用table_destroy()。 */
class TableHandleGuard {
public:
    TableHandleGuard(OcrDynamicLoader& loader, TableHandle* handle)
        : loader_(&loader), handle_(handle) {}

    ~TableHandleGuard() {
        if (handle_ && loader_) {
            loader_->table_destroy(handle_);
        }
    }

    TableHandleGuard(const TableHandleGuard&) = delete;
    TableHandleGuard& operator=(const TableHandleGuard&) = delete;

    TableHandle* get() const noexcept { return handle_; }
    TableHandle* release() noexcept {
        TableHandle* h = handle_;
        handle_ = nullptr;
        return h;
    }

    explicit operator bool() const noexcept { return handle_ != nullptr; }

private:
    OcrDynamicLoader* loader_;
    TableHandle* handle_;
};

/* RAII包装器，拥有TableResult。
   析构时自动调用table_free_result()。 */
class TableResultGuard {
public:
    TableResultGuard(OcrDynamicLoader& loader) : loader_(&loader) {}
    ~TableResultGuard() {
        if (loader_) loader_->table_free_result(&result_);
    }

    TableResultGuard(const TableResultGuard&) = delete;
    TableResultGuard& operator=(const TableResultGuard&) = delete;

    TableResult* get() noexcept { return &result_; }
    const TableResult* get() const noexcept { return &result_; }

private:
    OcrDynamicLoader* loader_;
    TableResult result_ = {nullptr, nullptr, 0, 0.0f};
};

/* RAII包装器，拥有DocLayoutItemList。
   析构时自动调用doc_layout_free_results()。 */
class DocLayoutResultGuard {
public:
    DocLayoutResultGuard(OcrDynamicLoader& loader) : loader_(&loader) {}
    ~DocLayoutResultGuard() {
        if (loader_) loader_->doc_layout_free_results(&list_);
    }

    DocLayoutResultGuard(const DocLayoutResultGuard&) = delete;
    DocLayoutResultGuard& operator=(const DocLayoutResultGuard&) = delete;

    DocLayoutItemList* get() noexcept { return &list_; }
    const DocLayoutItemList* get() const noexcept { return &list_; }

    int count() const noexcept { return list_.count; }
    const DocLayoutItem& at(int i) const { return list_.items[i]; }
    const DocLayoutItem& operator[](int i) const { return list_.items[i]; }

private:
    OcrDynamicLoader* loader_;
    DocLayoutItemList list_ = {nullptr, 0};
};

/* RAII包装器，拥有通过加载器创建的DocLayoutYOLOHandle。 */
class DocLayoutYOLOHandleGuard {
public:
    DocLayoutYOLOHandleGuard(OcrDynamicLoader& loader, DocLayoutYOLOHandle* handle)
        : loader_(&loader), handle_(handle) {}

    ~DocLayoutYOLOHandleGuard() {
        if (handle_ && loader_) {
            loader_->doclayout_yolo_destroy(handle_);
        }
    }

    DocLayoutYOLOHandleGuard(const DocLayoutYOLOHandleGuard&) = delete;
    DocLayoutYOLOHandleGuard& operator=(const DocLayoutYOLOHandleGuard&) = delete;

    DocLayoutYOLOHandle* get() const noexcept { return handle_; }
    DocLayoutYOLOHandle* release() noexcept {
        DocLayoutYOLOHandle* h = handle_;
        handle_ = nullptr;
        return h;
    }

    explicit operator bool() const noexcept { return handle_ != nullptr; }

private:
    OcrDynamicLoader* loader_;
    DocLayoutYOLOHandle* handle_;
};

} // namespace ocr

#endif /* OCR_DYNAMIC_LOADER_H */