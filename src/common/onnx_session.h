#ifndef ONNX_SESSION_H
#define ONNX_SESSION_H

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <onnxruntime_cxx_api.h>

namespace ocr {

/* RAII wrapper around an ONNX Runtime inference session.
   Handles model loading, input/output name caching, and inference. */
class OnnxSession {
public:
    OnnxSession(const std::string& model_path, bool use_gpu, int gpu_id);
    ~OnnxSession();

    OnnxSession(const OnnxSession&) = delete;
    OnnxSession& operator=(const OnnxSession&) = delete;

    /* Run inference with a single input. `inputs` is a flat float vector; the
       input shape is described by `input_shape`. Returns the output tensor(s). */
    std::vector<Ort::Value> run(
        const std::vector<int64_t>& input_shape,
        const std::vector<float>& input_data);

    /* Run inference with multiple inputs. Each pair of (shape, data) forms one
       input tensor, in the same order as the model's input names. */
    std::vector<Ort::Value> run_multi(
        const std::vector<std::vector<int64_t>>& input_shapes,
        const std::vector<std::vector<float>>& input_data_list);

    std::vector<int64_t> get_input_shape(size_t index = 0) const;
    const std::string& get_input_name(size_t index = 0) const { return input_names_[index]; }
    const std::vector<std::string>& get_input_names() const { return input_names_; }
    const std::vector<std::string>& get_output_names() const { return output_names_; }

    /* Get custom metadata map from the model. */
    std::unordered_map<std::string, std::string> get_metadata() const;

private:
    /* Ort::Env 不能作为静态全局成员在 DllMain 期间构造——ORT 静态状态可能
       尚未初始化，会导致 DLL 加载失败(ERROR_DLL_INIT_FAILED 1114)。
       改为函数局部 static 惰性初始化：首次调用 OnnxSession 构造时才创建，
       若失败则抛出异常由 ocr_create 捕获并报告。 */
    static Ort::Env& env();
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<const char*> input_name_ptrs_;
    std::vector<const char*> output_name_ptrs_;
};

} // namespace ocr

#endif /* ONNX_SESSION_H */
