#include "onnx_session.h"

#include <stdexcept>

namespace ocr {

/* 惰性创建的 Ort::Env。函数局部 static 在首次调用 OnnxSession 构造时
   才构造（C++11 线程安全），避免在 DllMain 期间初始化 ORT 运行时。
   日志级别设为 ERROR，屏蔽 paddle2onnx 导出模型遗留的标量形状警告
   （logical_and/fill_constant 等，ORT 会自动 lenient merge 兜底）。 */
Ort::Env& OnnxSession::env() {
    static Ort::Env env_instance(ORT_LOGGING_LEVEL_ERROR, "ocr");
    return env_instance;
}

OnnxSession::OnnxSession(const std::string& model_path, bool use_gpu, int gpu_id)
    : session_(nullptr) {

    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(4);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    opts.SetExecutionMode(ORT_SEQUENTIAL);
    opts.EnableMemPattern();

    if (use_gpu) {
        OrtCUDAProviderOptions cuda_opts;
        cuda_opts.device_id = gpu_id;
        cuda_opts.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
        opts.AppendExecutionProvider_CUDA(cuda_opts);
    }

#ifdef _WIN32
    std::wstring wide_path(model_path.begin(), model_path.end());
    session_ = std::make_unique<Ort::Session>(env(), wide_path.c_str(), opts);
#else
    session_ = std::make_unique<Ort::Session>(env(), model_path.c_str(), opts);
#endif

    // Cache input names
    Ort::AllocatorWithDefaultOptions alloc;
    size_t num_inputs = session_->GetInputCount();
    size_t num_outputs = session_->GetOutputCount();

    for (size_t i = 0; i < num_inputs; ++i) {
        auto name_ptr = session_->GetInputNameAllocated(i, alloc);
        input_names_.push_back(name_ptr.get());
    }
    for (size_t i = 0; i < num_outputs; ++i) {
        auto name_ptr = session_->GetOutputNameAllocated(i, alloc);
        output_names_.push_back(name_ptr.get());
    }

    input_name_ptrs_.reserve(input_names_.size());
    for (const auto& name : input_names_) {
        input_name_ptrs_.push_back(name.c_str());
    }
    output_name_ptrs_.reserve(output_names_.size());
    for (const auto& name : output_names_) {
        output_name_ptrs_.push_back(name.c_str());
    }
}

OnnxSession::~OnnxSession() = default;

std::vector<Ort::Value> OnnxSession::run(
    const std::vector<int64_t>& input_shape,
    const std::vector<float>& input_data) {

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input_tensor = Ort::Value::CreateTensor<float>(
        mem_info,
        const_cast<float*>(input_data.data()),
        input_data.size(),
        input_shape.data(),
        input_shape.size());

    return session_->Run(
        Ort::RunOptions{nullptr},
        input_name_ptrs_.data(),
        &input_tensor,
        1,
        output_name_ptrs_.data(),
        output_name_ptrs_.size());
}

std::vector<Ort::Value> OnnxSession::run_multi(
    const std::vector<std::vector<int64_t>>& input_shapes,
    const std::vector<std::vector<float>>& input_data_list) {

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<Ort::Value> input_tensors;
    input_tensors.reserve(input_shapes.size());
    for (size_t i = 0; i < input_shapes.size(); ++i) {
        auto tensor = Ort::Value::CreateTensor<float>(
            mem_info,
            const_cast<float*>(input_data_list[i].data()),
            input_data_list[i].size(),
            input_shapes[i].data(),
            input_shapes[i].size());
        input_tensors.push_back(std::move(tensor));
    }

    return session_->Run(
        Ort::RunOptions{nullptr},
        input_name_ptrs_.data(),
        input_tensors.data(),
        input_tensors.size(),
        output_name_ptrs_.data(),
        output_name_ptrs_.size());
}

std::vector<int64_t> OnnxSession::get_input_shape(size_t index) const {
    auto info = session_->GetInputTypeInfo(index);
    auto tensor_info = info.GetTensorTypeAndShapeInfo();
    return tensor_info.GetShape();
}

std::unordered_map<std::string, std::string> OnnxSession::get_metadata() const {
    Ort::AllocatorWithDefaultOptions alloc;
    Ort::ModelMetadata meta = session_->GetModelMetadata();
    std::unordered_map<std::string, std::string> result;

    auto keys = meta.GetCustomMetadataMapKeysAllocated(alloc);
    for (size_t i = 0; i < keys.size(); ++i) {
        auto val = meta.LookupCustomMetadataMapAllocated(keys[i].get(), alloc);
        result[keys[i].get()] = val.get();
    }
    return result;
}

} // namespace ocr
