#ifndef TABLE_RECOGNIZER_H
#define TABLE_RECOGNIZER_H

#include <memory>
#include <string>
#include <vector>
#include "onnx_session.h"
#include <opencv2/opencv.hpp>

namespace ocr {

/* Table recognition result: cell bounding boxes and logic points. */
struct TableResult {
    std::vector<std::vector<float>> cell_bboxes; /* Each cell: [x1,y1,x2,y2,...] (8 for quad) */
    std::vector<std::vector<float>> logic_points; /* Each cell: [row_start, row_end, col_start, col_end] */
    float score;                        /* Mean structure confidence */
};

/* Table model type selector.
   SLANet-family models (incl. ppstructure_zh/en) all use SLANET_PLUS;
   specify different model files via model_path. */
enum class TableModelType {
    SLANET_PLUS,            /* SLANet-family (slanet-plus, ppstructure_zh/en, wireless tables) */
    UNET,                   /* unet.onnx (wired/bordered table structure) */
    UNET_SLANET_PLUS,       /* combined: classify first, then UNet for wired, SLANet+ for wireless */
};

/* Table recognizer using PP-Structure SLANet models.
   Input: 488x488, ImageNet normalization.
   Output: 2 tensors (bbox_preds [1,N,8], structure_probs [1,N,num_classes]).
   Post-processing: structure token decode -> HTML, bbox decode -> cell boxes. */
class TableRecognizer {
public:
    struct Config {
        std::string model_path;             /* table structure ONNX model path (required).
                                               SLANET_PLUS/UNET: the model path;
                                               UNET_SLANET_PLUS: SLANet+ model path */
        TableModelType model_type = TableModelType::SLANET_PLUS;
        std::string cls_model_path;         /* table classifier model path (for UNET_SLANET_PLUS) */
        std::string unet_model_path;        /* UNet model path (for UNET_SLANET_PLUS) */
        bool   use_gpu = false;
        int    gpu_id = 0;
    };

    explicit TableRecognizer(const Config& cfg);
    ~TableRecognizer();

    TableRecognizer(const TableRecognizer&) = delete;
    TableRecognizer& operator=(const TableRecognizer&) = delete;

    /* Recognize table structure in a BGR image.
       Returns HTML, cell bboxes, and logic points. */
    TableResult recognize(const cv::Mat& img);

    /* Get the character list loaded from the model. */
    const std::vector<std::string>& get_characters() const { return characters_; }

private:
    std::unique_ptr<OnnxSession> session_;
    std::vector<std::string> characters_;  /* structure tokens including sos/eos */
    int beg_idx_;                           /* sos index */
    int end_idx_;                           /* eos index */
    TableModelType model_type_;             /* active model type */
    std::unique_ptr<class TableClassifier> classifier_;  /* for UNET_SLANET_PLUS */
    std::unique_ptr<class UnetTableRecognizer> unet_rec_; /* for UNET / UNET_SLANET_PLUS */
    static constexpr int MAX_LEN = 488;     /* input size */
    static constexpr float MEAN[3] = {0.485f, 0.456f, 0.406f};
    static constexpr float STD[3]  = {0.229f, 0.224f, 0.225f};

    /* Load character dict from model metadata and warm up. */
    void load_characters();

    /* Decode structure tokens and cell bboxes from model outputs. */
    void decode(const float* bbox_preds, const float* struct_probs,
                int num_tokens, int num_classes,
                int ori_h, int ori_w,
                std::vector<std::string>& structure_list,
                std::vector<std::vector<float>>& cell_bboxes,
                float& mean_score);

    /* Decode logic points (row/col spans) from structure tokens. */
    std::vector<std::vector<float>> decode_logic_points(
        const std::vector<std::string>& structure_list);
};

} // namespace ocr

#endif /* TABLE_RECOGNIZER_H */