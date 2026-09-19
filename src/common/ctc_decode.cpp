#include "ctc_decode.h"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

namespace ocr {

CTCLabelDecode::CTCLabelDecode(const std::string& char_dict_path, bool use_space_char)
    : use_space_char_(use_space_char) {
    load_dict(char_dict_path);
}

void CTCLabelDecode::load_dict(const std::string& path) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin.is_open()) {
        throw std::runtime_error("Cannot open character dictionary: " + path);
    }

    std::vector<std::string> chars;
    std::string line;
    while (std::getline(fin, line)) {
        // Strip \r and \n
        if (!line.empty() && line.back() == '\r') line.pop_back();
        chars.push_back(line);
    }

    if (use_space_char_) {
        chars.push_back(" ");
    }

    // Add blank token at index 0 (CTC blank)
    character_.clear();
    character_.push_back("blank");
    for (const auto& c : chars) {
        character_.push_back(c);
    }
}

std::vector<CTCLabelDecode::Result> CTCLabelDecode::decode(
    const float* preds, int batch_size, int seq_len, int num_classes) const {

    std::vector<Result> results;
    results.reserve(batch_size);

    // preds is [batch_size, seq_len, num_classes]
    for (int b = 0; b < batch_size; ++b) {
        const float* batch_pred = preds + b * seq_len * num_classes;

        std::string text;
        std::vector<float> confs;
        int prev_id = -1;

        for (int t = 0; t < seq_len; ++t) {
            const float* step_pred = batch_pred + t * num_classes;

            int max_id = 0;
            float max_val = step_pred[0];
            for (int c = 1; c < num_classes; ++c) {
                if (step_pred[c] > max_val) {
                    max_val = step_pred[c];
                    max_id = c;
                }
            }

            if (max_id == 0) {
                prev_id = max_id;
                continue;
            }

            if (max_id == prev_id) {
                continue;
            }
            prev_id = max_id;

            if (max_id < static_cast<int>(character_.size())) {
                text += character_[max_id];
                confs.push_back(max_val);
            }
        }

        float mean_conf = 0.0f;
        if (!confs.empty()) {
            float sum = 0.0f;
            for (float c : confs) sum += c;
            mean_conf = sum / confs.size();
        }

        results.push_back({text, mean_conf});
    }
    return results;
}

} // namespace ocr