#ifndef CTC_DECODE_H
#define CTC_DECODE_H

#include <string>
#include <vector>

namespace ocr {

/* CTC label decoder (compatible with PaddleOCR's CTCLabelDecode).
   Loads a character dictionary and decodes model output. */
class CTCLabelDecode {
public:
    CTCLabelDecode(const std::string& char_dict_path, bool use_space_char);

    /* Decode the model output.
       preds: [batch, seq_len, num_classes] flat float data
       batch_size, seq_len, num_classes: dimensions
       Returns a vector of (text, score) pairs. */
    struct Result { std::string text; float score; };
    std::vector<Result> decode(const float* preds, int batch_size, int seq_len, int num_classes) const;

private:
    void load_dict(const std::string& path);

    std::vector<std::string> character_;  // index -> character string
    bool use_space_char_;
};

} // namespace ocr

#endif /* CTC_DECODE_H */
