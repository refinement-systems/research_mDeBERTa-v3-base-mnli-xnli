#pragma once

#include "session_factory.h"
#include "tokenizer_assets.h"

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <sentencepiece_processor.h>

#include <array>
#include <iosfwd>
#include <string>
#include <string_view>
#include <vector>

namespace nli {

struct NliLogits {
    float entailment;
    float neutral;
    float contradiction;
};

struct NliScores {
    float entailment;
    float neutral;
    float contradiction;
};

struct EncodedInputs {
    std::string normalized_premise;
    std::string normalized_hypothesis;
    std::vector<int64_t> input_ids;
    std::vector<int64_t> attention_mask;
    std::vector<int64_t> token_type_ids;
};

inline constexpr std::array<std::string_view, 3> kNliScoreLabels = {
    "entailment",
    "neutral",
    "contradiction",
};

const char* DefaultModelPath();
const char* DefaultSentencePieceModelPath();

NliScores ScoresFromLogits(const NliLogits& logits);
std::string_view PredictedLabel(const NliLogits& logits);
std::string_view PredictedLabel(const NliScores& scores);

class DebertaNliModel {
public:
    DebertaNliModel(
        const std::string& model_path,
        const std::string& sentencepiece_path,
        SessionBackend backend,
        std::ostream& log);

    EncodedInputs Encode(const std::string& premise, const std::string& hypothesis);
    TokenizerSpecialTokenIds GetSpecialTokenIds() const;
    NliLogits PredictLogits(const std::string& premise, const std::string& hypothesis);
    NliScores Predict(const std::string& premise, const std::string& hypothesis);

private:
    sentencepiece::SentencePieceProcessor sp_;
    Ort::Env env_;
    Ort::Session session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    Ort::MemoryInfo memory_info_;
    std::vector<std::string> input_name_storage_;
    std::vector<const char*> input_names_;
    std::vector<std::string> output_name_storage_;
    std::vector<const char*> output_names_;
    TokenizerAssetConfig tokenizer_config_;
};

}  // namespace nli
