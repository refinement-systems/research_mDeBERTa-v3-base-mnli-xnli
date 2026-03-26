#include "nli_inference.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int64_t kClsId = 1;
constexpr int64_t kSepId = 2;
constexpr size_t kMaxLen = 512;

struct EncodedPair {
    std::vector<int64_t> input_ids;
    std::vector<int64_t> attention_mask;
    std::vector<int64_t> token_type_ids;
};

std::vector<int64_t> EncodeSentencePiece(
    sentencepiece::SentencePieceProcessor& sp,
    const std::string& text) {
    std::vector<int> ids;
    auto status = sp.Encode(text, &ids);
    if (!status.ok()) {
        throw std::runtime_error("SentencePiece encode failed: " + status.ToString());
    }
    return std::vector<int64_t>(ids.begin(), ids.end());
}

void TruncatePair(std::vector<int64_t>& premise, std::vector<int64_t>& hypothesis, size_t max_len) {
    while (premise.size() + hypothesis.size() + 3 > max_len) {
        if (premise.size() > hypothesis.size()) {
            premise.pop_back();
        } else {
            hypothesis.pop_back();
        }
    }
}

EncodedPair BuildDebertaPair(
    sentencepiece::SentencePieceProcessor& sp,
    const std::string& premise,
    const std::string& hypothesis) {
    auto premise_ids = EncodeSentencePiece(sp, premise);
    auto hypothesis_ids = EncodeSentencePiece(sp, hypothesis);
    TruncatePair(premise_ids, hypothesis_ids, kMaxLen);

    EncodedPair out;
    out.input_ids.reserve(premise_ids.size() + hypothesis_ids.size() + 3);
    out.attention_mask.reserve(premise_ids.size() + hypothesis_ids.size() + 3);
    out.token_type_ids.reserve(premise_ids.size() + hypothesis_ids.size() + 3);

    out.input_ids.push_back(kClsId);
    out.attention_mask.push_back(1);
    out.token_type_ids.push_back(0);

    for (auto id : premise_ids) {
        out.input_ids.push_back(id);
        out.attention_mask.push_back(1);
        out.token_type_ids.push_back(0);
    }

    out.input_ids.push_back(kSepId);
    out.attention_mask.push_back(1);
    out.token_type_ids.push_back(0);

    for (auto id : hypothesis_ids) {
        out.input_ids.push_back(id);
        out.attention_mask.push_back(1);
        out.token_type_ids.push_back(1);
    }

    out.input_ids.push_back(kSepId);
    out.attention_mask.push_back(1);
    out.token_type_ids.push_back(1);

    return out;
}

nli::NliScores SoftmaxScores(const float* logits, size_t n) {
    float max_logit = logits[0];
    for (size_t i = 1; i < n; ++i) {
        max_logit = std::max(max_logit, logits[i]);
    }

    std::vector<float> exps(n);
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        exps[i] = std::exp(logits[i] - max_logit);
        sum += exps[i];
    }
    for (auto& value : exps) {
        value /= sum;
    }

    return nli::NliScores{
        exps[0],
        exps[1],
        exps[2],
    };
}

std::vector<const char*> GetInputNames(
    Ort::Session& session,
    Ort::AllocatorWithDefaultOptions& allocator,
    std::vector<std::string>& owned) {
    owned.clear();

    const size_t count = session.GetInputCount();
    owned.reserve(count);

    std::vector<const char*> names;
    names.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        auto name = session.GetInputNameAllocated(i, allocator);
        owned.emplace_back(name.get());
    }
    for (const auto& name : owned) {
        names.push_back(name.c_str());
    }

    return names;
}

std::vector<const char*> GetOutputNames(
    Ort::Session& session,
    Ort::AllocatorWithDefaultOptions& allocator,
    std::vector<std::string>& owned) {
    owned.clear();

    const size_t count = session.GetOutputCount();
    owned.reserve(count);

    std::vector<const char*> names;
    names.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        auto name = session.GetOutputNameAllocated(i, allocator);
        owned.emplace_back(name.get());
    }
    for (const auto& name : owned) {
        names.push_back(name.c_str());
    }

    return names;
}

}  // namespace

namespace nli {

const char* DefaultModelPath() {
    return "models/mdeberta/onnx/model_quantized.onnx";
}

const char* DefaultSentencePieceModelPath() {
    return "models/mdeberta/spm.model";
}

std::string_view PredictedLabel(const NliScores& scores) {
    const std::array<float, 3> values = {
        scores.entailment,
        scores.neutral,
        scores.contradiction,
    };
    auto best_it = std::max_element(values.begin(), values.end());
    const size_t best_idx = static_cast<size_t>(std::distance(values.begin(), best_it));
    return kNliScoreLabels[best_idx];
}

DebertaNliModel::DebertaNliModel(
    const std::string& model_path,
    const std::string& sentencepiece_path,
    SessionBackend backend,
    std::ostream& log)
    : env_(ORT_LOGGING_LEVEL_WARNING, "mdeberta_nli"),
      session_(nullptr),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    auto sp_status = sp_.Load(sentencepiece_path);
    if (!sp_status.ok()) {
        throw std::runtime_error("Failed to load SentencePiece model: " + sp_status.ToString());
    }

    auto session_result = CreateInferenceSession(env_, model_path, backend, log);
    session_ = std::move(session_result.value);

    input_names_ = GetInputNames(session_, allocator_, input_name_storage_);
    output_names_ = GetOutputNames(session_, allocator_, output_name_storage_);
}

NliScores DebertaNliModel::Predict(const std::string& premise, const std::string& hypothesis) {
    EncodedPair encoded = BuildDebertaPair(sp_, premise, hypothesis);
    const std::array<int64_t, 2> input_shape = {
        1, static_cast<int64_t>(encoded.input_ids.size())
    };

    auto input_ids_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info_,
        encoded.input_ids.data(),
        encoded.input_ids.size(),
        input_shape.data(),
        input_shape.size());

    auto attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info_,
        encoded.attention_mask.data(),
        encoded.attention_mask.size(),
        input_shape.data(),
        input_shape.size());

    auto token_type_ids_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info_,
        encoded.token_type_ids.data(),
        encoded.token_type_ids.size(),
        input_shape.data(),
        input_shape.size());

    std::vector<Ort::Value> input_values;
    input_values.reserve(input_names_.size());

    for (const auto& input_name : input_name_storage_) {
        if (input_name == "input_ids") {
            input_values.emplace_back(std::move(input_ids_tensor));
        } else if (input_name == "attention_mask") {
            input_values.emplace_back(std::move(attention_mask_tensor));
        } else if (input_name == "token_type_ids") {
            input_values.emplace_back(std::move(token_type_ids_tensor));
        } else {
            throw std::runtime_error("Unexpected input name: " + input_name);
        }
    }

    auto outputs = session_.Run(
        Ort::RunOptions{nullptr},
        input_names_.data(),
        input_values.data(),
        input_values.size(),
        output_names_.data(),
        output_names_.size());

    if (outputs.empty()) {
        throw std::runtime_error("Model returned no outputs.");
    }

    const float* logits = outputs[0].GetTensorData<float>();
    auto info = outputs[0].GetTensorTypeAndShapeInfo();
    const auto shape = info.GetShape();

    if (shape.size() != 2 || shape[0] != 1 || shape[1] != 3) {
        throw std::runtime_error("Unexpected logits shape.");
    }

    return SoftmaxScores(logits, 3);
}

}  // namespace nli
