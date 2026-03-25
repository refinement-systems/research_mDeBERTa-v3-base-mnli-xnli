#include <onnxruntime_cxx_api.h>
#include <sentencepiece_processor.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

// Special token ids from tokenizer_config.json for this model repo.
constexpr int64_t kPadId = 0;
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

// Truncate the longer side until the pair fits.
// Total format is: [CLS] A [SEP] B [SEP]
void TruncatePair(std::vector<int64_t>& a, std::vector<int64_t>& b, size_t max_len) {
    while (a.size() + b.size() + 3 > max_len) {
        if (a.size() > b.size()) {
            a.pop_back();
        } else {
            b.pop_back();
        }
    }
}

EncodedPair BuildDebertaPair(
    sentencepiece::SentencePieceProcessor& sp,
    const std::string& premise,
    const std::string& hypothesis) {

    auto a = EncodeSentencePiece(sp, premise);
    auto b = EncodeSentencePiece(sp, hypothesis);
    TruncatePair(a, b, kMaxLen);

    EncodedPair out;
    out.input_ids.reserve(a.size() + b.size() + 3);
    out.attention_mask.reserve(a.size() + b.size() + 3);
    out.token_type_ids.reserve(a.size() + b.size() + 3);

    // [CLS]
    out.input_ids.push_back(kClsId);
    out.attention_mask.push_back(1);
    out.token_type_ids.push_back(0);

    // A
    for (auto id : a) {
        out.input_ids.push_back(id);
        out.attention_mask.push_back(1);
        out.token_type_ids.push_back(0);
    }

    // [SEP]
    out.input_ids.push_back(kSepId);
    out.attention_mask.push_back(1);
    out.token_type_ids.push_back(0);

    // B
    for (auto id : b) {
        out.input_ids.push_back(id);
        out.attention_mask.push_back(1);
        out.token_type_ids.push_back(1);
    }

    // [SEP]
    out.input_ids.push_back(kSepId);
    out.attention_mask.push_back(1);
    out.token_type_ids.push_back(1);

    return out;
}

std::vector<float> Softmax(const float* logits, size_t n) {
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
    for (auto& v : exps) {
        v /= sum;
    }
    return exps;
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
    for (const auto& s : owned) {
        names.push_back(s.c_str());
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
    for (const auto& s : owned) {
        names.push_back(s.c_str());
    }
    return names;
}

}  // namespace

int main() {
    try {
        const std::string model_path = "models/mdeberta/onnx/model_quantized.onnx";
        const std::string spm_path   = "models/mdeberta/spm.model";

        // Example NLI pair:
        const std::string premise =
            "Angela Merkel ist eine Politikerin in Deutschland und Vorsitzende der CDU";
        const std::string hypothesis =
            "Emmanuel Macron is the President of France";

        // Load tokenizer.
        sentencepiece::SentencePieceProcessor sp;
        auto sp_status = sp.Load(spm_path);
        if (!sp_status.ok()) {
            throw std::runtime_error("Failed to load SentencePiece model: " + sp_status.ToString());
        }

        EncodedPair encoded = BuildDebertaPair(sp, premise, hypothesis);
        const std::array<int64_t, 2> input_shape = {
            1, static_cast<int64_t>(encoded.input_ids.size())
        };

        // ONNX Runtime setup.
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "mdeberta_nli");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        Ort::Session session(env, model_path.c_str(), session_options);
        Ort::AllocatorWithDefaultOptions allocator;
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);

        // Build tensors.
        auto input_ids_tensor = Ort::Value::CreateTensor<int64_t>(
            mem_info,
            encoded.input_ids.data(),
            encoded.input_ids.size(),
            input_shape.data(),
            input_shape.size());

        auto attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(
            mem_info,
            encoded.attention_mask.data(),
            encoded.attention_mask.size(),
            input_shape.data(),
            input_shape.size());

        auto token_type_ids_tensor = Ort::Value::CreateTensor<int64_t>(
            mem_info,
            encoded.token_type_ids.data(),
            encoded.token_type_ids.size(),
            input_shape.data(),
            input_shape.size());

        // Match actual model input names dynamically.
        std::vector<std::string> input_name_storage;
        std::vector<const char*> input_names =
            GetInputNames(session, allocator, input_name_storage);

        std::vector<Ort::Value> input_values;
        input_values.reserve(input_names.size());

        for (const char* name : input_names) {
            if (std::strcmp(name, "input_ids") == 0) {
                input_values.emplace_back(std::move(input_ids_tensor));
            } else if (std::strcmp(name, "attention_mask") == 0) {
                input_values.emplace_back(std::move(attention_mask_tensor));
            } else if (std::strcmp(name, "token_type_ids") == 0) {
                input_values.emplace_back(std::move(token_type_ids_tensor));
            } else {
                throw std::runtime_error(std::string("Unexpected input name: ") + name);
            }
        }

        std::vector<std::string> output_name_storage;
        std::vector<const char*> output_names =
            GetOutputNames(session, allocator, output_name_storage);

        auto outputs = session.Run(
            Ort::RunOptions{nullptr},
            input_names.data(),
            input_values.data(),
            input_values.size(),
            output_names.data(),
            output_names.size());

        if (outputs.empty()) {
            throw std::runtime_error("Model returned no outputs.");
        }

        // Assume first output is logits [1, 3].
        const float* logits = outputs[0].GetTensorData<float>();
        auto info = outputs[0].GetTensorTypeAndShapeInfo();
        auto shape = info.GetShape();

        if (shape.size() != 2 || shape[0] != 1 || shape[1] != 3) {
            throw std::runtime_error("Unexpected logits shape.");
        }

        const std::vector<std::string> labels = {
            "entailment", "neutral", "contradiction"
        };
        std::vector<float> probs = Softmax(logits, 3);

        for (size_t i = 0; i < labels.size(); ++i) {
            std::cout << labels[i] << ": " << probs[i] << "\n";
        }

        auto best_it = std::max_element(probs.begin(), probs.end());
        size_t best_idx = static_cast<size_t>(std::distance(probs.begin(), best_it));
        std::cout << "predicted_label: " << labels[best_idx] << "\n";

        return 0;
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
