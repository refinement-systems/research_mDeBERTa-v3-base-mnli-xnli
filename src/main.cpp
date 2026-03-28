#include "command_line.h"
#include "nli_inference.h"

#include <iostream>
#include <string>
#include <vector>

namespace {

void PrintVector(const std::string& name, const std::vector<int64_t>& values) {
    std::cout << name << ":";
    for (const auto value : values) {
        std::cout << ' ' << value;
    }
    std::cout << "\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        const nli::ExampleCommandLineOptions options = nli::ParseExampleCommandLine(argc, argv);
        nli::DebertaNliModel model(
            options.model_path,
            nli::DefaultSentencePieceModelPath(),
            options.backend,
            std::cerr);
        if (options.dump_special_token_ids) {
            const auto special_token_ids = model.GetSpecialTokenIds();
            std::cout << "special_token_ids:"
                      << " pad=" << special_token_ids.pad
                      << " cls=" << special_token_ids.cls
                      << " sep=" << special_token_ids.sep
                      << " unk=" << special_token_ids.unk
                      << " mask=" << special_token_ids.mask
                      << "\n";
        }
        if (options.dump_encoding) {
            const nli::EncodedInputs encoded = model.Encode(options.premise, options.hypothesis);
            std::cout << "normalized_premise: " << encoded.normalized_premise << "\n";
            std::cout << "normalized_hypothesis: " << encoded.normalized_hypothesis << "\n";
            PrintVector("input_ids", encoded.input_ids);
            PrintVector("attention_mask", encoded.attention_mask);
            PrintVector("token_type_ids", encoded.token_type_ids);
        }
        const nli::NliScores scores = model.Predict(options.premise, options.hypothesis);

        std::cout << "entailment: " << scores.entailment << "\n";
        std::cout << "neutral: " << scores.neutral << "\n";
        std::cout << "contradiction: " << scores.contradiction << "\n";
        std::cout << "predicted_label: " << nli::PredictedLabel(scores) << "\n";

        return 0;
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
