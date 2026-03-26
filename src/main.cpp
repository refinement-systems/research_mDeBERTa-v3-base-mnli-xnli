#include "command_line.h"
#include "nli_inference.h"

#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    try {
        const nli::ExampleCommandLineOptions options = nli::ParseExampleCommandLine(argc, argv);
        const std::string premise =
            "Angela Merkel ist eine Politikerin in Deutschland und Vorsitzende der CDU";
        const std::string hypothesis =
            "Emmanuel Macron is the President of France";
        nli::DebertaNliModel model(
            nli::DefaultModelPath(),
            nli::DefaultSentencePieceModelPath(),
            options.backend,
            std::cerr);
        const nli::NliScores scores = model.Predict(premise, hypothesis);

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
