#include "command_line.h"
#include "nli_inference.h"
#include "topical_chat.h"

#include <iostream>

int main(int argc, char* argv[]) {
    try {
        const nli::TopicalChatCommandLineOptions options =
            nli::ParseTopicalChatCommandLine(argc, argv);
        const auto turns = nli::ReadTopicalChatTurnInputs(options.input_path);

        nli::DebertaNliModel model(
            options.model_path,
            nli::DefaultSentencePieceModelPath(),
            options.backend,
            std::cerr);

        std::cout << "entailment,neutral,contradiction\n";
        for (const auto& turn : turns) {
            const nli::NliScores scores = model.Predict(turn.premise, turn.hypothesis);
            std::cout
                << scores.entailment << ','
                << scores.neutral << ','
                << scores.contradiction << '\n';
        }

        return 0;
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
