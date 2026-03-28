#include "command_line.h"
#include "nli_eval.h"
#include "nli_inference.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

size_t LabelIndex(std::string_view label) {
    const auto it = std::find(
        nli::kNliScoreLabels.begin(),
        nli::kNliScoreLabels.end(),
        label);
    if (it == nli::kNliScoreLabels.end()) {
        throw std::runtime_error("Unsupported label: " + std::string(label));
    }
    return static_cast<size_t>(std::distance(nli::kNliScoreLabels.begin(), it));
}

float LabelScore(const nli::NliScores& scores, std::string_view label) {
    if (label == "entailment") {
        return scores.entailment;
    }
    if (label == "neutral") {
        return scores.neutral;
    }
    if (label == "contradiction") {
        return scores.contradiction;
    }

    throw std::runtime_error("Unsupported label: " + std::string(label));
}

struct ModelStats {
    std::string name;
    std::array<size_t, 3> label_counts = {0, 0, 0};
    size_t labeled_examples = 0;
    size_t correct_predictions = 0;
};

void RecordPrediction(
    ModelStats& stats,
    const std::string_view predicted_label,
    const std::optional<std::string>& gold_label) {
    ++stats.label_counts[LabelIndex(predicted_label)];
    if (gold_label) {
        ++stats.labeled_examples;
        if (predicted_label == *gold_label) {
            ++stats.correct_predictions;
        }
    }
}

void PrintModelStats(const ModelStats& stats) {
    std::cout << stats.name << "_predictions:";
    for (size_t i = 0; i < nli::kNliScoreLabels.size(); ++i) {
        std::cout << ' ' << nli::kNliScoreLabels[i] << '=' << stats.label_counts[i];
    }
    std::cout << '\n';

    if (stats.labeled_examples == 0) {
        std::cout << stats.name << "_accuracy: n/a\n";
        return;
    }

    const double accuracy = static_cast<double>(stats.correct_predictions) /
                            static_cast<double>(stats.labeled_examples);
    std::cout << std::fixed << std::setprecision(4)
              << stats.name << "_accuracy: " << accuracy
              << " (" << stats.correct_predictions << '/' << stats.labeled_examples << ")\n"
              << std::defaultfloat;
}

struct Disagreement {
    size_t example_index;
    std::optional<std::string> id;
    std::optional<std::string> gold_label;
    std::string premise;
    std::string hypothesis;
    std::string primary_label;
    float primary_score;
    std::string compare_label;
    float compare_score;
};

void PrintDisagreement(const Disagreement& disagreement) {
    std::cout << "disagreement[" << disagreement.example_index << "]";
    if (disagreement.id) {
        std::cout << " id=" << *disagreement.id;
    }
    if (disagreement.gold_label) {
        std::cout << " gold=" << *disagreement.gold_label;
    }
    std::cout << '\n';
    std::cout << "  primary=" << disagreement.primary_label
              << " (" << disagreement.primary_score << ")\n";
    std::cout << "  compare=" << disagreement.compare_label
              << " (" << disagreement.compare_score << ")\n";
    std::cout << "  premise=" << disagreement.premise << '\n';
    std::cout << "  hypothesis=" << disagreement.hypothesis << '\n';
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        const nli::EvalCommandLineOptions options = nli::ParseEvalCommandLine(argc, argv);
        const auto examples = nli::ReadNliEvalExamples(options.input_path);

        nli::DebertaNliModel primary_model(
            options.model_path,
            nli::DefaultSentencePieceModelPath(),
            options.backend,
            std::cerr);

        std::optional<nli::DebertaNliModel> compare_model;
        if (!options.compare_model_path.empty()) {
            compare_model.emplace(
                options.compare_model_path,
                nli::DefaultSentencePieceModelPath(),
                options.backend,
                std::cerr);
        }

        ModelStats primary_stats{"primary"};
        ModelStats compare_stats{"compare"};
        size_t model_agreements = 0;
        std::vector<Disagreement> disagreements;
        disagreements.reserve(options.max_disagreements);

        for (size_t i = 0; i < examples.size(); ++i) {
            const auto& example = examples[i];
            const auto primary_scores = primary_model.Predict(example.premise, example.hypothesis);
            const std::string primary_label(nli::PredictedLabel(primary_scores));
            RecordPrediction(primary_stats, primary_label, example.label);

            if (!compare_model) {
                continue;
            }

            const auto compare_scores = compare_model->Predict(example.premise, example.hypothesis);
            const std::string compare_label(nli::PredictedLabel(compare_scores));
            RecordPrediction(compare_stats, compare_label, example.label);

            if (primary_label == compare_label) {
                ++model_agreements;
                continue;
            }

            if (disagreements.size() < options.max_disagreements) {
                disagreements.push_back(Disagreement{
                    i + 1,
                    example.id.empty() ? std::nullopt : std::optional<std::string>(example.id),
                    example.label,
                    example.premise,
                    example.hypothesis,
                    primary_label,
                    LabelScore(primary_scores, primary_label),
                    compare_label,
                    LabelScore(compare_scores, compare_label),
                });
            }
        }

        std::cout << "examples: " << examples.size() << '\n';
        const size_t labeled_examples = std::count_if(
            examples.begin(),
            examples.end(),
            [](const nli::NliEvalExample& example) { return example.label.has_value(); });
        std::cout << "labeled_examples: " << labeled_examples << '\n';
        std::cout << "primary_model: " << options.model_path << '\n';
        PrintModelStats(primary_stats);

        if (compare_model) {
            std::cout << "compare_model: " << options.compare_model_path << '\n';
            PrintModelStats(compare_stats);

            const double agreement = examples.empty()
                                         ? 0.0
                                         : static_cast<double>(model_agreements) /
                                               static_cast<double>(examples.size());
            std::cout << std::fixed << std::setprecision(4)
                      << "model_agreement: " << agreement
                      << " (" << model_agreements << '/' << examples.size() << ")\n"
                      << std::defaultfloat;

            std::cout << "disagreements_shown: " << disagreements.size() << '\n';
            for (const auto& disagreement : disagreements) {
                PrintDisagreement(disagreement);
            }
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
