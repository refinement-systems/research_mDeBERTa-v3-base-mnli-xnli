#include "command_line.h"
#include "nli_inference.h"
#include "process_memory.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
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

double PercentileMillis(const std::vector<double>& sorted_millis, double percentile) {
    if (sorted_millis.empty()) {
        return 0.0;
    }

    const double clamped = std::clamp(percentile, 0.0, 1.0);
    const size_t index = static_cast<size_t>(
        std::ceil(clamped * static_cast<double>(sorted_millis.size())) - 1.0);
    return sorted_millis[std::min(index, sorted_millis.size() - 1)];
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        const nli::ExampleCommandLineOptions options = nli::ParseExampleCommandLine(argc, argv);

        const auto load_start = std::chrono::steady_clock::now();
        nli::DebertaNliModel model(
            options.model_path,
            nli::DefaultSentencePieceModelPath(),
            options.backend,
            std::cerr);
        const auto load_end = std::chrono::steady_clock::now();
        const double load_ms = std::chrono::duration<double, std::milli>(load_end - load_start).count();
        const nli::ProcessMemorySnapshot memory_after_load = nli::GetProcessMemorySnapshot();

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

        for (size_t warmup_index = 0; warmup_index < options.warmup_count; ++warmup_index) {
            (void)model.PredictLogits(options.premise, options.hypothesis);
        }
        const nli::ProcessMemorySnapshot memory_after_warmup = nli::GetProcessMemorySnapshot();

        std::vector<double> timing_runs_ms;
        timing_runs_ms.reserve(options.repeat_count);
        nli::NliLogits logits{};
        for (size_t repeat_index = 0; repeat_index < options.repeat_count; ++repeat_index) {
            const auto inference_start = std::chrono::steady_clock::now();
            logits = model.PredictLogits(options.premise, options.hypothesis);
            const auto inference_end = std::chrono::steady_clock::now();
            timing_runs_ms.push_back(
                std::chrono::duration<double, std::milli>(inference_end - inference_start).count());
        }
        const nli::ProcessMemorySnapshot memory_after_timed_runs = nli::GetProcessMemorySnapshot();

        const nli::NliScores scores = nli::ScoresFromLogits(logits);

        if (options.dump_logits) {
            std::cout << "logits:"
                      << " entailment=" << logits.entailment
                      << " neutral=" << logits.neutral
                      << " contradiction=" << logits.contradiction
                      << "\n";
            std::cout << "predicted_logit_label: " << nli::PredictedLabel(logits) << "\n";
        }

        if (!options.quiet) {
            std::cout << "entailment: " << scores.entailment << "\n";
            std::cout << "neutral: " << scores.neutral << "\n";
            std::cout << "contradiction: " << scores.contradiction << "\n";
            std::cout << "predicted_label: " << nli::PredictedLabel(scores) << "\n";
        }

        if (options.timing) {
            const double total_ms = std::accumulate(
                timing_runs_ms.begin(),
                timing_runs_ms.end(),
                0.0);
            std::vector<double> sorted_runs_ms = timing_runs_ms;
            std::sort(sorted_runs_ms.begin(), sorted_runs_ms.end());
            const double mean_ms = total_ms / static_cast<double>(timing_runs_ms.size());
            const double median_ms = PercentileMillis(sorted_runs_ms, 0.5);
            const double p95_ms = PercentileMillis(sorted_runs_ms, 0.95);
            const double min_ms = sorted_runs_ms.front();
            const double max_ms = sorted_runs_ms.back();

            std::cout << "load_ms: " << load_ms << "\n";
            std::cout << "warmup_runs: " << options.warmup_count << "\n";
            std::cout << "timed_runs: " << options.repeat_count << "\n";
            std::cout << "timing_total_ms: " << total_ms << "\n";
            std::cout << "timing_mean_ms: " << mean_ms << "\n";
            std::cout << "timing_median_ms: " << median_ms << "\n";
            std::cout << "timing_p95_ms: " << p95_ms << "\n";
            std::cout << "timing_min_ms: " << min_ms << "\n";
            std::cout << "timing_max_ms: " << max_ms << "\n";
            if (memory_after_load.available) {
                std::cout << "resident_after_load_bytes: " << memory_after_load.resident_bytes << "\n";
                std::cout << "peak_rss_after_load_bytes: " << memory_after_load.peak_resident_bytes << "\n";
            }
            if (memory_after_warmup.available) {
                std::cout << "resident_after_warmup_bytes: " << memory_after_warmup.resident_bytes << "\n";
                std::cout << "peak_rss_after_warmup_bytes: " << memory_after_warmup.peak_resident_bytes << "\n";
            }
            if (memory_after_timed_runs.available) {
                std::cout << "resident_after_timed_runs_bytes: " << memory_after_timed_runs.resident_bytes << "\n";
                std::cout << "peak_rss_after_timed_runs_bytes: " << memory_after_timed_runs.peak_resident_bytes << "\n";
            }

            if (options.dump_timing_runs) {
                std::cout << "timing_runs_ms:";
                for (const double run_ms : timing_runs_ms) {
                    std::cout << ' ' << run_ms;
                }
                std::cout << "\n";
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
