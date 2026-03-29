#include "command_line.h"
#include "nli_eval.h"
#include "nli_inference.h"
#include "process_memory.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct ExampleTimingSummary {
    std::string benchmark;
    std::string id;
    double mean_ms = 0.0;
    double median_ms = 0.0;
    double p95_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    std::vector<double> runs_ms;
};

double PercentileMillis(const std::vector<double>& sorted_millis, double percentile) {
    if (sorted_millis.empty()) {
        return 0.0;
    }

    const double clamped = std::clamp(percentile, 0.0, 1.0);
    const size_t index = static_cast<size_t>(
        std::ceil(clamped * static_cast<double>(sorted_millis.size())) - 1.0);
    return sorted_millis[std::min(index, sorted_millis.size() - 1)];
}

ExampleTimingSummary SummarizeRuns(
    const std::string& benchmark,
    const std::string& id,
    const std::vector<double>& runs_ms) {
    if (runs_ms.empty()) {
        throw std::runtime_error("Cannot summarize empty timing run list");
    }

    std::vector<double> sorted_runs = runs_ms;
    std::sort(sorted_runs.begin(), sorted_runs.end());
    const double total_ms = std::accumulate(sorted_runs.begin(), sorted_runs.end(), 0.0);

    ExampleTimingSummary summary;
    summary.benchmark = benchmark;
    summary.id = id;
    summary.mean_ms = total_ms / static_cast<double>(sorted_runs.size());
    summary.median_ms = PercentileMillis(sorted_runs, 0.5);
    summary.p95_ms = PercentileMillis(sorted_runs, 0.95);
    summary.min_ms = sorted_runs.front();
    summary.max_ms = sorted_runs.back();
    summary.runs_ms = runs_ms;
    return summary;
}

void PrintSummaryLine(const std::string& key, double value) {
    std::cout << key << ": " << value << '\n';
}

void PrintSummaryLine(const std::string& key, uint64_t value) {
    std::cout << key << ": " << value << '\n';
}

void PrintExampleTimingLine(const std::string& prefix, const ExampleTimingSummary& summary) {
    std::cout << prefix
              << ": benchmark=" << summary.benchmark
              << '\t' << "id=" << summary.id
              << '\t' << "mean_ms=" << summary.mean_ms
              << '\t' << "median_ms=" << summary.median_ms
              << '\t' << "p95_ms=" << summary.p95_ms
              << '\t' << "min_ms=" << summary.min_ms
              << '\t' << "max_ms=" << summary.max_ms
              << '\n';
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        const nli::RuntimeBenchCommandLineOptions options =
            nli::ParseRuntimeBenchCommandLine(argc, argv);
        const auto examples = nli::ReadNliEvalExamples(options.input_path);
        if (examples.empty()) {
            throw std::runtime_error("Runtime benchmark input has no examples");
        }

        const auto load_start = std::chrono::steady_clock::now();
        nli::DebertaNliModel model(
            options.model_path,
            nli::DefaultSentencePieceModelPath(),
            options.backend,
            std::cerr);
        const auto load_end = std::chrono::steady_clock::now();
        const double load_ms = std::chrono::duration<double, std::milli>(load_end - load_start).count();
        const nli::ProcessMemorySnapshot memory_after_load = nli::GetProcessMemorySnapshot();

        std::vector<double> all_runs_ms;
        all_runs_ms.reserve(examples.size() * options.repeat_count);
        std::vector<ExampleTimingSummary> per_example_summaries;
        per_example_summaries.reserve(examples.size());
        nli::ProcessMemorySnapshot memory_after_initial_warmup = memory_after_load;
        bool captured_after_initial_warmup = false;

        for (const auto& example : examples) {
            for (size_t warmup_index = 0; warmup_index < options.warmup_count; ++warmup_index) {
                (void)model.PredictLogits(example.premise, example.hypothesis);
            }
            if (!captured_after_initial_warmup) {
                memory_after_initial_warmup = nli::GetProcessMemorySnapshot();
                captured_after_initial_warmup = true;
            }

            std::vector<double> runs_ms;
            runs_ms.reserve(options.repeat_count);
            for (size_t repeat_index = 0; repeat_index < options.repeat_count; ++repeat_index) {
                const auto inference_start = std::chrono::steady_clock::now();
                (void)model.PredictLogits(example.premise, example.hypothesis);
                const auto inference_end = std::chrono::steady_clock::now();
                const double run_ms = std::chrono::duration<double, std::milli>(
                                          inference_end - inference_start)
                                          .count();
                runs_ms.push_back(run_ms);
                all_runs_ms.push_back(run_ms);
            }

            per_example_summaries.push_back(SummarizeRuns(
                example.benchmark,
                example.id.empty() ? "<none>" : example.id,
                runs_ms));
        }
        const nli::ProcessMemorySnapshot memory_after_benchmark = nli::GetProcessMemorySnapshot();

        std::vector<double> sorted_all_runs = all_runs_ms;
        std::sort(sorted_all_runs.begin(), sorted_all_runs.end());
        const double total_ms = std::accumulate(sorted_all_runs.begin(), sorted_all_runs.end(), 0.0);

        std::cout << "examples: " << examples.size() << '\n';
        std::map<std::string, size_t> benchmark_counts;
        for (const auto& example : examples) {
            ++benchmark_counts[example.benchmark];
        }
        std::cout << "benchmarks:";
        bool first = true;
        for (const auto& [benchmark, _] : benchmark_counts) {
            std::cout << (first ? " " : ", ") << benchmark;
            first = false;
        }
        std::cout << '\n';

        PrintSummaryLine("load_ms", load_ms);
        std::cout << "warmup_runs_per_example: " << options.warmup_count << '\n';
        std::cout << "timed_runs_per_example: " << options.repeat_count << '\n';
        PrintSummaryLine("timing_total_ms", total_ms);
        PrintSummaryLine(
            "timing_mean_ms",
            total_ms / static_cast<double>(sorted_all_runs.size()));
        PrintSummaryLine("timing_median_ms", PercentileMillis(sorted_all_runs, 0.5));
        PrintSummaryLine("timing_p95_ms", PercentileMillis(sorted_all_runs, 0.95));
        PrintSummaryLine("timing_min_ms", sorted_all_runs.front());
        PrintSummaryLine("timing_max_ms", sorted_all_runs.back());
        if (memory_after_load.available) {
            PrintSummaryLine(
                "resident_after_load_bytes",
                memory_after_load.resident_bytes);
            PrintSummaryLine(
                "peak_rss_after_load_bytes",
                memory_after_load.peak_resident_bytes);
        }
        if (memory_after_initial_warmup.available) {
            PrintSummaryLine(
                "resident_after_warmup_bytes",
                memory_after_initial_warmup.resident_bytes);
            PrintSummaryLine(
                "peak_rss_after_warmup_bytes",
                memory_after_initial_warmup.peak_resident_bytes);
        }
        if (memory_after_benchmark.available) {
            PrintSummaryLine(
                "resident_after_timed_runs_bytes",
                memory_after_benchmark.resident_bytes);
            PrintSummaryLine(
                "peak_rss_after_timed_runs_bytes",
                memory_after_benchmark.peak_resident_bytes);
        }

        std::map<std::string, std::vector<double>> runs_by_benchmark;
        for (const auto& summary : per_example_summaries) {
            auto& runs = runs_by_benchmark[summary.benchmark];
            runs.insert(runs.end(), summary.runs_ms.begin(), summary.runs_ms.end());
        }
        for (const auto& [benchmark, runs] : runs_by_benchmark) {
            PrintExampleTimingLine(
                "benchmark_timing",
                SummarizeRuns(benchmark, "<aggregate>", runs));
        }

        if (options.dump_example_timings) {
            for (const auto& summary : per_example_summaries) {
                PrintExampleTimingLine("example_timing", summary);
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
