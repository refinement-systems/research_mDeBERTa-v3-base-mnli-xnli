#pragma once

#include "optparse.h"
#include "session_factory.h"

#include <cstddef>
#include <string>

namespace nli {

struct ExampleCommandLineOptions {
    SessionBackend backend;
    std::string model_path;
    std::string premise;
    std::string hypothesis;
    size_t repeat_count;
    size_t warmup_count;
    bool dump_encoding;
    bool dump_logits;
    bool dump_special_token_ids;
    bool timing;
    bool dump_timing_runs;
    bool quiet;
};

struct TopicalChatCommandLineOptions {
    SessionBackend backend;
    std::string model_path;
    std::string input_path;
};

struct EvalCommandLineOptions {
    SessionBackend backend;
    std::string model_path;
    std::string compare_model_path;
    size_t max_disagreements;
    std::string input_path;
};

struct RuntimeBenchCommandLineOptions {
    SessionBackend backend;
    std::string model_path;
    size_t repeat_count;
    size_t warmup_count;
    bool dump_example_timings;
    std::string input_path;
};

void ConfigureExampleOptionParser(optparse::OptionParser& parser);
optparse::OptionParser BuildExampleOptionParser();
ExampleCommandLineOptions FinalizeExampleCommandLine(
    const optparse::OptionParser& parser,
    const optparse::Values& options);
ExampleCommandLineOptions ParseExampleCommandLine(int argc, char* argv[]);

void ConfigureTopicalChatOptionParser(optparse::OptionParser& parser);
optparse::OptionParser BuildTopicalChatOptionParser();
TopicalChatCommandLineOptions FinalizeTopicalChatCommandLine(
    const optparse::OptionParser& parser,
    const optparse::Values& options);
TopicalChatCommandLineOptions ParseTopicalChatCommandLine(int argc, char* argv[]);

void ConfigureEvalOptionParser(optparse::OptionParser& parser);
optparse::OptionParser BuildEvalOptionParser();
EvalCommandLineOptions FinalizeEvalCommandLine(
    const optparse::OptionParser& parser,
    const optparse::Values& options);
EvalCommandLineOptions ParseEvalCommandLine(int argc, char* argv[]);

void ConfigureRuntimeBenchOptionParser(optparse::OptionParser& parser);
optparse::OptionParser BuildRuntimeBenchOptionParser();
RuntimeBenchCommandLineOptions FinalizeRuntimeBenchCommandLine(
    const optparse::OptionParser& parser,
    const optparse::Values& options);
RuntimeBenchCommandLineOptions ParseRuntimeBenchCommandLine(int argc, char* argv[]);

}  // namespace nli
