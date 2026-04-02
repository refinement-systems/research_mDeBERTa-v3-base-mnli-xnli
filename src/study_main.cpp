#include "study_command_line.h"
#include "study_workflow.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void PrintUsage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " <init|run> [options]\n";
}

std::ofstream OpenEvaluationLog(const nli::StudyRunCommandLineOptions& options) {
    const std::filesystem::path scratchpad_root =
        std::filesystem::absolute(std::filesystem::path(options.scratchpad_root)).lexically_normal();
    const std::filesystem::path log_path =
        scratchpad_root / "logs" / "evaluation" /
        (options.quantization_name + "-" +
         std::string(nli::SessionBackendOptionName(options.backend)) + "-" +
         options.dataset_name + ".log");
    std::filesystem::create_directories(log_path.parent_path());
    std::ofstream log_stream(log_path, std::ios::trunc);
    if (!log_stream) {
        throw std::runtime_error("Failed to open evaluation log: " + log_path.string());
    }
    return log_stream;
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        if (argc < 2) {
            PrintUsage(argv[0]);
            return 2;
        }

        const std::string subcommand = argv[1];
        if (subcommand == "init") {
            auto parser = nli::BuildStudyInitOptionParser();
            parser.prog(std::string(argv[0]) + " init");
            const optparse::Values& values = parser.parse_args(argc - 1, argv + 1);
            const auto options = nli::FinalizeStudyInitCommandLine(parser, values);
            nli::InitializeStudyWorkspace(options, std::cout);
            return 0;
        }

        if (subcommand == "run") {
            auto parser = nli::BuildStudyRunOptionParser();
            parser.prog(std::string(argv[0]) + " run");
            const optparse::Values& values = parser.parse_args(argc - 1, argv + 1);
            const auto options = nli::FinalizeStudyRunCommandLine(parser, values);
            std::ofstream log_stream = OpenEvaluationLog(options);
            nli::RunStudyEvaluation(options, log_stream);
            return 0;
        }

        PrintUsage(argv[0]);
        return 2;
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
