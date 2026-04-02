#include "study_command_line.h"

#include "session_factory.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string JoinStrings(const std::vector<std::string>& values, const std::string& separator) {
    std::string joined;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            joined += separator;
        }
        joined += values[i];
    }
    return joined;
}

void AddScratchpadRootOption(optparse::OptionParser& parser) {
    parser.add_option("--scratchpad-root")
        .dest("scratchpad_root")
        .metavar("PATH")
        .set_default(nli::DefaultStudyScratchpadRoot())
        .help("scratchpad root directory (default: %default)");
}

void AddCatalogOption(optparse::OptionParser& parser) {
    parser.add_option("--catalog")
        .dest("catalog_path")
        .metavar("PATH")
        .set_default(nli::DefaultStudyCatalogPath())
        .help("quantization catalog JSON path (default: %default)");
}

void AddBackendOption(optparse::OptionParser& parser) {
    const std::vector<std::string> available_backends = nli::AvailableSessionBackendOptionNames();
    parser.add_option("--backend")
        .dest("backend")
        .metavar("BACKEND")
        .choices(available_backends.begin(), available_backends.end())
        .help(
            "execution backend (available: " +
            JoinStrings(available_backends, ", ") +
            ")");
}

}  // namespace

namespace nli {

std::string DefaultStudyScratchpadRoot() {
#if defined(NLI_SOURCE_DIR)
    return std::string(NLI_SOURCE_DIR) + "/scratchpad";
#else
    return "scratchpad";
#endif
}

std::string DefaultStudyCatalogPath() {
#if defined(NLI_SOURCE_DIR)
    return std::string(NLI_SOURCE_DIR) +
           "/research/attempt2_course-correction/study_quantization_catalog.json";
#else
    return "research/attempt2_course-correction/study_quantization_catalog.json";
#endif
}

void ConfigureStudyInitOptionParser(optparse::OptionParser& parser) {
    parser.usage("%prog [options]");
    parser.description("Initialize the scratchpad-backed study database and catalog.");
    AddScratchpadRootOption(parser);
    AddCatalogOption(parser);
    parser.add_option("--force")
        .dest("force")
        .action("store_true")
        .set_default(false)
        .help("recreate the study database from scratch");
}

optparse::OptionParser BuildStudyInitOptionParser() {
    optparse::OptionParser parser;
    ConfigureStudyInitOptionParser(parser);
    return parser;
}

StudyInitCommandLineOptions FinalizeStudyInitCommandLine(
    const optparse::OptionParser& parser,
    const optparse::Values& options) {
    if (!parser.args().empty()) {
        parser.error("unexpected positional arguments");
    }

    return StudyInitCommandLineOptions{
        options["scratchpad_root"],
        options["catalog_path"],
        options.is_set_by_user("force"),
    };
}

void ConfigureStudyRunOptionParser(optparse::OptionParser& parser) {
    parser.usage("%prog [options]");
    parser.description("Run one quantization/backend/dataset evaluation through the study database.");
    AddScratchpadRootOption(parser);
    AddBackendOption(parser);
    parser.add_option("--quantization")
        .dest("quantization_name")
        .metavar("NAME")
        .help("quantization name from the catalog");
    parser.add_option("--dataset")
        .dest("dataset_name")
        .metavar("NAME")
        .help("dataset name imported into the study database");
    parser.add_option("--force-regenerate")
        .dest("force_regenerate")
        .action("store_true")
        .set_default(false)
        .help("recreate the artifact even if a valid file already exists");
    parser.add_option("--force-rerun")
        .dest("force_rerun")
        .action("store_true")
        .set_default(false)
        .help("clear existing evaluation rows for this run before re-evaluating");
}

optparse::OptionParser BuildStudyRunOptionParser() {
    optparse::OptionParser parser;
    ConfigureStudyRunOptionParser(parser);
    return parser;
}

StudyRunCommandLineOptions FinalizeStudyRunCommandLine(
    const optparse::OptionParser& parser,
    const optparse::Values& options) {
    if (!parser.args().empty()) {
        parser.error("unexpected positional arguments");
    }
    if (options["backend"].empty()) {
        parser.error("--backend is required");
    }
    if (options["quantization_name"].empty()) {
        parser.error("--quantization is required");
    }
    if (options["dataset_name"].empty()) {
        parser.error("--dataset is required");
    }

    return StudyRunCommandLineOptions{
        options["scratchpad_root"],
        options["quantization_name"],
        ParseSessionBackendOption(options["backend"]),
        options["dataset_name"],
        options.is_set_by_user("force_regenerate"),
        options.is_set_by_user("force_rerun"),
    };
}

}  // namespace nli
