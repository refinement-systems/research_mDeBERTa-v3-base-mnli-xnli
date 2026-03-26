#include "command_line.h"

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

void AddBackendOption(optparse::OptionParser& parser) {
    const std::vector<std::string> available_backends = nli::AvailableSessionBackendOptionNames();

    parser.add_option("-b", "--backend")
        .dest("backend")
        .metavar("BACKEND")
        .choices(available_backends.begin(), available_backends.end())
        .set_default(nli::SessionBackendOptionName(nli::DefaultSessionBackend()))
        .help(
            "preferred execution backend (available: " +
            JoinStrings(available_backends, ", ") +
            "; default: %default)");
}

}  // namespace

namespace nli {

void ConfigureExampleOptionParser(optparse::OptionParser& parser) {
    parser.usage("%prog [options]");
    parser.description("Run the NLI example.");
    AddBackendOption(parser);
}

optparse::OptionParser BuildExampleOptionParser() {
    optparse::OptionParser parser;
    ConfigureExampleOptionParser(parser);
    return parser;
}

ExampleCommandLineOptions FinalizeExampleCommandLine(
    const optparse::OptionParser& parser,
    const optparse::Values& options) {
    if (!parser.args().empty()) {
        parser.error("unexpected positional arguments");
    }

    return ExampleCommandLineOptions{
        ParseSessionBackendOption(options["backend"]),
    };
}

ExampleCommandLineOptions ParseExampleCommandLine(int argc, char* argv[]) {
    auto parser = BuildExampleOptionParser();
    const optparse::Values& options = parser.parse_args(argc, argv);
    return FinalizeExampleCommandLine(parser, options);
}

void ConfigureTopicalChatOptionParser(optparse::OptionParser& parser) {
    parser.usage("%prog [options] INPUT_JSON");
    parser.description("Run the DeBERTa NLI model over every Topical Chat dialog turn.");
    AddBackendOption(parser);
}

optparse::OptionParser BuildTopicalChatOptionParser() {
    optparse::OptionParser parser;
    ConfigureTopicalChatOptionParser(parser);
    return parser;
}

TopicalChatCommandLineOptions FinalizeTopicalChatCommandLine(
    const optparse::OptionParser& parser,
    const optparse::Values& options) {
    if (parser.args().size() != 1) {
        parser.error("expected exactly one input file argument");
    }

    return TopicalChatCommandLineOptions{
        ParseSessionBackendOption(options["backend"]),
        parser.args().front(),
    };
}

TopicalChatCommandLineOptions ParseTopicalChatCommandLine(int argc, char* argv[]) {
    auto parser = BuildTopicalChatOptionParser();
    const optparse::Values& options = parser.parse_args(argc, argv);
    return FinalizeTopicalChatCommandLine(parser, options);
}

}  // namespace nli
