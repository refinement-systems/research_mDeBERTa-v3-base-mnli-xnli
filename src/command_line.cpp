#include "command_line.h"
#include "nli_inference.h"

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

void AddModelOption(optparse::OptionParser& parser) {
    parser.add_option("-m", "--model")
        .dest("model")
        .metavar("PATH")
        .set_default(nli::DefaultModelPath())
        .help("path to ONNX model file (default: %default)");
}

void AddPremiseOption(optparse::OptionParser& parser) {
    parser.add_option("--premise")
        .dest("premise")
        .metavar("TEXT")
        .set_default(
            "Angela Merkel ist eine Politikerin in Deutschland und Vorsitzende der CDU")
        .help("premise text to classify (default: %default)");
}

void AddHypothesisOption(optparse::OptionParser& parser) {
    parser.add_option("--hypothesis")
        .dest("hypothesis")
        .metavar("TEXT")
        .set_default("Emmanuel Macron is the President of France")
        .help("hypothesis text to classify (default: %default)");
}

void AddDumpEncodingOption(optparse::OptionParser& parser) {
    parser.add_option("--dump-encoding")
        .dest("dump_encoding")
        .action("store_true")
        .set_default(false)
        .help("print normalized text and encoded model inputs before inference");
}

void AddDumpLogitsOption(optparse::OptionParser& parser) {
    parser.add_option("--dump-logits")
        .dest("dump_logits")
        .action("store_true")
        .set_default(false)
        .help("print raw model logits before softmax");
}

void AddDumpSpecialTokenIdsOption(optparse::OptionParser& parser) {
    parser.add_option("--dump-special-token-ids")
        .dest("dump_special_token_ids")
        .action("store_true")
        .set_default(false)
        .help("print SentencePiece ids for the known special tokens");
}

void AddCompareModelOption(optparse::OptionParser& parser) {
    parser.add_option("--compare-model")
        .dest("compare_model")
        .metavar("PATH")
        .set_default("")
        .help("optional second ONNX model path for side-by-side comparison");
}

void AddMaxDisagreementsOption(optparse::OptionParser& parser) {
    parser.add_option("--max-disagreements")
        .dest("max_disagreements")
        .metavar("N")
        .type("int")
        .set_default(10)
        .help("maximum number of prediction disagreements to print (default: %default)");
}

size_t ParseNonNegativeSize(const std::string& value, const std::string& option_name) {
    long parsed = 0;
    try {
        parsed = std::stol(value);
    } catch (const std::exception&) {
        throw std::invalid_argument("Invalid value for " + option_name + ": " + value);
    }

    if (parsed < 0) {
        throw std::invalid_argument(option_name + " must be non-negative");
    }

    return static_cast<size_t>(parsed);
}

}  // namespace

namespace nli {

void ConfigureExampleOptionParser(optparse::OptionParser& parser) {
    parser.usage("%prog [options]");
    parser.description("Run the NLI example.");
    AddBackendOption(parser);
    AddModelOption(parser);
    AddPremiseOption(parser);
    AddHypothesisOption(parser);
    AddDumpEncodingOption(parser);
    AddDumpLogitsOption(parser);
    AddDumpSpecialTokenIdsOption(parser);
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
        options["model"],
        options["premise"],
        options["hypothesis"],
        options.is_set_by_user("dump_encoding"),
        options.is_set_by_user("dump_logits"),
        options.is_set_by_user("dump_special_token_ids"),
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
    AddModelOption(parser);
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
        options["model"],
        parser.args().front(),
    };
}

TopicalChatCommandLineOptions ParseTopicalChatCommandLine(int argc, char* argv[]) {
    auto parser = BuildTopicalChatOptionParser();
    const optparse::Values& options = parser.parse_args(argc, argv);
    return FinalizeTopicalChatCommandLine(parser, options);
}

void ConfigureEvalOptionParser(optparse::OptionParser& parser) {
    parser.usage("%prog [options] INPUT_TSV");
    parser.description(
        "Evaluate one or two DeBERTa NLI ONNX models on a TSV file containing premise, "
        "hypothesis, and optional label columns.");
    AddBackendOption(parser);
    AddModelOption(parser);
    AddCompareModelOption(parser);
    AddMaxDisagreementsOption(parser);
}

optparse::OptionParser BuildEvalOptionParser() {
    optparse::OptionParser parser;
    ConfigureEvalOptionParser(parser);
    return parser;
}

EvalCommandLineOptions FinalizeEvalCommandLine(
    const optparse::OptionParser& parser,
    const optparse::Values& options) {
    if (parser.args().size() != 1) {
        parser.error("expected exactly one TSV input file argument");
    }

    return EvalCommandLineOptions{
        ParseSessionBackendOption(options["backend"]),
        options["model"],
        options["compare_model"],
        ParseNonNegativeSize(options["max_disagreements"], "--max-disagreements"),
        parser.args().front(),
    };
}

EvalCommandLineOptions ParseEvalCommandLine(int argc, char* argv[]) {
    auto parser = BuildEvalOptionParser();
    const optparse::Values& options = parser.parse_args(argc, argv);
    return FinalizeEvalCommandLine(parser, options);
}

}  // namespace nli
