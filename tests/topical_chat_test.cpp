#include "command_line.h"
#include "nli_eval.h"
#include "nli_inference.h"
#include "tokenizer_utils.h"
#include "topical_chat.h"

#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

std::string FixturePath() {
    return std::string(NLI_SOURCE_DIR) + "/tests/data/topical_chat_fixture.json";
}

std::string EvalFixturePath() {
    return std::string(NLI_SOURCE_DIR) + "/tests/data/nli_eval_fixture.tsv";
}

void ExpectParserExitCode(const std::function<void()>& callback, int expected_code) {
    try {
        callback();
    } catch (int code) {
        if (code == expected_code) {
            return;
        }
        throw std::runtime_error(
            "expected parser exit code " + std::to_string(expected_code) +
            ", got " + std::to_string(code));
    }

    throw std::runtime_error(
        "expected parser exit code " + std::to_string(expected_code));
}

optparse::OptionParserExcept MakeTopicalChatParser() {
    optparse::OptionParserExcept parser;
    nli::ConfigureTopicalChatOptionParser(parser);
    parser.prog("nli-topical-chat");
    return parser;
}

optparse::OptionParserExcept MakeExampleParser() {
    optparse::OptionParserExcept parser;
    nli::ConfigureExampleOptionParser(parser);
    parser.prog("nli");
    return parser;
}

optparse::OptionParserExcept MakeEvalParser() {
    optparse::OptionParserExcept parser;
    nli::ConfigureEvalOptionParser(parser);
    parser.prog("nli-eval");
    return parser;
}

void VerifyTopicalChatParsingPreservesOrder() {
    const auto turns = nli::ReadTopicalChatTurnInputs(FixturePath());
    const std::vector<std::pair<std::string, std::string>> expected = {
        {"Are you following the launch?", "Curious to dive deeper"},
        {"I read the mission update this morning.", "Happy"},
        {"That sounds risky.", "Surprised"},
    };

    if (turns.size() != expected.size()) {
        throw std::runtime_error("unexpected number of parsed turns");
    }

    for (size_t i = 0; i < expected.size(); ++i) {
        if (turns[i].premise != expected[i].first) {
            throw std::runtime_error("parsed premise order mismatch");
        }
        if (turns[i].hypothesis != expected[i].second) {
            throw std::runtime_error("parsed hypothesis order mismatch");
        }
    }
}

void VerifyTopicalChatOptionsAcceptSingleInput() {
    auto parser = MakeTopicalChatParser();
    const std::vector<std::string> args = {"--backend=cpu", FixturePath()};
    const optparse::Values& values = parser.parse_args(args);
    const auto options = nli::FinalizeTopicalChatCommandLine(parser, values);

    if (options.backend != nli::SessionBackend::kCPU) {
        throw std::runtime_error("expected explicit cpu backend to parse");
    }
    if (options.model_path != nli::DefaultModelPath()) {
        throw std::runtime_error("expected default model path to be used");
    }
    if (options.input_path != FixturePath()) {
        throw std::runtime_error("expected input path to round-trip");
    }
}

void VerifyTopicalChatOptionsAcceptExplicitModelPath() {
    auto parser = MakeTopicalChatParser();
    const std::vector<std::string> args = {
        "--backend=cpu",
        "--model=/tmp/custom.onnx",
        FixturePath(),
    };
    const optparse::Values& values = parser.parse_args(args);
    const auto options = nli::FinalizeTopicalChatCommandLine(parser, values);

    if (options.backend != nli::SessionBackend::kCPU) {
        throw std::runtime_error("expected explicit cpu backend to parse");
    }
    if (options.model_path != "/tmp/custom.onnx") {
        throw std::runtime_error("expected explicit model path to round-trip");
    }
    if (options.input_path != FixturePath()) {
        throw std::runtime_error("expected input path to round-trip");
    }
}

void VerifyTopicalChatOptionsDefaultBackend() {
    auto parser = MakeTopicalChatParser();
    const std::vector<std::string> args = {FixturePath()};
    const optparse::Values& values = parser.parse_args(args);
    const auto options = nli::FinalizeTopicalChatCommandLine(parser, values);

    if (options.backend != nli::DefaultSessionBackend()) {
        throw std::runtime_error("expected default backend to match nli");
    }
    if (options.model_path != nli::DefaultModelPath()) {
        throw std::runtime_error("expected default model path to match nli");
    }
}

void VerifyTopicalChatOptionsRejectMissingInput() {
    auto parser = MakeTopicalChatParser();
    const std::vector<std::string> args;
    const optparse::Values& values = parser.parse_args(args);

    ExpectParserExitCode(
        [&]() {
            (void)nli::FinalizeTopicalChatCommandLine(parser, values);
        },
        2);
}

void VerifyTopicalChatOptionsRejectExtraInput() {
    auto parser = MakeTopicalChatParser();
    const std::vector<std::string> args = {FixturePath(), "extra.json"};
    const optparse::Values& values = parser.parse_args(args);

    ExpectParserExitCode(
        [&]() {
            (void)nli::FinalizeTopicalChatCommandLine(parser, values);
        },
        2);
}

void VerifyExampleOptionsAcceptExplicitModelPath() {
    auto parser = MakeExampleParser();
    const std::vector<std::string> args = {"--backend=cpu", "--model=/tmp/custom.onnx"};
    const optparse::Values& values = parser.parse_args(args);
    const auto options = nli::FinalizeExampleCommandLine(parser, values);

    if (options.backend != nli::SessionBackend::kCPU) {
        throw std::runtime_error("expected explicit cpu backend to parse");
    }
    if (options.model_path != "/tmp/custom.onnx") {
        throw std::runtime_error("expected explicit model path to round-trip");
    }
    if (options.dump_encoding) {
        throw std::runtime_error("expected dump encoding to default to false");
    }
}

void VerifyExampleOptionsDefaultModelPath() {
    auto parser = MakeExampleParser();
    const std::vector<std::string> args;
    const optparse::Values& values = parser.parse_args(args);
    const auto options = nli::FinalizeExampleCommandLine(parser, values);

    if (options.backend != nli::DefaultSessionBackend()) {
        throw std::runtime_error("expected default backend to match nli");
    }
    if (options.model_path != nli::DefaultModelPath()) {
        throw std::runtime_error("expected default model path to match nli");
    }
    if (options.premise.empty() || options.hypothesis.empty()) {
        throw std::runtime_error("expected default example texts to be populated");
    }
}

void VerifyExampleOptionsAcceptCustomTextsAndEncodingFlag() {
    auto parser = MakeExampleParser();
    const std::vector<std::string> args = {
        "--premise=Premise text",
        "--hypothesis=Hypothesis text",
        "--dump-encoding",
    };
    const optparse::Values& values = parser.parse_args(args);
    const auto options = nli::FinalizeExampleCommandLine(parser, values);

    if (options.premise != "Premise text") {
        throw std::runtime_error("expected custom premise to round-trip");
    }
    if (options.hypothesis != "Hypothesis text") {
        throw std::runtime_error("expected custom hypothesis to round-trip");
    }
    if (!options.dump_encoding) {
        throw std::runtime_error("expected dump encoding flag to be enabled");
    }
}

void VerifyExampleOptionsRejectUnexpectedPositionalArgs() {
    auto parser = MakeExampleParser();
    const std::vector<std::string> args = {"unexpected"};
    const optparse::Values& values = parser.parse_args(args);

    ExpectParserExitCode(
        [&]() {
            (void)nli::FinalizeExampleCommandLine(parser, values);
        },
        2);
}

void VerifyTokenizerNormalizationMatchesExpectedWhitespaceHandling() {
    const std::string input = "  Hello\tworld \n from\r\nCodex  ";
    const std::string expected = " Hello world from Codex";
    const std::string actual = nli::NormalizeDebertaTokenizerInput(input);

    if (actual != expected) {
        throw std::runtime_error("unexpected tokenizer normalization result");
    }
}

void VerifyEvalFixtureParsingPreservesRows() {
    const auto examples = nli::ReadNliEvalExamples(EvalFixturePath());
    if (examples.size() != 3) {
        throw std::runtime_error("unexpected number of eval examples");
    }

    if (examples[0].id != "row-1") {
        throw std::runtime_error("expected eval id to round-trip");
    }
    if (!examples[1].label || *examples[1].label != "contradiction") {
        throw std::runtime_error("expected eval label to round-trip");
    }
    if (examples[2].premise != "Angela Merkel is a politician.") {
        throw std::runtime_error("expected eval premise to round-trip");
    }
    if (examples[2].hypothesis != "The economy is growing.") {
        throw std::runtime_error("expected eval hypothesis to round-trip");
    }
}

void VerifyEvalOptionsAcceptComparisonModel() {
    auto parser = MakeEvalParser();
    const std::vector<std::string> args = {
        "--backend=cpu",
        "--model=/tmp/float.onnx",
        "--compare-model=/tmp/quant.onnx",
        "--max-disagreements=7",
        EvalFixturePath(),
    };
    const optparse::Values& values = parser.parse_args(args);
    const auto options = nli::FinalizeEvalCommandLine(parser, values);

    if (options.backend != nli::SessionBackend::kCPU) {
        throw std::runtime_error("expected eval backend to parse");
    }
    if (options.model_path != "/tmp/float.onnx") {
        throw std::runtime_error("expected eval model path to round-trip");
    }
    if (options.compare_model_path != "/tmp/quant.onnx") {
        throw std::runtime_error("expected compare model path to round-trip");
    }
    if (options.max_disagreements != 7) {
        throw std::runtime_error("expected max disagreements to round-trip");
    }
    if (options.input_path != EvalFixturePath()) {
        throw std::runtime_error("expected eval input path to round-trip");
    }
}

void VerifyEvalOptionsUseDefaults() {
    auto parser = MakeEvalParser();
    const std::vector<std::string> args = {EvalFixturePath()};
    const optparse::Values& values = parser.parse_args(args);
    const auto options = nli::FinalizeEvalCommandLine(parser, values);

    if (options.backend != nli::DefaultSessionBackend()) {
        throw std::runtime_error("expected eval default backend to match nli");
    }
    if (options.model_path != nli::DefaultModelPath()) {
        throw std::runtime_error("expected eval default model path to match nli");
    }
    if (!options.compare_model_path.empty()) {
        throw std::runtime_error("expected compare model path to default to empty");
    }
    if (options.max_disagreements != 10) {
        throw std::runtime_error("expected max disagreements default to be 10");
    }
}

void VerifyEvalOptionsRejectMissingInput() {
    auto parser = MakeEvalParser();
    const std::vector<std::string> args;
    const optparse::Values& values = parser.parse_args(args);

    ExpectParserExitCode(
        [&]() {
            (void)nli::FinalizeEvalCommandLine(parser, values);
        },
        2);
}

}  // namespace

int main() {
    VerifyTopicalChatParsingPreservesOrder();
    VerifyTopicalChatOptionsAcceptSingleInput();
    VerifyTopicalChatOptionsAcceptExplicitModelPath();
    VerifyTopicalChatOptionsDefaultBackend();
    VerifyTopicalChatOptionsRejectMissingInput();
    VerifyTopicalChatOptionsRejectExtraInput();
    VerifyExampleOptionsAcceptExplicitModelPath();
    VerifyExampleOptionsDefaultModelPath();
    VerifyExampleOptionsAcceptCustomTextsAndEncodingFlag();
    VerifyExampleOptionsRejectUnexpectedPositionalArgs();
    VerifyTokenizerNormalizationMatchesExpectedWhitespaceHandling();
    VerifyEvalFixtureParsingPreservesRows();
    VerifyEvalOptionsAcceptComparisonModel();
    VerifyEvalOptionsUseDefaults();
    VerifyEvalOptionsRejectMissingInput();
    return 0;
}
