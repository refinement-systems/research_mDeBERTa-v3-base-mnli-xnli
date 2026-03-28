#include "command_line.h"
#include "nli_inference.h"
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
    VerifyExampleOptionsRejectUnexpectedPositionalArgs();
    return 0;
}
