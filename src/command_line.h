#pragma once

#include "optparse.h"
#include "session_factory.h"

#include <string>

namespace nli {

struct ExampleCommandLineOptions {
    SessionBackend backend;
};

struct TopicalChatCommandLineOptions {
    SessionBackend backend;
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

}  // namespace nli
