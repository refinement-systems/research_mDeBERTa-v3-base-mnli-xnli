#pragma once

#include "session_factory.h"

#include <string>

namespace nli {

struct StudyInitCommandLineOptions {
    std::string scratchpad_root;
    std::string catalog_path;
    bool force;
};

struct StudyRunCommandLineOptions {
    std::string scratchpad_root;
    std::string quantization_name;
    SessionBackend backend;
    std::string dataset_name;
    bool force_regenerate;
    bool force_rerun;
};

}  // namespace nli
