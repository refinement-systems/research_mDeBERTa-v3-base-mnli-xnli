#pragma once

#include <optional>
#include <string>
#include <vector>

namespace nli {

struct NliEvalExample {
    std::string id;
    std::optional<std::string> label;
    std::string premise;
    std::string hypothesis;
};

std::vector<NliEvalExample> ReadNliEvalExamples(const std::string& path);

}  // namespace nli
