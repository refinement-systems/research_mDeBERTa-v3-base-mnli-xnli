#pragma once

#include <string>
#include <vector>

namespace nli {

struct TopicalChatTurnInput {
    std::string premise;
    std::string hypothesis;
};

std::vector<TopicalChatTurnInput> ReadTopicalChatTurnInputs(const std::string& path);

}  // namespace nli
