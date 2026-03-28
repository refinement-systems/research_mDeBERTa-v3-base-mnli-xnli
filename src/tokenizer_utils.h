#pragma once

#include <string>
#include <string_view>

namespace nli {

std::string NormalizeDebertaTokenizerInput(std::string_view text);

}  // namespace nli
