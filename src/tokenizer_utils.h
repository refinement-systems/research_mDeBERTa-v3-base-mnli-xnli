#pragma once

#include "tokenizer_assets.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace nli {

struct TokenizerTextSegment {
    std::string text;
    bool is_special_token = false;
    int64_t token_id = -1;
};

std::string NormalizeDebertaTokenizerInput(
    std::string_view text,
    const TokenizerAssetConfig& config);
std::vector<TokenizerTextSegment> SplitDebertaTokenizerInput(
    std::string_view normalized_text,
    const TokenizerAssetConfig& config);

}  // namespace nli
