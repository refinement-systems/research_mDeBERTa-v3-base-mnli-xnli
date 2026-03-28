#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace nli {

struct TokenizerSpecialTokenIds {
    int64_t pad;
    int64_t cls;
    int64_t sep;
    int64_t unk;
    int64_t mask;
};

struct TokenizerTemplateIds {
    int64_t cls = 0;
    int64_t first_sequence = 0;
    int64_t first_sep = 0;
    int64_t second_sequence = 1;
    int64_t second_sep = 1;
};

struct TokenizerAssetConfig {
    std::string pad_token = "[PAD]";
    std::string cls_token = "[CLS]";
    std::string sep_token = "[SEP]";
    std::string unk_token = "[UNK]";
    std::string mask_token = "[MASK]";
    TokenizerSpecialTokenIds special_token_ids = {0, 1, 2, 3, 3};
    TokenizerTemplateIds template_ids = {};
    bool do_lower_case = false;
    bool strip_left = true;
    bool strip_right = true;
    bool collapse_spaces = true;
    bool add_prefix_space = true;
    size_t max_length = 512;
};

TokenizerAssetConfig DefaultTokenizerAssetConfig();
TokenizerAssetConfig LoadTokenizerAssetConfigFromDir(const std::string& asset_dir);
TokenizerAssetConfig LoadTokenizerAssetConfigForSentencePiece(const std::string& sentencepiece_path);

}  // namespace nli
