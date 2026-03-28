#include "tokenizer_utils.h"

#include <algorithm>
#include <string>
#include <string_view>
#include <vector>

namespace {

bool IsAsciiWhitespace(const char ch) {
    switch (ch) {
        case ' ':
        case '\f':
        case '\n':
        case '\r':
        case '\t':
        case '\v':
            return true;
        default:
            return false;
    }
}

struct SpecialTokenCandidate {
    std::string_view content;
    int64_t id;
};

std::vector<SpecialTokenCandidate> SpecialTokenCandidates(
    const nli::TokenizerAssetConfig& config) {
    std::vector<SpecialTokenCandidate> tokens;
    for (const auto& token :
         std::initializer_list<SpecialTokenCandidate>{
             {config.pad_token, config.special_token_ids.pad},
             {config.cls_token, config.special_token_ids.cls},
             {config.sep_token, config.special_token_ids.sep},
             {config.unk_token, config.special_token_ids.unk},
             {config.mask_token, config.special_token_ids.mask},
         }) {
        if (!token.content.empty()) {
            tokens.push_back(token);
        }
    }

    std::sort(
        tokens.begin(),
        tokens.end(),
        [](const SpecialTokenCandidate& left, const SpecialTokenCandidate& right) {
            return left.content.size() > right.content.size();
        });
    return tokens;
}

}  // namespace

namespace nli {

std::string NormalizeDebertaTokenizerInput(
    std::string_view text,
    const TokenizerAssetConfig& config) {
    // Approximate the Hugging Face tokenizer.json behavior for this checkpoint:
    // strip surrounding whitespace, then collapse internal ASCII whitespace runs
    // to a single space before SentencePiece applies its own model-specific
    // normalization.
    std::string normalized;
    normalized.reserve(text.size());

    bool previous_was_space = false;
    bool seen_non_space = false;
    for (const char ch : text) {
        if (IsAsciiWhitespace(ch)) {
            if ((!config.strip_left || seen_non_space) &&
                config.collapse_spaces &&
                !previous_was_space) {
                normalized.push_back(' ');
                previous_was_space = true;
            }
            continue;
        }

        if (config.do_lower_case && ch >= 'A' && ch <= 'Z') {
            normalized.push_back(static_cast<char>(ch - 'A' + 'a'));
        } else {
            normalized.push_back(ch);
        }
        previous_was_space = false;
        seen_non_space = true;
    }

    while (config.strip_right && !normalized.empty() && normalized.back() == ' ') {
        normalized.pop_back();
    }

    return normalized;
}

std::vector<TokenizerTextSegment> SplitDebertaTokenizerInput(
    std::string_view normalized_text,
    const TokenizerAssetConfig& config) {
    std::vector<TokenizerTextSegment> segments;
    const auto special_tokens = SpecialTokenCandidates(config);

    size_t segment_start = 0;
    size_t offset = 0;
    while (offset < normalized_text.size()) {
        bool matched_special_token = false;
        for (const auto& token : special_tokens) {
            if (token.content.empty() ||
                normalized_text.substr(offset, token.content.size()) != token.content) {
                continue;
            }

            if (offset > segment_start) {
                segments.push_back(TokenizerTextSegment{
                    std::string(normalized_text.substr(segment_start, offset - segment_start)),
                    false,
                    -1,
                });
            }
            segments.push_back(TokenizerTextSegment{
                std::string(token.content),
                true,
                token.id,
            });
            offset += token.content.size();
            segment_start = offset;
            matched_special_token = true;
            break;
        }

        if (!matched_special_token) {
            ++offset;
        }
    }

    if (segment_start < normalized_text.size()) {
        segments.push_back(TokenizerTextSegment{
            std::string(normalized_text.substr(segment_start)),
            false,
            -1,
        });
    }

    return segments;
}

}  // namespace nli
