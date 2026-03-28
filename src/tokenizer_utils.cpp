#include "tokenizer_utils.h"

#include <string>
#include <string_view>

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

}  // namespace

namespace nli {

std::string NormalizeDebertaTokenizerInput(std::string_view text) {
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
            if (seen_non_space && !previous_was_space) {
                normalized.push_back(' ');
                previous_was_space = true;
            }
            continue;
        }

        normalized.push_back(ch);
        previous_was_space = false;
        seen_non_space = true;
    }

    while (!normalized.empty() && normalized.back() == ' ') {
        normalized.pop_back();
    }

    return normalized;
}

}  // namespace nli
