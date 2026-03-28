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
    // Approximate the Hugging Face DeBERTa tokenizer's whitespace handling:
    // collapse whitespace runs to a single space and strip trailing whitespace.
    std::string normalized;
    normalized.reserve(text.size());

    bool previous_was_space = false;
    for (const char ch : text) {
        if (IsAsciiWhitespace(ch)) {
            if (!previous_was_space) {
                normalized.push_back(' ');
                previous_was_space = true;
            }
            continue;
        }

        normalized.push_back(ch);
        previous_was_space = false;
    }

    while (!normalized.empty() && normalized.back() == ' ') {
        normalized.pop_back();
    }

    return normalized;
}

}  // namespace nli
