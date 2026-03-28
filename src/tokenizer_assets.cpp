#include "tokenizer_assets.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

namespace {

using ordered_json = nlohmann::ordered_json;

std::optional<ordered_json> ReadOptionalJsonFile(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        return std::nullopt;
    }

    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Failed to open tokenizer asset file: " + path.string());
    }

    ordered_json root;
    try {
        input >> root;
    } catch (const nlohmann::json::parse_error& e) {
        throw std::runtime_error(
            "Failed to parse tokenizer asset JSON '" + path.string() + "': " + e.what());
    }
    return root;
}

std::optional<std::string> ReadTokenContent(const ordered_json& field) {
    if (field.is_string()) {
        return field.get<std::string>();
    }
    if (field.is_object()) {
        auto content_it = field.find("content");
        if (content_it != field.end() && content_it->is_string()) {
            return content_it->get<std::string>();
        }
    }
    return std::nullopt;
}

void ApplySpecialTokenName(
    nli::TokenizerAssetConfig& config,
    const std::string& key,
    const std::string& value) {
    if (key == "pad_token") {
        config.pad_token = value;
    } else if (key == "cls_token" || key == "bos_token") {
        config.cls_token = value;
    } else if (key == "sep_token" || key == "eos_token") {
        config.sep_token = value;
    } else if (key == "unk_token") {
        config.unk_token = value;
    } else if (key == "mask_token") {
        config.mask_token = value;
    }
}

void SetSpecialTokenId(
    nli::TokenizerSpecialTokenIds& ids,
    const std::string& token,
    int64_t value) {
    if (token == "[PAD]") {
        ids.pad = value;
    } else if (token == "[CLS]") {
        ids.cls = value;
    } else if (token == "[SEP]") {
        ids.sep = value;
    } else if (token == "[UNK]") {
        ids.unk = value;
    } else if (token == "[MASK]") {
        ids.mask = value;
    }
}

void ApplyTokenizerConfigJson(
    const ordered_json& root,
    nli::TokenizerAssetConfig& config) {
    for (const auto& key : {"pad_token", "cls_token", "sep_token", "unk_token", "mask_token"}) {
        auto it = root.find(key);
        if (it == root.end()) {
            continue;
        }
        if (auto token_content = ReadTokenContent(*it)) {
            ApplySpecialTokenName(config, key, *token_content);
        }
    }

    auto lower_it = root.find("do_lower_case");
    if (lower_it != root.end() && lower_it->is_boolean()) {
        config.do_lower_case = lower_it->get<bool>();
    }

    auto max_length_it = root.find("model_max_length");
    if (max_length_it != root.end() && max_length_it->is_number_unsigned()) {
        config.max_length = max_length_it->get<size_t>();
    }

    auto added_tokens_decoder_it = root.find("added_tokens_decoder");
    if (added_tokens_decoder_it == root.end() || !added_tokens_decoder_it->is_object()) {
        return;
    }

    for (const auto& [id_text, token_info] : added_tokens_decoder_it->items()) {
        if (!token_info.is_object()) {
            continue;
        }
        auto content_it = token_info.find("content");
        if (content_it == token_info.end() || !content_it->is_string()) {
            continue;
        }
        SetSpecialTokenId(
            config.special_token_ids,
            content_it->get<std::string>(),
            std::stoll(id_text));
    }
}

void ApplySpecialTokensMapJson(
    const ordered_json& root,
    nli::TokenizerAssetConfig& config) {
    for (const auto& [key, value] : root.items()) {
        if (auto token_content = ReadTokenContent(value)) {
            ApplySpecialTokenName(config, key, *token_content);
        }
    }
}

void ApplyTokenizerJson(
    const ordered_json& root,
    nli::TokenizerAssetConfig& config) {
    auto added_tokens_it = root.find("added_tokens");
    if (added_tokens_it != root.end() && added_tokens_it->is_array()) {
        for (const auto& token : *added_tokens_it) {
            if (!token.is_object()) {
                continue;
            }
            auto id_it = token.find("id");
            auto content_it = token.find("content");
            if (id_it == token.end() || content_it == token.end() ||
                !id_it->is_number_integer() || !content_it->is_string()) {
                continue;
            }
            SetSpecialTokenId(
                config.special_token_ids,
                content_it->get<std::string>(),
                id_it->get<int64_t>());
        }
    }

    auto normalizer_it = root.find("normalizer");
    if (normalizer_it != root.end() && normalizer_it->is_object()) {
        auto sequence_it = normalizer_it->find("normalizers");
        if (sequence_it != normalizer_it->end() && sequence_it->is_array()) {
            for (const auto& normalizer : *sequence_it) {
                if (!normalizer.is_object()) {
                    continue;
                }
                const auto type_it = normalizer.find("type");
                if (type_it == normalizer.end() || !type_it->is_string()) {
                    continue;
                }
                const std::string type = type_it->get<std::string>();
                if (type == "Strip") {
                    auto left_it = normalizer.find("strip_left");
                    auto right_it = normalizer.find("strip_right");
                    if (left_it != normalizer.end() && left_it->is_boolean()) {
                        config.strip_left = left_it->get<bool>();
                    }
                    if (right_it != normalizer.end() && right_it->is_boolean()) {
                        config.strip_right = right_it->get<bool>();
                    }
                } else if (type == "Replace") {
                    const auto pattern_it = normalizer.find("pattern");
                    const auto content_it = normalizer.find("content");
                    if (pattern_it == normalizer.end() || content_it == normalizer.end() ||
                        !pattern_it->is_object() || !content_it->is_string()) {
                        continue;
                    }
                    auto regex_it = pattern_it->find("Regex");
                    if (regex_it != pattern_it->end() && regex_it->is_string() &&
                        regex_it->get<std::string>() == " {2,}" &&
                        content_it->get<std::string>() == " ") {
                        config.collapse_spaces = true;
                    }
                }
            }
        }
    }

    auto pre_tokenizer_it = root.find("pre_tokenizer");
    if (pre_tokenizer_it != root.end() && pre_tokenizer_it->is_object()) {
        auto pretokenizers_it = pre_tokenizer_it->find("pretokenizers");
        if (pretokenizers_it != pre_tokenizer_it->end() && pretokenizers_it->is_array()) {
            for (const auto& pretokenizer : *pretokenizers_it) {
                if (!pretokenizer.is_object()) {
                    continue;
                }
                auto type_it = pretokenizer.find("type");
                if (type_it == pretokenizer.end() || !type_it->is_string()) {
                    continue;
                }
                if (type_it->get<std::string>() != "Metaspace") {
                    continue;
                }
                auto add_prefix_space_it = pretokenizer.find("add_prefix_space");
                if (add_prefix_space_it != pretokenizer.end() &&
                    add_prefix_space_it->is_boolean()) {
                    config.add_prefix_space = add_prefix_space_it->get<bool>();
                }
            }
        }
    }

    auto post_processor_it = root.find("post_processor");
    if (post_processor_it != root.end() && post_processor_it->is_object()) {
        auto pair_it = post_processor_it->find("pair");
        if (pair_it != post_processor_it->end() && pair_it->is_array()) {
            for (const auto& item : *pair_it) {
                if (!item.is_object()) {
                    continue;
                }
                auto cls_it = item.find("SpecialToken");
                if (cls_it != item.end() && cls_it->is_object()) {
                    auto id_it = cls_it->find("id");
                    auto type_id_it = cls_it->find("type_id");
                    if (id_it == cls_it->end() || type_id_it == cls_it->end() ||
                        !id_it->is_string() || !type_id_it->is_number_integer()) {
                        continue;
                    }
                    const std::string id = id_it->get<std::string>();
                    const int64_t type_id = type_id_it->get<int64_t>();
                    if (id == config.cls_token) {
                        config.template_ids.cls = type_id;
                    } else if (id == config.sep_token) {
                        if (config.template_ids.first_sep == 0 && type_id == 0) {
                            config.template_ids.first_sep = type_id;
                        } else {
                            config.template_ids.second_sep = type_id;
                        }
                    }
                    continue;
                }

                auto sequence_it = item.find("Sequence");
                if (sequence_it == item.end() || !sequence_it->is_object()) {
                    continue;
                }
                auto id_it = sequence_it->find("id");
                auto type_id_it = sequence_it->find("type_id");
                if (id_it == sequence_it->end() || type_id_it == sequence_it->end() ||
                    !id_it->is_string() || !type_id_it->is_number_integer()) {
                    continue;
                }
                const std::string id = id_it->get<std::string>();
                const int64_t type_id = type_id_it->get<int64_t>();
                if (id == "A") {
                    config.template_ids.first_sequence = type_id;
                } else if (id == "B") {
                    config.template_ids.second_sequence = type_id;
                }
            }
        }
    }
}

}  // namespace

namespace nli {

TokenizerAssetConfig DefaultTokenizerAssetConfig() {
    return TokenizerAssetConfig{};
}

TokenizerAssetConfig LoadTokenizerAssetConfigFromDir(const std::string& asset_dir) {
    TokenizerAssetConfig config = DefaultTokenizerAssetConfig();
    const std::filesystem::path base_dir(asset_dir);

    if (const auto tokenizer_config = ReadOptionalJsonFile(base_dir / "tokenizer_config.json")) {
        ApplyTokenizerConfigJson(*tokenizer_config, config);
    }
    if (const auto special_tokens_map = ReadOptionalJsonFile(base_dir / "special_tokens_map.json")) {
        ApplySpecialTokensMapJson(*special_tokens_map, config);
    }
    if (const auto tokenizer_json = ReadOptionalJsonFile(base_dir / "tokenizer.json")) {
        ApplyTokenizerJson(*tokenizer_json, config);
    }

    return config;
}

TokenizerAssetConfig LoadTokenizerAssetConfigForSentencePiece(const std::string& sentencepiece_path) {
    return LoadTokenizerAssetConfigFromDir(
        std::filesystem::path(sentencepiece_path).parent_path().string());
}

}  // namespace nli
