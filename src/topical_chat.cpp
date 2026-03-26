#include "topical_chat.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <stdexcept>
#include <string>

namespace {

using ordered_json = nlohmann::ordered_json;

const ordered_json& RequireObjectField(
    const ordered_json& object,
    const std::string& field_name,
    const std::string& context) {
    auto field_it = object.find(field_name);
    if (field_it == object.end()) {
        throw std::runtime_error(context + " is missing required field '" + field_name + "'");
    }
    return *field_it;
}

std::string RequireStringField(
    const ordered_json& object,
    const std::string& field_name,
    const std::string& context) {
    const ordered_json& field = RequireObjectField(object, field_name, context);
    if (!field.is_string()) {
        throw std::runtime_error(context + " field '" + field_name + "' must be a string");
    }
    return field.get<std::string>();
}

}  // namespace

namespace nli {

std::vector<TopicalChatTurnInput> ReadTopicalChatTurnInputs(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Failed to open Topical Chat file: " + path);
    }

    ordered_json root;
    try {
        input >> root;
    } catch (const nlohmann::json::parse_error& e) {
        throw std::runtime_error("Failed to parse Topical Chat JSON '" + path + "': " + e.what());
    }

    if (!root.is_object()) {
        throw std::runtime_error("Topical Chat file must contain a top-level object: " + path);
    }

    std::vector<TopicalChatTurnInput> turns;

    for (const auto& [conversation_id, conversation] : root.items()) {
        if (!conversation.is_object()) {
            throw std::runtime_error(
                "Conversation '" + conversation_id + "' must be a JSON object");
        }

        const ordered_json& content = RequireObjectField(
            conversation,
            "content",
            "Conversation '" + conversation_id + "'");
        if (!content.is_array()) {
            throw std::runtime_error(
                "Conversation '" + conversation_id + "' field 'content' must be an array");
        }

        for (size_t turn_index = 0; turn_index < content.size(); ++turn_index) {
            const auto& turn = content[turn_index];
            const std::string turn_context =
                "Conversation '" + conversation_id + "' turn " + std::to_string(turn_index + 1);

            if (!turn.is_object()) {
                throw std::runtime_error(turn_context + " must be a JSON object");
            }

            turns.push_back(TopicalChatTurnInput{
                RequireStringField(turn, "message", turn_context),
                RequireStringField(turn, "sentiment", turn_context),
            });
        }
    }

    return turns;
}

}  // namespace nli
