#include "nli_eval.h"

#include "nli_inference.h"

#include <algorithm>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace {

std::string TrimTrailingCarriageReturn(std::string line) {
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }
    return line;
}

std::vector<std::string> SplitTsvLine(const std::string& line) {
    std::vector<std::string> fields;
    size_t field_start = 0;

    while (field_start <= line.size()) {
        const size_t tab_pos = line.find('\t', field_start);
        if (tab_pos == std::string::npos) {
            fields.push_back(line.substr(field_start));
            break;
        }

        fields.push_back(line.substr(field_start, tab_pos - field_start));
        field_start = tab_pos + 1;
    }

    return fields;
}

std::optional<size_t> FindColumnIndex(
    const std::unordered_map<std::string, size_t>& header_map,
    const std::string& name) {
    auto it = header_map.find(name);
    if (it == header_map.end()) {
        return std::nullopt;
    }
    return it->second;
}

bool IsKnownNliLabel(std::string_view label) {
    return std::find(
               nli::kNliScoreLabels.begin(),
               nli::kNliScoreLabels.end(),
               label) != nli::kNliScoreLabels.end();
}

}  // namespace

namespace nli {

std::vector<NliEvalExample> ReadNliEvalExamples(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Failed to open NLI eval file: " + path);
    }

    std::string header_line;
    if (!std::getline(input, header_line)) {
        throw std::runtime_error("NLI eval file is empty: " + path);
    }

    const auto header_fields = SplitTsvLine(TrimTrailingCarriageReturn(header_line));
    if (header_fields.empty()) {
        throw std::runtime_error("NLI eval file has an empty header row: " + path);
    }

    std::unordered_map<std::string, size_t> header_map;
    for (size_t i = 0; i < header_fields.size(); ++i) {
        header_map.emplace(header_fields[i], i);
    }

    const auto premise_index = FindColumnIndex(header_map, "premise");
    const auto hypothesis_index = FindColumnIndex(header_map, "hypothesis");
    if (!premise_index || !hypothesis_index) {
        throw std::runtime_error(
            "NLI eval file must include 'premise' and 'hypothesis' columns: " + path);
    }

    auto label_index = FindColumnIndex(header_map, "label");
    if (!label_index) {
        label_index = FindColumnIndex(header_map, "gold_label");
    }
    const auto id_index = FindColumnIndex(header_map, "id");

    std::vector<NliEvalExample> examples;
    std::string line;
    size_t line_number = 1;

    while (std::getline(input, line)) {
        ++line_number;
        line = TrimTrailingCarriageReturn(line);
        if (line.empty()) {
            continue;
        }

        const auto fields = SplitTsvLine(line);
        if (fields.size() != header_fields.size()) {
            throw std::runtime_error(
                "NLI eval row " + std::to_string(line_number) +
                " has " + std::to_string(fields.size()) +
                " fields but header has " + std::to_string(header_fields.size()));
        }

        NliEvalExample example;
        if (id_index) {
            example.id = fields[*id_index];
        }
        example.premise = fields[*premise_index];
        example.hypothesis = fields[*hypothesis_index];

        if (label_index && !fields[*label_index].empty()) {
            if (!IsKnownNliLabel(fields[*label_index])) {
                throw std::runtime_error(
                    "NLI eval row " + std::to_string(line_number) +
                    " has unsupported label '" + fields[*label_index] + "'");
            }
            example.label = fields[*label_index];
        }

        examples.push_back(std::move(example));
    }

    return examples;
}

}  // namespace nli
