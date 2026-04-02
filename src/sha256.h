#pragma once

#include <filesystem>
#include <string>

namespace nli {

std::string ComputeFileSha256Hex(const std::filesystem::path& path);

}  // namespace nli

