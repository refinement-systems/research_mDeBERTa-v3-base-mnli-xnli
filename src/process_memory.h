#pragma once

#include <cstdint>

namespace nli {

struct ProcessMemorySnapshot {
    bool available = false;
    uint64_t resident_bytes = 0;
    uint64_t peak_resident_bytes = 0;
};

ProcessMemorySnapshot GetProcessMemorySnapshot();

}  // namespace nli
