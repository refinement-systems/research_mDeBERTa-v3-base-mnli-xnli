#include "process_memory.h"

#if defined(__APPLE__)
#include <mach/mach.h>
#elif defined(__linux__)
#include <sys/resource.h>
#include <unistd.h>

#include <cstdint>
#include <fstream>
#endif

namespace nli {

ProcessMemorySnapshot GetProcessMemorySnapshot() {
    ProcessMemorySnapshot snapshot;

#if defined(__APPLE__)
    mach_task_basic_info_data_t info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(
            mach_task_self(),
            MACH_TASK_BASIC_INFO,
            reinterpret_cast<task_info_t>(&info),
            &count) == KERN_SUCCESS) {
        snapshot.available = true;
        snapshot.resident_bytes = static_cast<uint64_t>(info.resident_size);
        snapshot.peak_resident_bytes = static_cast<uint64_t>(info.resident_size_max);
    }
#elif defined(__linux__)
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size > 0) {
        std::ifstream statm("/proc/self/statm");
        uint64_t total_pages = 0;
        uint64_t resident_pages = 0;
        if (statm >> total_pages >> resident_pages) {
            snapshot.resident_bytes = resident_pages * static_cast<uint64_t>(page_size);
        }
    }

    struct rusage usage {};
    if (getrusage(RUSAGE_SELF, &usage) == 0 && usage.ru_maxrss > 0) {
        snapshot.peak_resident_bytes = static_cast<uint64_t>(usage.ru_maxrss) * 1024ULL;
    }

    snapshot.available =
        snapshot.resident_bytes > 0 || snapshot.peak_resident_bytes > 0;
#endif

    return snapshot;
}

}  // namespace nli
