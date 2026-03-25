#include "session_factory.h"

#include <sstream>
#include <stdexcept>

namespace {

void VerifyCoreMLSuccess() {
    std::ostringstream log;
    int cpu_attempts = 0;

    auto result = nli::CreateSessionWithPreferredBackend(
        true,
        []() { return 7; },
        [&]() {
            ++cpu_attempts;
            return 3;
        },
        log);

    if (result.backend != nli::SessionBackend::kCoreML) {
        throw std::runtime_error("expected CoreML backend");
    }
    if (result.value != 7) {
        throw std::runtime_error("expected CoreML result value");
    }
    if (result.used_fallback) {
        throw std::runtime_error("did not expect fallback");
    }
    if (cpu_attempts != 0) {
        throw std::runtime_error("cpu fallback should not run on CoreML success");
    }
}

void VerifyCoreMLFallback() {
    std::ostringstream log;
    int cpu_attempts = 0;

    auto result = nli::CreateSessionWithPreferredBackend(
        true,
        []() -> int {
            throw std::runtime_error("coreml init failed");
        },
        [&]() {
            ++cpu_attempts;
            return 11;
        },
        log);

    if (result.backend != nli::SessionBackend::kCPU) {
        throw std::runtime_error("expected CPU backend after fallback");
    }
    if (result.value != 11) {
        throw std::runtime_error("expected CPU fallback result value");
    }
    if (!result.used_fallback) {
        throw std::runtime_error("expected CoreML fallback to be recorded");
    }
    if (cpu_attempts != 1) {
        throw std::runtime_error("expected one CPU fallback attempt");
    }
}

void VerifyCPUOnlyPath() {
    std::ostringstream log;
    int coreml_attempts = 0;

    auto result = nli::CreateSessionWithPreferredBackend(
        false,
        [&]() {
            ++coreml_attempts;
            return 5;
        },
        []() { return 13; },
        log);

    if (result.backend != nli::SessionBackend::kCPU) {
        throw std::runtime_error("expected CPU backend in CPU-only mode");
    }
    if (result.value != 13) {
        throw std::runtime_error("expected CPU-only result value");
    }
    if (result.used_fallback) {
        throw std::runtime_error("did not expect fallback in CPU-only mode");
    }
    if (coreml_attempts != 0) {
        throw std::runtime_error("should not attempt CoreML in CPU-only mode");
    }
}

}  // namespace

int main() {
    VerifyCoreMLSuccess();
    VerifyCoreMLFallback();
    VerifyCPUOnlyPath();
    return 0;
}
