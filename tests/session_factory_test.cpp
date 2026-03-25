#include "session_factory.h"

#include <sstream>
#include <stdexcept>
#include <vector>

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

void VerifyExplicitCPUSelection() {
    std::ostringstream log;
    int coreml_attempts = 0;

    auto result = nli::CreateSessionForBackend(
        nli::SessionBackend::kCPU,
        true,
        [&]() {
            ++coreml_attempts;
            return 5;
        },
        []() { return 17; },
        log);

    if (result.backend != nli::SessionBackend::kCPU) {
        throw std::runtime_error("expected explicit CPU selection to use CPU");
    }
    if (result.value != 17) {
        throw std::runtime_error("expected explicit CPU result value");
    }
    if (result.used_fallback) {
        throw std::runtime_error("did not expect fallback for explicit CPU selection");
    }
    if (coreml_attempts != 0) {
        throw std::runtime_error("explicit CPU selection should not attempt CoreML");
    }
}

void VerifyExplicitCoreMLSelectionWhenUnavailable() {
    std::ostringstream log;
    int cpu_attempts = 0;

    try {
        (void)nli::CreateSessionForBackend(
            nli::SessionBackend::kCoreML,
            false,
            []() { return 5; },
            [&]() {
                ++cpu_attempts;
                return 17;
            },
            log);
    } catch (const std::invalid_argument&) {
        if (cpu_attempts != 0) {
            throw std::runtime_error("unavailable CoreML selection should not fall back to CPU");
        }
        return;
    }

    throw std::runtime_error("expected unavailable CoreML selection to throw");
}

void VerifyBackendMetadata() {
    const std::vector<nli::SessionBackend> available_backends = nli::AvailableSessionBackends();
    const std::vector<std::string> backend_names = nli::AvailableSessionBackendOptionNames();

    if (available_backends.empty()) {
        throw std::runtime_error("expected at least one available backend");
    }
    if (available_backends.size() != backend_names.size()) {
        throw std::runtime_error("backend names should align with available backends");
    }
    if (available_backends.front() != nli::SessionBackend::kCPU) {
        throw std::runtime_error("expected CPU to be the first available backend");
    }
    if (backend_names.front() != "cpu") {
        throw std::runtime_error("expected CPU backend option name");
    }
    if (nli::ParseSessionBackendOption("cpu") != nli::SessionBackend::kCPU) {
        throw std::runtime_error("expected cpu option parsing to succeed");
    }

#if defined(NLI_HAS_COREML_PROVIDER)
    if (available_backends.size() != 2) {
        throw std::runtime_error("expected CoreML builds to expose two backends");
    }
    if (available_backends.back() != nli::SessionBackend::kCoreML) {
        throw std::runtime_error("expected CoreML backend to be available");
    }
    if (backend_names.back() != "coreml") {
        throw std::runtime_error("expected CoreML backend option name");
    }
    if (nli::DefaultSessionBackend() != nli::SessionBackend::kCoreML) {
        throw std::runtime_error("expected CoreML to remain the default on CoreML builds");
    }
    if (nli::ParseSessionBackendOption("coreml") != nli::SessionBackend::kCoreML) {
        throw std::runtime_error("expected coreml option parsing to succeed");
    }
#else
    if (available_backends.size() != 1) {
        throw std::runtime_error("expected CPU-only builds to expose one backend");
    }
    if (nli::DefaultSessionBackend() != nli::SessionBackend::kCPU) {
        throw std::runtime_error("expected CPU to be the default on CPU-only builds");
    }
    try {
        (void)nli::ParseSessionBackendOption("coreml");
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("expected CPU-only builds to reject the coreml option");
#endif
}

}  // namespace

int main() {
    VerifyCoreMLSuccess();
    VerifyCoreMLFallback();
    VerifyCPUOnlyPath();
    VerifyExplicitCPUSelection();
    VerifyExplicitCoreMLSelectionWhenUnavailable();
    VerifyBackendMetadata();
    return 0;
}
