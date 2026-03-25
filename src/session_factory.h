#pragma once

#include <exception>
#include <functional>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace nli {

enum class SessionBackend {
    kCPU,
    kCoreML,
};

template <typename T>
struct SessionCreationResult {
    T value;
    SessionBackend backend;
    bool used_fallback;
};

inline const char* SessionBackendName(SessionBackend backend) {
    switch (backend) {
        case SessionBackend::kCPU:
            return "CPU";
        case SessionBackend::kCoreML:
            return "CoreML";
    }

    return "unknown";
}

inline const char* SessionBackendOptionName(SessionBackend backend) {
    switch (backend) {
        case SessionBackend::kCPU:
            return "cpu";
        case SessionBackend::kCoreML:
            return "coreml";
    }

    return "unknown";
}

inline std::vector<SessionBackend> AvailableSessionBackends() {
    std::vector<SessionBackend> backends = {SessionBackend::kCPU};

#if defined(NLI_HAS_COREML_PROVIDER)
    backends.push_back(SessionBackend::kCoreML);
#endif

    return backends;
}

inline std::vector<std::string> AvailableSessionBackendOptionNames() {
    std::vector<std::string> backend_names;
    for (SessionBackend backend : AvailableSessionBackends()) {
        backend_names.emplace_back(SessionBackendOptionName(backend));
    }
    return backend_names;
}

inline SessionBackend DefaultSessionBackend() {
#if defined(NLI_HAS_COREML_PROVIDER)
    return SessionBackend::kCoreML;
#else
    return SessionBackend::kCPU;
#endif
}

inline SessionBackend ParseSessionBackendOption(const std::string& backend_name) {
    for (SessionBackend backend : AvailableSessionBackends()) {
        if (backend_name == SessionBackendOptionName(backend)) {
            return backend;
        }
    }

    throw std::invalid_argument("Unsupported backend option for this build: " + backend_name);
}

template <typename TryCoreMLFn, typename CreateCPUFn>
auto CreateSessionWithPreferredBackend(
    bool can_try_coreml,
    TryCoreMLFn&& try_coreml,
    CreateCPUFn&& create_cpu,
    std::ostream& log)
    -> SessionCreationResult<std::invoke_result_t<CreateCPUFn>> {
    using ResultT = std::invoke_result_t<CreateCPUFn>;

    if (can_try_coreml) {
        try {
            return SessionCreationResult<ResultT>{
                std::invoke(std::forward<TryCoreMLFn>(try_coreml)),
                SessionBackend::kCoreML,
                false,
            };
        } catch (const std::exception& e) {
            log << "CoreML backend unavailable, falling back to CPU: " << e.what() << "\n";
        } catch (...) {
            log << "CoreML backend unavailable, falling back to CPU.\n";
        }

        return SessionCreationResult<ResultT>{
            std::invoke(std::forward<CreateCPUFn>(create_cpu)),
            SessionBackend::kCPU,
            true,
        };
    }

    return SessionCreationResult<ResultT>{
        std::invoke(std::forward<CreateCPUFn>(create_cpu)),
        SessionBackend::kCPU,
        false,
    };
}

template <typename CreateCoreMLFn, typename CreateCPUFn>
auto CreateSessionForBackend(
    SessionBackend backend,
    bool can_try_coreml,
    CreateCoreMLFn&& create_coreml,
    CreateCPUFn&& create_cpu,
    std::ostream& log)
    -> SessionCreationResult<std::invoke_result_t<CreateCPUFn>> {
    switch (backend) {
        case SessionBackend::kCPU:
            return SessionCreationResult<std::invoke_result_t<CreateCPUFn>>{
                std::invoke(std::forward<CreateCPUFn>(create_cpu)),
                SessionBackend::kCPU,
                false,
            };
        case SessionBackend::kCoreML:
            if (!can_try_coreml) {
                throw std::invalid_argument("CoreML backend is not available in this build.");
            }
            return CreateSessionWithPreferredBackend(
                true,
                std::forward<CreateCoreMLFn>(create_coreml),
                std::forward<CreateCPUFn>(create_cpu),
                log);
    }

    throw std::invalid_argument("Unsupported session backend.");
}

SessionCreationResult<Ort::Session> CreateInferenceSession(
    const Ort::Env& env,
    const std::string& model_path,
    SessionBackend backend,
    std::ostream& log);

SessionCreationResult<Ort::Session> CreateInferenceSession(
    const Ort::Env& env,
    const std::string& model_path,
    std::ostream& log);

}  // namespace nli
