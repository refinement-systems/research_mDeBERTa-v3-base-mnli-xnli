#pragma once

#include <exception>
#include <functional>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <ostream>
#include <string>
#include <type_traits>
#include <utility>

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

SessionCreationResult<Ort::Session> CreateInferenceSession(
    const Ort::Env& env,
    const std::string& model_path,
    std::ostream& log);

}  // namespace nli
