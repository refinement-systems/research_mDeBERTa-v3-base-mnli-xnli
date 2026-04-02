#include "session_factory.h"

#if defined(NLI_HAS_COREML_PROVIDER)
#include <onnxruntime/core/providers/coreml/coreml_provider_factory.h>
#endif

namespace nli {
namespace {

Ort::SessionOptions CreateBaseSessionOptions(GraphOptimizationLevel optimization_level) {
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(optimization_level);
    return session_options;
}

Ort::Session CreateCpuSession(const Ort::Env& env, const std::string& model_path) {
    auto session_options = CreateBaseSessionOptions(GraphOptimizationLevel::ORT_ENABLE_ALL);
    return Ort::Session(env, model_path.c_str(), session_options);
}

#if defined(NLI_HAS_COREML_PROVIDER)
Ort::Session CreateCoreMLSession(const Ort::Env& env, const std::string& model_path) {
    // BASIC avoids aggressive fusions that break the fp16 CoreML attempt models
    // while still allowing standard ORT preprocessing on this backend.
    auto session_options = CreateBaseSessionOptions(GraphOptimizationLevel::ORT_ENABLE_BASIC);
    Ort::ThrowOnError(
        OrtSessionOptionsAppendExecutionProvider_CoreML(session_options, COREML_FLAG_USE_NONE));
    return Ort::Session(env, model_path.c_str(), session_options);
}
#endif

}  // namespace

SessionCreationResult<Ort::Session> CreateInferenceSession(
    const Ort::Env& env,
    const std::string& model_path,
    SessionBackend backend,
    std::ostream& log) {
    auto result = CreateSessionForBackend(
        backend,
#if defined(NLI_HAS_COREML_PROVIDER)
        true,
        [&]() { return CreateCoreMLSession(env, model_path); },
#else
        false,
        []() -> Ort::Session {
            throw std::invalid_argument("CoreML backend is not available in this build.");
        },
#endif
        [&]() { return CreateCpuSession(env, model_path); },
        log);

    log << "Using ONNX Runtime backend: " << SessionBackendName(result.backend);
    if (result.used_fallback) {
        log << " (fallback)";
    }
    log << "\n";

    return result;
}

SessionCreationResult<Ort::Session> CreateInferenceSession(
    const Ort::Env& env,
    const std::string& model_path,
    std::ostream& log) {
    return CreateInferenceSession(env, model_path, DefaultSessionBackend(), log);
}

}  // namespace nli
