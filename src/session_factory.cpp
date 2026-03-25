#include "session_factory.h"

#if defined(NLI_HAS_COREML_PROVIDER)
#include <onnxruntime/core/providers/coreml/coreml_provider_factory.h>
#endif

namespace nli {
namespace {

Ort::SessionOptions CreateBaseSessionOptions() {
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    return session_options;
}

Ort::Session CreateCpuSession(const Ort::Env& env, const std::string& model_path) {
    auto session_options = CreateBaseSessionOptions();
    return Ort::Session(env, model_path.c_str(), session_options);
}

#if defined(NLI_HAS_COREML_PROVIDER)
Ort::Session CreateCoreMLSession(const Ort::Env& env, const std::string& model_path) {
    auto session_options = CreateBaseSessionOptions();
    Ort::ThrowOnError(
        OrtSessionOptionsAppendExecutionProvider_CoreML(session_options, COREML_FLAG_USE_NONE));
    return Ort::Session(env, model_path.c_str(), session_options);
}
#endif

}  // namespace

SessionCreationResult<Ort::Session> CreateInferenceSession(
    const Ort::Env& env,
    const std::string& model_path,
    std::ostream& log) {
#if defined(NLI_HAS_COREML_PROVIDER)
    auto result = CreateSessionWithPreferredBackend(
        true,
        [&]() { return CreateCoreMLSession(env, model_path); },
        [&]() { return CreateCpuSession(env, model_path); },
        log);

    log << "Using ONNX Runtime backend: " << SessionBackendName(result.backend);
    if (result.used_fallback) {
        log << " (fallback)";
    }
    log << "\n";

    return result;
#else
    auto session = CreateCpuSession(env, model_path);
    log << "Using ONNX Runtime backend: " << SessionBackendName(SessionBackend::kCPU) << "\n";
    return SessionCreationResult<Ort::Session>{
        std::move(session),
        SessionBackend::kCPU,
        false,
    };
#endif
}

}  // namespace nli
