#pragma once

#include "nli_inference.h"
#include "study_types.h"

#include <functional>
#include <iosfwd>
#include <memory>
#include <string>

namespace nli {

class StudyPredictor {
public:
    virtual ~StudyPredictor() = default;
    virtual SessionBackend ActualBackend() const = 0;
    virtual bool UsedFallback() const = 0;
    virtual NliLogits PredictLogits(const std::string& premise, const std::string& hypothesis) = 0;
};

using StudyPredictorFactory = std::function<std::unique_ptr<StudyPredictor>(
    const std::string& model_path,
    const std::string& sentencepiece_path,
    SessionBackend backend,
    std::ostream& log)>;

StudyPredictorFactory DefaultStudyPredictorFactory();

void InitializeStudyWorkspace(const StudyInitCommandLineOptions& options, std::ostream& log);
void RunStudyEvaluation(const StudyRunCommandLineOptions& options, std::ostream& log);
void RunStudyEvaluation(
    const StudyRunCommandLineOptions& options,
    const StudyPredictorFactory& predictor_factory,
    std::ostream& log);

}  // namespace nli
