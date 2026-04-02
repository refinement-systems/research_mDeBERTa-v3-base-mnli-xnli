#include "study_command_line.h"
#include "study_workflow.h"

#include <sqlite3.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <nlohmann/json.hpp>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ordered_json = nlohmann::ordered_json;

std::filesystem::path RepoRoot() {
    return std::filesystem::path(NLI_SOURCE_DIR);
}

std::filesystem::path EvalFixturePath() {
    return RepoRoot() / "tests/data/nli_eval_fixture.tsv";
}

std::filesystem::path FakeGeneratorPath() {
    return RepoRoot() / "tests/data/fake_generator.py";
}

void ExpectParserExitCode(const std::function<void()>& callback, int expected_code) {
    try {
        callback();
    } catch (int code) {
        if (code == expected_code) {
            return;
        }
        throw std::runtime_error(
            "expected parser exit code " + std::to_string(expected_code) +
            ", got " + std::to_string(code));
    }
    throw std::runtime_error(
        "expected parser exit code " + std::to_string(expected_code));
}

optparse::OptionParserExcept MakeStudyInitParser() {
    optparse::OptionParserExcept parser;
    nli::ConfigureStudyInitOptionParser(parser);
    parser.prog("nli-study init");
    return parser;
}

optparse::OptionParserExcept MakeStudyRunParser() {
    optparse::OptionParserExcept parser;
    nli::ConfigureStudyRunOptionParser(parser);
    parser.prog("nli-study run");
    return parser;
}

class FakePredictor final : public nli::StudyPredictor {
public:
    FakePredictor(std::string model_path, nli::SessionBackend actual_backend, bool used_fallback)
        : model_path_(std::move(model_path)),
          actual_backend_(actual_backend),
          used_fallback_(used_fallback) {}

    nli::SessionBackend ActualBackend() const override {
        return actual_backend_;
    }

    bool UsedFallback() const override {
        return used_fallback_;
    }

    nli::NliLogits PredictLogits(const std::string& premise, const std::string& hypothesis) override {
        if (model_path_.find("model.onnx") != std::string::npos) {
            return nli::NliLogits{5.0f, 1.0f, 0.0f};
        }
        if (premise.find("Germany") != std::string::npos || hypothesis.find("Germany") != std::string::npos) {
            return nli::NliLogits{0.5f, 1.0f, 3.0f};
        }
        if (premise.find("Merkel") != std::string::npos) {
            return nli::NliLogits{1.5f, 2.5f, 1.0f};
        }
        return nli::NliLogits{4.0f, 1.0f, 0.5f};
    }

private:
    std::string model_path_;
    nli::SessionBackend actual_backend_;
    bool used_fallback_;
};

nli::StudyPredictorFactory FakePredictorFactory() {
    return [](const std::string& model_path,
              const std::string&,
              nli::SessionBackend backend,
              std::ostream&) {
        return std::make_unique<FakePredictor>(model_path, backend, false);
    };
}

nli::StudyPredictorFactory CoreMLFallbackPredictorFactory() {
    return [](const std::string& model_path,
              const std::string&,
              nli::SessionBackend backend,
              std::ostream&) {
        if (backend == nli::SessionBackend::kCoreML) {
            return std::make_unique<FakePredictor>(
                model_path,
                nli::SessionBackend::kCPU,
                true);
        }
        return std::make_unique<FakePredictor>(model_path, backend, false);
    };
}

std::string ReadFile(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Failed to open file: " + path.string());
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

struct TempStudyWorkspace {
    std::filesystem::path root;
    std::filesystem::path scratchpad_root;
    std::filesystem::path catalog_path;
};

TempStudyWorkspace CreateTempStudyWorkspace() {
    const auto unique_name =
        "nli-study-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    const std::filesystem::path root = std::filesystem::temp_directory_path() / unique_name;
    std::filesystem::create_directories(root);
    return TempStudyWorkspace{
        root,
        root / "scratchpad",
        root / "catalog.json",
    };
}

void RemoveTempStudyWorkspace(const TempStudyWorkspace& workspace) {
    std::filesystem::remove_all(workspace.root);
}

void WriteFixtureDatasets(const TempStudyWorkspace& workspace) {
    const std::filesystem::path dataset_root = workspace.scratchpad_root / "datasets";
    std::filesystem::create_directories(dataset_root);
    for (const std::string& dataset_name : {
             "mnli-train-calibration-64-per-label.tsv",
             "mnli-train-search-validation-skip64-64-per-label.tsv",
         }) {
        std::filesystem::copy_file(
            EvalFixturePath(),
            dataset_root / dataset_name,
            std::filesystem::copy_options::overwrite_existing);
    }
}

void WriteReferenceAssets(const TempStudyWorkspace& workspace) {
    const std::filesystem::path model_root = workspace.scratchpad_root / "models" / "mdeberta";
    std::filesystem::create_directories(model_root / "onnx");
    {
        std::ofstream output(model_root / "spm.model", std::ios::binary);
        output << "fake sentencepiece\n";
    }
    {
        std::ofstream output(model_root / "onnx" / "model.onnx", std::ios::binary);
        output << "fake reference model\n";
    }
}

void WriteCatalog(
    const TempStudyWorkspace& workspace,
    bool allow_coreml,
    bool include_fake_candidate = true) {
    ordered_json entries = ordered_json::array();
    entries.push_back(
        {
            {"name", "reference"},
            {"generator_program", ""},
            {"generator_args_json", ordered_json::array()},
            {"source_artifact_name", nullptr},
            {"output_relpath", "models/mdeberta/onnx/model.onnx"},
            {"calibration_role", nullptr},
            {"validation_role", nullptr},
            {"allowed_backends", ordered_json::array({"cpu", "coreml"})},
            {"notes", "test reference"},
        });

    if (include_fake_candidate) {
        entries.push_back(
            {
                {"name", "fake_candidate"},
                {"generator_program", "python3"},
                {"generator_args_json",
                 ordered_json::array(
                     {
                         FakeGeneratorPath().string(),
                         "--src=${SRC}",
                         "--dest=${DEST}",
                         "--capture=${SCRATCHPAD}/capture.json",
                         "--calibration-tsv=${CALIBRATION_TSVS}",
                         "--validation-tsv=${VALIDATION_TSVS}",
                     })},
                {"source_artifact_name", "reference"},
                {"output_relpath", "candidates/fake_candidate.onnx"},
                {"calibration_role", "calibration"},
                {"validation_role", "fidelity_validation"},
                {"allowed_backends", allow_coreml ? ordered_json::array({"cpu", "coreml"}) : ordered_json::array({"cpu"})},
                {"notes", "test candidate"},
            });
    }

    std::ofstream output(workspace.catalog_path);
    output << entries.dump(2) << "\n";
}

sqlite3* OpenDb(const TempStudyWorkspace& workspace) {
    sqlite3* db = nullptr;
    if (sqlite3_open((workspace.scratchpad_root / "db.sqlite3").string().c_str(), &db) != SQLITE_OK) {
        throw std::runtime_error("Failed to open temp SQLite database");
    }
    return db;
}

sqlite3_int64 QueryCount(sqlite3* db, const std::string& sql) {
    sqlite3_stmt* statement = nullptr;
    if (sqlite3_prepare_v2(db, sql.c_str(), -1, &statement, nullptr) != SQLITE_OK) {
        throw std::runtime_error("Failed to prepare count query");
    }
    const int rc = sqlite3_step(statement);
    if (rc != SQLITE_ROW) {
        sqlite3_finalize(statement);
        throw std::runtime_error("Failed to execute count query");
    }
    const sqlite3_int64 value = sqlite3_column_int64(statement, 0);
    sqlite3_finalize(statement);
    return value;
}

std::string QueryText(sqlite3* db, const std::string& sql) {
    sqlite3_stmt* statement = nullptr;
    if (sqlite3_prepare_v2(db, sql.c_str(), -1, &statement, nullptr) != SQLITE_OK) {
        throw std::runtime_error("Failed to prepare text query");
    }
    const int rc = sqlite3_step(statement);
    if (rc != SQLITE_ROW) {
        sqlite3_finalize(statement);
        throw std::runtime_error("Failed to execute text query");
    }
    const unsigned char* value = sqlite3_column_text(statement, 0);
    const std::string text = value == nullptr ? "" : reinterpret_cast<const char*>(value);
    sqlite3_finalize(statement);
    return text;
}

void ExecSql(sqlite3* db, const std::string& sql) {
    char* error_message = nullptr;
    if (sqlite3_exec(db, sql.c_str(), nullptr, nullptr, &error_message) != SQLITE_OK) {
        std::string message = error_message ? error_message : "unknown SQLite error";
        sqlite3_free(error_message);
        throw std::runtime_error(message);
    }
}

void VerifyStudyInitOptionsUseDefaults() {
    auto parser = MakeStudyInitParser();
    const std::vector<std::string> args;
    const optparse::Values& values = parser.parse_args(args);
    const auto options = nli::FinalizeStudyInitCommandLine(parser, values);
    if (options.scratchpad_root != nli::DefaultStudyScratchpadRoot()) {
        throw std::runtime_error("expected study init scratchpad default");
    }
    if (options.catalog_path != nli::DefaultStudyCatalogPath()) {
        throw std::runtime_error("expected study init catalog default");
    }
    if (options.force) {
        throw std::runtime_error("expected force to default to false");
    }
}

void VerifyStudyRunOptionsRequireQuantizationAndDataset() {
    auto parser = MakeStudyRunParser();
    const std::vector<std::string> args = {"--backend=cpu"};
    const optparse::Values& values = parser.parse_args(args);
    ExpectParserExitCode(
        [&]() {
            (void)nli::FinalizeStudyRunCommandLine(parser, values);
        },
        2);
}

void VerifyStudyInitIsIdempotent() {
    const TempStudyWorkspace workspace = CreateTempStudyWorkspace();
    try {
        WriteFixtureDatasets(workspace);
        WriteReferenceAssets(workspace);
        WriteCatalog(workspace, true);

        const nli::StudyInitCommandLineOptions options{
            workspace.scratchpad_root.string(),
            workspace.catalog_path.string(),
            false,
        };
        std::ostringstream log;
        nli::InitializeStudyWorkspace(options, log);
        nli::InitializeStudyWorkspace(options, log);

        sqlite3* db = OpenDb(workspace);
        try {
            if (QueryCount(db, "SELECT COUNT(*) FROM dataset") != 2) {
                throw std::runtime_error("expected two imported datasets");
            }
            if (QueryCount(db, "SELECT COUNT(*) FROM dataset_row") != 6) {
                throw std::runtime_error("expected six imported dataset rows");
            }
            if (QueryCount(db, "SELECT COUNT(*) FROM quantization") != 2) {
                throw std::runtime_error("expected seeded quantizations");
            }
        } catch (...) {
            sqlite3_close(db);
            throw;
        }
        sqlite3_close(db);
    } catch (...) {
        RemoveTempStudyWorkspace(workspace);
        throw;
    }
    RemoveTempStudyWorkspace(workspace);
}

void VerifyStudyRunMaterializesArtifactAndStoresEvaluations() {
    const TempStudyWorkspace workspace = CreateTempStudyWorkspace();
    try {
        WriteFixtureDatasets(workspace);
        WriteReferenceAssets(workspace);
        WriteCatalog(workspace, false);

        const nli::StudyInitCommandLineOptions init_options{
            workspace.scratchpad_root.string(),
            workspace.catalog_path.string(),
            false,
        };
        std::ostringstream init_log;
        nli::InitializeStudyWorkspace(init_options, init_log);

        const nli::StudyRunCommandLineOptions run_options{
            workspace.scratchpad_root.string(),
            "fake_candidate",
            nli::SessionBackend::kCPU,
            "mnli-train-search-validation-skip64-64-per-label.tsv",
            false,
            false,
        };
        std::ostringstream run_log;
        nli::RunStudyEvaluation(run_options, FakePredictorFactory(), run_log);
        nli::RunStudyEvaluation(run_options, FakePredictorFactory(), run_log);

        sqlite3* db = OpenDb(workspace);
        try {
            if (QueryCount(db, "SELECT COUNT(*) FROM evaluation_run") != 2) {
                throw std::runtime_error("expected reference and candidate evaluation runs");
            }
            if (QueryCount(db, "SELECT COUNT(*) FROM evaluation") != 6) {
                throw std::runtime_error("expected three reference rows and three candidate rows");
            }
        } catch (...) {
            sqlite3_close(db);
            throw;
        }
        sqlite3_close(db);

        const auto capture = ordered_json::parse(
            ReadFile(workspace.scratchpad_root / "capture.json"));
        if (capture.at("src").get<std::string>().find("model.onnx") == std::string::npos) {
            throw std::runtime_error("expected generator src expansion to point at reference model");
        }
        if (capture.at("dest").get<std::string>().find("fake_candidate.onnx") == std::string::npos) {
            throw std::runtime_error("expected generator dest expansion to point at candidate path");
        }
        if (capture.at("calibration_tsvs").size() != 1 || capture.at("validation_tsvs").size() != 1) {
            throw std::runtime_error("expected generator calibration and validation placeholder expansion");
        }
    } catch (...) {
        RemoveTempStudyWorkspace(workspace);
        throw;
    }
    RemoveTempStudyWorkspace(workspace);
}

void VerifyStudyRunResumesMissingRowsAndRegeneratesZeroByteArtifacts() {
    const TempStudyWorkspace workspace = CreateTempStudyWorkspace();
    try {
        WriteFixtureDatasets(workspace);
        WriteReferenceAssets(workspace);
        WriteCatalog(workspace, false);

        const nli::StudyInitCommandLineOptions init_options{
            workspace.scratchpad_root.string(),
            workspace.catalog_path.string(),
            false,
        };
        std::ostringstream init_log;
        nli::InitializeStudyWorkspace(init_options, init_log);

        const nli::StudyRunCommandLineOptions run_options{
            workspace.scratchpad_root.string(),
            "fake_candidate",
            nli::SessionBackend::kCPU,
            "mnli-train-search-validation-skip64-64-per-label.tsv",
            false,
            false,
        };
        std::ostringstream run_log;
        nli::RunStudyEvaluation(run_options, FakePredictorFactory(), run_log);

        sqlite3* db = OpenDb(workspace);
        try {
            ExecSql(
                db,
                "DELETE FROM evaluation "
                "WHERE id IN (SELECT id FROM evaluation ORDER BY id DESC LIMIT 1)");
        } catch (...) {
            sqlite3_close(db);
            throw;
        }
        sqlite3_close(db);

        const auto candidate_path = workspace.scratchpad_root / "candidates" / "fake_candidate.onnx";
        std::ofstream(candidate_path, std::ios::trunc).close();

        nli::RunStudyEvaluation(run_options, FakePredictorFactory(), run_log);

        db = OpenDb(workspace);
        try {
            if (QueryCount(db, "SELECT COUNT(*) FROM evaluation") != 6) {
                throw std::runtime_error("expected rerun to restore the missing evaluation row");
            }
            sqlite3_stmt* statement = nullptr;
            if (sqlite3_prepare_v2(
                    db,
                    "SELECT status FROM artifact WHERE path LIKE '%fake_candidate.onnx'",
                    -1,
                    &statement,
                    nullptr) != SQLITE_OK) {
                throw std::runtime_error("failed to prepare artifact query");
            }
            if (sqlite3_step(statement) != SQLITE_ROW) {
                sqlite3_finalize(statement);
                throw std::runtime_error("expected candidate artifact row");
            }
            const std::string status =
                reinterpret_cast<const char*>(sqlite3_column_text(statement, 0));
            sqlite3_finalize(statement);
            if (status != "materialized") {
                throw std::runtime_error("expected zero-byte artifact to be regenerated");
            }
        } catch (...) {
            sqlite3_close(db);
            throw;
        }
        sqlite3_close(db);
    } catch (...) {
        RemoveTempStudyWorkspace(workspace);
        throw;
    }
    RemoveTempStudyWorkspace(workspace);
}

void VerifyStudyRunRejectsDisallowedBackendBeforeMaterialization() {
    const TempStudyWorkspace workspace = CreateTempStudyWorkspace();
    try {
        WriteFixtureDatasets(workspace);
        WriteReferenceAssets(workspace);
        WriteCatalog(workspace, false);

        const nli::StudyInitCommandLineOptions init_options{
            workspace.scratchpad_root.string(),
            workspace.catalog_path.string(),
            false,
        };
        std::ostringstream init_log;
        nli::InitializeStudyWorkspace(init_options, init_log);

        const nli::StudyRunCommandLineOptions run_options{
            workspace.scratchpad_root.string(),
            "fake_candidate",
            nli::SessionBackend::kCoreML,
            "mnli-train-search-validation-skip64-64-per-label.tsv",
            false,
            false,
        };
        std::ostringstream run_log;
        try {
            nli::RunStudyEvaluation(run_options, FakePredictorFactory(), run_log);
        } catch (const std::runtime_error&) {
            if (std::filesystem::exists(workspace.scratchpad_root / "candidates" / "fake_candidate.onnx")) {
                throw std::runtime_error("candidate artifact should not materialize on disallowed backend");
            }
            return;
        }
        throw std::runtime_error("expected disallowed backend to be rejected");
    } catch (...) {
        RemoveTempStudyWorkspace(workspace);
        throw;
    }
}

void VerifyStudyRunFailsCoreMLFallbackWithoutRecordingRows() {
    const TempStudyWorkspace workspace = CreateTempStudyWorkspace();
    try {
        WriteFixtureDatasets(workspace);
        WriteReferenceAssets(workspace);
        WriteCatalog(workspace, true);

        const nli::StudyInitCommandLineOptions init_options{
            workspace.scratchpad_root.string(),
            workspace.catalog_path.string(),
            false,
        };
        std::ostringstream init_log;
        nli::InitializeStudyWorkspace(init_options, init_log);

        const nli::StudyRunCommandLineOptions run_options{
            workspace.scratchpad_root.string(),
            "reference",
            nli::SessionBackend::kCoreML,
            "mnli-train-search-validation-skip64-64-per-label.tsv",
            false,
            false,
        };
        std::ostringstream run_log;
        bool failed_as_expected = false;
        try {
            nli::RunStudyEvaluation(run_options, CoreMLFallbackPredictorFactory(), run_log);
        } catch (const std::runtime_error&) {
            failed_as_expected = true;
        }
        if (!failed_as_expected) {
            throw std::runtime_error("expected CoreML CPU fallback to fail the study run");
        }

        sqlite3* db = OpenDb(workspace);
        try {
            if (QueryCount(db, "SELECT COUNT(*) FROM evaluation") != 0) {
                throw std::runtime_error("fallback run should not store evaluation rows");
            }
            if (QueryCount(
                    db,
                    "SELECT COUNT(*) FROM evaluation_run WHERE status = 'completed'") != 0) {
                throw std::runtime_error("fallback run should not be marked completed");
            }
            const std::string status =
                QueryText(db, "SELECT status FROM evaluation_run LIMIT 1");
            if (status != "failed") {
                throw std::runtime_error("fallback run should be marked failed");
            }
        } catch (...) {
            sqlite3_close(db);
            throw;
        }
        sqlite3_close(db);
    } catch (...) {
        RemoveTempStudyWorkspace(workspace);
        throw;
    }
    RemoveTempStudyWorkspace(workspace);
}

}  // namespace

int main() {
    VerifyStudyInitOptionsUseDefaults();
    VerifyStudyRunOptionsRequireQuantizationAndDataset();
    VerifyStudyInitIsIdempotent();
    VerifyStudyRunMaterializesArtifactAndStoresEvaluations();
    VerifyStudyRunResumesMissingRowsAndRegeneratesZeroByteArtifacts();
    VerifyStudyRunRejectsDisallowedBackendBeforeMaterialization();
    VerifyStudyRunFailsCoreMLFallbackWithoutRecordingRows();
    return 0;
}
