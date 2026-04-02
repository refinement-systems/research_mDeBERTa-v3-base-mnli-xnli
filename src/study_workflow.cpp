#include "study_workflow.h"

#include "nli_eval.h"
#include "sha256.h"
#include "study_command_line.h"

#include <nlohmann/json.hpp>
#include <sqlite3.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

using ordered_json = nlohmann::ordered_json;

constexpr std::string_view kStatusMaterialized = "materialized";
constexpr std::string_view kStatusMissing = "missing";
constexpr std::string_view kStatusInvalid = "invalid";
constexpr std::string_view kStatusFailed = "failed";
constexpr std::string_view kRunStatusPending = "pending";
constexpr std::string_view kRunStatusRunning = "running";
constexpr std::string_view kRunStatusCompleted = "completed";
constexpr std::string_view kRunStatusFailed = "failed";

struct StudyCatalogEntry {
    std::string name;
    std::string generator_program;
    std::vector<std::string> generator_args;
    std::optional<std::string> source_artifact_name;
    std::string output_relpath;
    std::optional<std::string> calibration_role;
    std::optional<std::string> validation_role;
    std::vector<std::string> allowed_backends;
    std::string notes;
};

struct MaterializationInfo {
    std::string status;
    std::string sha256;
    sqlite3_int64 size_bytes = 0;
    bool exists = false;
};

struct DatasetRowRecord {
    sqlite3_int64 id = 0;
    std::string premise;
    std::string hypothesis;
};

class SqliteStatement {
public:
    SqliteStatement(sqlite3* db, const std::string& sql) : db_(db), statement_(nullptr) {
        const int rc = sqlite3_prepare_v2(db_, sql.c_str(), -1, &statement_, nullptr);
        if (rc != SQLITE_OK) {
            throw std::runtime_error(
                "Failed to prepare SQLite statement: " + std::string(sqlite3_errmsg(db_)));
        }
    }

    ~SqliteStatement() {
        if (statement_ != nullptr) {
            sqlite3_finalize(statement_);
        }
    }

    SqliteStatement(const SqliteStatement&) = delete;
    SqliteStatement& operator=(const SqliteStatement&) = delete;

    void BindText(int index, const std::string& value) {
        const int rc = sqlite3_bind_text(
            statement_, index, value.c_str(), static_cast<int>(value.size()), SQLITE_TRANSIENT);
        if (rc != SQLITE_OK) {
            throw std::runtime_error("Failed to bind SQLite text parameter");
        }
    }

    void BindOptionalText(int index, const std::optional<std::string>& value) {
        if (value) {
            BindText(index, *value);
            return;
        }
        BindNull(index);
    }

    void BindInt64(int index, sqlite3_int64 value) {
        const int rc = sqlite3_bind_int64(statement_, index, value);
        if (rc != SQLITE_OK) {
            throw std::runtime_error("Failed to bind SQLite integer parameter");
        }
    }

    void BindDouble(int index, double value) {
        const int rc = sqlite3_bind_double(statement_, index, value);
        if (rc != SQLITE_OK) {
            throw std::runtime_error("Failed to bind SQLite floating-point parameter");
        }
    }

    void BindNull(int index) {
        const int rc = sqlite3_bind_null(statement_, index);
        if (rc != SQLITE_OK) {
            throw std::runtime_error("Failed to bind SQLite null parameter");
        }
    }

    bool StepRow() {
        const int rc = sqlite3_step(statement_);
        if (rc == SQLITE_ROW) {
            return true;
        }
        if (rc == SQLITE_DONE) {
            return false;
        }
        throw std::runtime_error("SQLite step failed: " + std::string(sqlite3_errmsg(db_)));
    }

    void StepDone() {
        const int rc = sqlite3_step(statement_);
        if (rc != SQLITE_DONE) {
            throw std::runtime_error("SQLite step failed: " + std::string(sqlite3_errmsg(db_)));
        }
    }

    void Reset() {
        sqlite3_reset(statement_);
        sqlite3_clear_bindings(statement_);
    }

    sqlite3_int64 ColumnInt64(int index) const {
        return sqlite3_column_int64(statement_, index);
    }

    double ColumnDouble(int index) const {
        return sqlite3_column_double(statement_, index);
    }

    bool ColumnIsNull(int index) const {
        return sqlite3_column_type(statement_, index) == SQLITE_NULL;
    }

    std::string ColumnText(int index) const {
        const unsigned char* value = sqlite3_column_text(statement_, index);
        if (value == nullptr) {
            return "";
        }
        return reinterpret_cast<const char*>(value);
    }

private:
    sqlite3* db_;
    sqlite3_stmt* statement_;
};

class SqliteDatabase {
public:
    explicit SqliteDatabase(const std::filesystem::path& path) : db_(nullptr) {
        const int rc = sqlite3_open(path.string().c_str(), &db_);
        if (rc != SQLITE_OK) {
            std::string message = db_ ? sqlite3_errmsg(db_) : "unknown error";
            if (db_ != nullptr) {
                sqlite3_close(db_);
            }
            throw std::runtime_error("Failed to open SQLite database: " + message);
        }
        sqlite3_busy_timeout(db_, 5000);
    }

    ~SqliteDatabase() {
        if (db_ != nullptr) {
            sqlite3_close(db_);
        }
    }

    SqliteDatabase(const SqliteDatabase&) = delete;
    SqliteDatabase& operator=(const SqliteDatabase&) = delete;

    sqlite3* handle() const {
        return db_;
    }

    void Exec(const std::string& sql) {
        char* error_message = nullptr;
        const int rc = sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &error_message);
        if (rc != SQLITE_OK) {
            std::string message = error_message ? error_message : sqlite3_errmsg(db_);
            sqlite3_free(error_message);
            throw std::runtime_error("SQLite exec failed: " + message);
        }
    }

    sqlite3_int64 LastInsertRowId() const {
        return sqlite3_last_insert_rowid(db_);
    }

private:
    sqlite3* db_;
};

class OnnxStudyPredictor final : public nli::StudyPredictor {
public:
    OnnxStudyPredictor(
        const std::string& model_path,
        const std::string& sentencepiece_path,
        nli::SessionBackend backend,
        std::ostream& log)
        : model_(model_path, sentencepiece_path, backend, log) {}

    nli::SessionBackend ActualBackend() const override {
        return model_.backend();
    }

    bool UsedFallback() const override {
        return model_.used_fallback();
    }

    nli::NliLogits PredictLogits(const std::string& premise, const std::string& hypothesis) override {
        return model_.PredictLogits(premise, hypothesis);
    }

private:
    nli::DebertaNliModel model_;
};

std::filesystem::path RepoRootPath() {
#if defined(NLI_SOURCE_DIR)
    return std::filesystem::path(NLI_SOURCE_DIR);
#else
    return std::filesystem::current_path();
#endif
}

std::string CurrentTimestamp() {
    const std::time_t now = std::time(nullptr);
    std::tm tm = *std::gmtime(&now);
    std::ostringstream timestamp;
    timestamp << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return timestamp.str();
}

std::filesystem::path ResolveAbsolutePath(const std::string& value) {
    return std::filesystem::absolute(std::filesystem::path(value)).lexically_normal();
}

void EnsureDirectoryLayout(const std::filesystem::path& scratchpad_root) {
    for (const auto& path : std::initializer_list<std::filesystem::path>{
             scratchpad_root,
             scratchpad_root / "models",
             scratchpad_root / "datasets",
             scratchpad_root / "candidates",
             scratchpad_root / "logs",
             scratchpad_root / "logs" / "generation",
             scratchpad_root / "logs" / "evaluation",
             scratchpad_root / "reports",
         }) {
        std::filesystem::create_directories(path);
    }
}

void RemoveDatabaseFiles(const std::filesystem::path& db_path) {
    std::filesystem::remove(db_path);
    std::filesystem::remove(db_path.string() + "-wal");
    std::filesystem::remove(db_path.string() + "-shm");
}

std::string ClassifyDatasetRole(const std::string& dataset_name) {
    if (dataset_name == "hf-probe-set.tsv" || dataset_name == "hf-core-probe.tsv") {
        return "smoke";
    }
    if (dataset_name.find("calibration") != std::string::npos) {
        return "calibration";
    }
    if (dataset_name.find("search-validation") != std::string::npos) {
        return "fidelity_validation";
    }
    if (dataset_name.find("validation_matched") != std::string::npos ||
        dataset_name.find("validation_mismatched") != std::string::npos ||
        dataset_name.find("-test-") != std::string::npos) {
        return "fidelity_test";
    }
    throw std::runtime_error("Could not infer dataset role from filename: " + dataset_name);
}

std::vector<std::filesystem::path> DiscoverDatasetFiles(const std::filesystem::path& dataset_root) {
    std::vector<std::filesystem::path> paths;
    if (!std::filesystem::exists(dataset_root)) {
        return paths;
    }
    for (const auto& entry : std::filesystem::directory_iterator(dataset_root)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        if (entry.path().extension() == ".tsv") {
            paths.push_back(entry.path());
        }
    }
    std::sort(paths.begin(), paths.end());
    return paths;
}

std::vector<StudyCatalogEntry> LoadStudyCatalog(const std::filesystem::path& catalog_path) {
    std::ifstream input(catalog_path);
    if (!input) {
        throw std::runtime_error("Failed to open study catalog: " + catalog_path.string());
    }

    ordered_json root;
    input >> root;
    if (!root.is_array()) {
        throw std::runtime_error("Study catalog must contain a JSON array");
    }

    std::vector<StudyCatalogEntry> entries;
    std::set<std::string> seen_names;
    for (const auto& item : root) {
        const std::string name = item.at("name").get<std::string>();
        if (!seen_names.insert(name).second) {
            throw std::runtime_error("Study catalog contains duplicate quantization name: " + name);
        }
        entries.push_back(StudyCatalogEntry{
            name,
            item.at("generator_program").get<std::string>(),
            item.at("generator_args_json").get<std::vector<std::string>>(),
            item.at("source_artifact_name").is_null()
                ? std::nullopt
                : std::optional<std::string>(item.at("source_artifact_name").get<std::string>()),
            item.at("output_relpath").get<std::string>(),
            item.at("calibration_role").is_null()
                ? std::nullopt
                : std::optional<std::string>(item.at("calibration_role").get<std::string>()),
            item.at("validation_role").is_null()
                ? std::nullopt
                : std::optional<std::string>(item.at("validation_role").get<std::string>()),
            item.at("allowed_backends").get<std::vector<std::string>>(),
            item.at("notes").get<std::string>(),
        });
    }
    return entries;
}

MaterializationInfo InspectArtifact(
    const std::filesystem::path& artifact_path,
    const std::optional<std::string>& expected_sha256) {
    if (!std::filesystem::exists(artifact_path)) {
        return MaterializationInfo{std::string(kStatusMissing), "", 0, false};
    }

    const auto size = static_cast<sqlite3_int64>(std::filesystem::file_size(artifact_path));
    if (size == 0) {
        return MaterializationInfo{std::string(kStatusInvalid), "", 0, false};
    }

    const std::string sha256 = nli::ComputeFileSha256Hex(artifact_path);
    if (expected_sha256 && !expected_sha256->empty() && *expected_sha256 != sha256) {
        return MaterializationInfo{std::string(kStatusInvalid), sha256, size, true};
    }
    return MaterializationInfo{std::string(kStatusMaterialized), sha256, size, true};
}

std::string ResolvePathArgument(const std::string& argument) {
    if (argument.empty() || argument.front() == '-') {
        return argument;
    }
    const std::filesystem::path path(argument);
    if (path.is_absolute()) {
        return path.string();
    }
    const std::filesystem::path candidate = RepoRootPath() / path;
    if (std::filesystem::exists(candidate)) {
        return candidate.lexically_normal().string();
    }
    return argument;
}

std::string ResolveProgramArgument(const std::string& argument) {
    if (argument == "python3") {
        const std::filesystem::path venv_python = RepoRootPath() / ".venv" / "bin" / "python";
        if (std::filesystem::exists(venv_python)) {
            return venv_python.lexically_normal().string();
        }
    }
    return ResolvePathArgument(argument);
}

int RunProcess(
    const std::string& program,
    const std::vector<std::string>& arguments,
    const std::filesystem::path& stdout_path,
    const std::filesystem::path& stderr_path) {
    std::filesystem::create_directories(stdout_path.parent_path());
    std::filesystem::create_directories(stderr_path.parent_path());

    const int stdout_fd = ::open(stdout_path.string().c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (stdout_fd < 0) {
        throw std::runtime_error("Failed to open generation stdout log: " + stdout_path.string());
    }
    const int stderr_fd = ::open(stderr_path.string().c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (stderr_fd < 0) {
        ::close(stdout_fd);
        throw std::runtime_error("Failed to open generation stderr log: " + stderr_path.string());
    }

    pid_t child = fork();
    if (child < 0) {
        ::close(stdout_fd);
        ::close(stderr_fd);
        throw std::runtime_error("fork() failed while launching generator");
    }
    if (child == 0) {
        ::dup2(stdout_fd, STDOUT_FILENO);
        ::dup2(stderr_fd, STDERR_FILENO);
        ::close(stdout_fd);
        ::close(stderr_fd);

        std::vector<char*> argv;
        argv.reserve(arguments.size() + 2);
        argv.push_back(const_cast<char*>(program.c_str()));
        for (const std::string& argument : arguments) {
            argv.push_back(const_cast<char*>(argument.c_str()));
        }
        argv.push_back(nullptr);
        ::execvp(program.c_str(), argv.data());
        std::perror("execvp");
        _exit(127);
    }

    ::close(stdout_fd);
    ::close(stderr_fd);

    int status = 0;
    if (::waitpid(child, &status, 0) < 0) {
        throw std::runtime_error("waitpid() failed while waiting for generator");
    }
    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    if (WIFSIGNALED(status)) {
        return 128 + WTERMSIG(status);
    }
    return 1;
}

void CreateSchema(SqliteDatabase& db) {
    db.Exec("PRAGMA journal_mode=WAL;");
    db.Exec("PRAGMA foreign_keys=ON;");
    db.Exec(
        "CREATE TABLE IF NOT EXISTS dataset ("
        "  id INTEGER PRIMARY KEY,"
        "  name TEXT NOT NULL UNIQUE,"
        "  role TEXT NOT NULL,"
        "  source_path TEXT NOT NULL,"
        "  source_sha256 TEXT NOT NULL"
        ");"
        "CREATE TABLE IF NOT EXISTS dataset_row ("
        "  id INTEGER PRIMARY KEY,"
        "  dataset_id INTEGER NOT NULL,"
        "  row_idx INTEGER NOT NULL,"
        "  source_row_id TEXT,"
        "  label TEXT,"
        "  premise TEXT NOT NULL,"
        "  hypothesis TEXT NOT NULL,"
        "  FOREIGN KEY(dataset_id) REFERENCES dataset(id) ON DELETE CASCADE,"
        "  UNIQUE(dataset_id, row_idx)"
        ");"
        "CREATE TABLE IF NOT EXISTS quantization ("
        "  id INTEGER PRIMARY KEY,"
        "  name TEXT NOT NULL UNIQUE,"
        "  generator_program TEXT NOT NULL,"
        "  generator_args_json TEXT NOT NULL,"
        "  source_artifact_name TEXT,"
        "  output_relpath TEXT NOT NULL,"
        "  calibration_role TEXT,"
        "  validation_role TEXT,"
        "  allowed_backends_json TEXT NOT NULL,"
        "  notes TEXT NOT NULL"
        ");"
        "CREATE TABLE IF NOT EXISTS artifact ("
        "  id INTEGER PRIMARY KEY,"
        "  quantization_id INTEGER NOT NULL UNIQUE,"
        "  path TEXT NOT NULL,"
        "  artifact_sha256 TEXT,"
        "  size_bytes INTEGER NOT NULL,"
        "  status TEXT NOT NULL,"
        "  stdout_log_path TEXT NOT NULL,"
        "  stderr_log_path TEXT NOT NULL,"
        "  materialized_at TEXT,"
        "  FOREIGN KEY(quantization_id) REFERENCES quantization(id) ON DELETE CASCADE"
        ");"
        "CREATE TABLE IF NOT EXISTS backend ("
        "  id INTEGER PRIMARY KEY,"
        "  name TEXT NOT NULL UNIQUE"
        ");"
        "CREATE TABLE IF NOT EXISTS evaluation_run ("
        "  id INTEGER PRIMARY KEY,"
        "  artifact_id INTEGER NOT NULL,"
        "  backend_id INTEGER NOT NULL,"
        "  dataset_id INTEGER NOT NULL,"
        "  command_json TEXT NOT NULL,"
        "  status TEXT NOT NULL,"
        "  started_at TEXT,"
        "  finished_at TEXT,"
        "  FOREIGN KEY(artifact_id) REFERENCES artifact(id) ON DELETE CASCADE,"
        "  FOREIGN KEY(backend_id) REFERENCES backend(id),"
        "  FOREIGN KEY(dataset_id) REFERENCES dataset(id) ON DELETE CASCADE,"
        "  UNIQUE(artifact_id, backend_id, dataset_id)"
        ");"
        "CREATE TABLE IF NOT EXISTS evaluation ("
        "  id INTEGER PRIMARY KEY,"
        "  evaluation_run_id INTEGER NOT NULL,"
        "  dataset_row_id INTEGER NOT NULL,"
        "  entailment_logit REAL NOT NULL,"
        "  neutral_logit REAL NOT NULL,"
        "  contradiction_logit REAL NOT NULL,"
        "  predicted_label TEXT NOT NULL,"
        "  FOREIGN KEY(evaluation_run_id) REFERENCES evaluation_run(id) ON DELETE CASCADE,"
        "  FOREIGN KEY(dataset_row_id) REFERENCES dataset_row(id) ON DELETE CASCADE,"
        "  UNIQUE(evaluation_run_id, dataset_row_id)"
        ");");
}

void SeedBackends(SqliteDatabase& db) {
    SqliteStatement statement(
        db.handle(),
        "INSERT INTO backend (name) VALUES (?) "
        "ON CONFLICT(name) DO NOTHING");
    for (const std::string& backend_name : {"CPU", "CoreML"}) {
        statement.BindText(1, backend_name);
        statement.StepDone();
        statement.Reset();
    }
}

std::optional<sqlite3_int64> LookupSingleId(
    sqlite3* db,
    const std::string& sql,
    const std::string& value) {
    SqliteStatement statement(db, sql);
    statement.BindText(1, value);
    if (!statement.StepRow()) {
        return std::nullopt;
    }
    return statement.ColumnInt64(0);
}

sqlite3_int64 RequireSingleId(
    sqlite3* db,
    const std::string& sql,
    const std::string& value,
    const std::string& context) {
    const auto id = LookupSingleId(db, sql, value);
    if (!id) {
        throw std::runtime_error(context + ": " + value);
    }
    return *id;
}

void UpsertCatalogEntries(
    SqliteDatabase& db,
    const std::vector<StudyCatalogEntry>& entries,
    const std::filesystem::path& scratchpad_root) {
    SqliteStatement quantization_statement(
        db.handle(),
        "INSERT INTO quantization "
        "    (name, generator_program, generator_args_json, source_artifact_name, output_relpath, "
        "     calibration_role, validation_role, allowed_backends_json, notes) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(name) DO UPDATE SET "
        "    generator_program=excluded.generator_program, "
        "    generator_args_json=excluded.generator_args_json, "
        "    source_artifact_name=excluded.source_artifact_name, "
        "    output_relpath=excluded.output_relpath, "
        "    calibration_role=excluded.calibration_role, "
        "    validation_role=excluded.validation_role, "
        "    allowed_backends_json=excluded.allowed_backends_json, "
        "    notes=excluded.notes");

    SqliteStatement artifact_statement(
        db.handle(),
        "INSERT INTO artifact "
        "    (quantization_id, path, artifact_sha256, size_bytes, status, stdout_log_path, stderr_log_path, materialized_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(quantization_id) DO UPDATE SET "
        "    path=excluded.path, "
        "    artifact_sha256=excluded.artifact_sha256, "
        "    size_bytes=excluded.size_bytes, "
        "    status=excluded.status, "
        "    stdout_log_path=excluded.stdout_log_path, "
        "    stderr_log_path=excluded.stderr_log_path, "
        "    materialized_at=excluded.materialized_at");

    for (const StudyCatalogEntry& entry : entries) {
        quantization_statement.BindText(1, entry.name);
        quantization_statement.BindText(2, entry.generator_program);
        quantization_statement.BindText(3, ordered_json(entry.generator_args).dump());
        quantization_statement.BindOptionalText(4, entry.source_artifact_name);
        quantization_statement.BindText(5, entry.output_relpath);
        quantization_statement.BindOptionalText(6, entry.calibration_role);
        quantization_statement.BindOptionalText(7, entry.validation_role);
        quantization_statement.BindText(8, ordered_json(entry.allowed_backends).dump());
        quantization_statement.BindText(9, entry.notes);
        quantization_statement.StepDone();
        quantization_statement.Reset();

        const sqlite3_int64 quantization_id = RequireSingleId(
            db.handle(),
            "SELECT id FROM quantization WHERE name = ?",
            entry.name,
            "Unknown quantization after upsert");

        const std::filesystem::path artifact_path =
            (scratchpad_root / entry.output_relpath).lexically_normal();
        const auto artifact_state = InspectArtifact(artifact_path, std::nullopt);
        const std::optional<std::string> artifact_sha =
            artifact_state.sha256.empty() ? std::nullopt : std::optional<std::string>(artifact_state.sha256);
        const std::optional<std::string> materialized_at =
            artifact_state.status == kStatusMaterialized ? std::optional<std::string>(CurrentTimestamp()) : std::nullopt;

        artifact_statement.BindInt64(1, quantization_id);
        artifact_statement.BindText(2, artifact_path.string());
        artifact_statement.BindOptionalText(3, artifact_sha);
        artifact_statement.BindInt64(4, artifact_state.size_bytes);
        artifact_statement.BindText(5, artifact_state.status);
        artifact_statement.BindText(
            6, (scratchpad_root / "logs" / "generation" / (entry.name + ".stdout.log")).string());
        artifact_statement.BindText(
            7, (scratchpad_root / "logs" / "generation" / (entry.name + ".stderr.log")).string());
        artifact_statement.BindOptionalText(8, materialized_at);
        artifact_statement.StepDone();
        artifact_statement.Reset();
    }
}

void ImportDatasets(SqliteDatabase& db, const std::filesystem::path& dataset_root) {
    SqliteStatement lookup_dataset(
        db.handle(),
        "SELECT id, source_sha256 FROM dataset WHERE name = ?");
    SqliteStatement insert_dataset(
        db.handle(),
        "INSERT INTO dataset (name, role, source_path, source_sha256) VALUES (?, ?, ?, ?)");
    SqliteStatement count_rows(
        db.handle(),
        "SELECT COUNT(*) FROM dataset_row WHERE dataset_id = ?");
    SqliteStatement delete_rows(
        db.handle(),
        "DELETE FROM dataset_row WHERE dataset_id = ?");
    SqliteStatement insert_row(
        db.handle(),
        "INSERT INTO dataset_row (dataset_id, row_idx, source_row_id, label, premise, hypothesis) "
        "VALUES (?, ?, ?, ?, ?, ?)");

    for (const std::filesystem::path& path : DiscoverDatasetFiles(dataset_root)) {
        const std::string dataset_name = path.filename().string();
        const std::string role = ClassifyDatasetRole(dataset_name);
        const std::string source_sha256 = nli::ComputeFileSha256Hex(path);
        const auto examples = nli::ReadNliEvalExamples(path.string());

        lookup_dataset.BindText(1, dataset_name);
        sqlite3_int64 dataset_id = 0;
        if (lookup_dataset.StepRow()) {
            dataset_id = lookup_dataset.ColumnInt64(0);
            const std::string existing_sha256 = lookup_dataset.ColumnText(1);
            lookup_dataset.Reset();
            if (existing_sha256 != source_sha256) {
                throw std::runtime_error(
                    "Dataset name collision with different content: " + dataset_name);
            }

            count_rows.BindInt64(1, dataset_id);
            const bool has_count = count_rows.StepRow();
            const sqlite3_int64 row_count = has_count ? count_rows.ColumnInt64(0) : 0;
            count_rows.Reset();
            if (row_count == static_cast<sqlite3_int64>(examples.size())) {
                continue;
            }

            delete_rows.BindInt64(1, dataset_id);
            delete_rows.StepDone();
            delete_rows.Reset();
        } else {
            lookup_dataset.Reset();
            insert_dataset.BindText(1, dataset_name);
            insert_dataset.BindText(2, role);
            insert_dataset.BindText(3, path.string());
            insert_dataset.BindText(4, source_sha256);
            insert_dataset.StepDone();
            insert_dataset.Reset();
            dataset_id = db.LastInsertRowId();
        }

        for (size_t row_index = 0; row_index < examples.size(); ++row_index) {
            const auto& example = examples[row_index];
            insert_row.BindInt64(1, dataset_id);
            insert_row.BindInt64(2, static_cast<sqlite3_int64>(row_index));
            insert_row.BindText(3, example.id);
            insert_row.BindOptionalText(4, example.label);
            insert_row.BindText(5, example.premise);
            insert_row.BindText(6, example.hypothesis);
            insert_row.StepDone();
            insert_row.Reset();
        }
    }
}

struct QuantizationRecord {
    sqlite3_int64 quantization_id = 0;
    sqlite3_int64 artifact_id = 0;
    std::string name;
    std::string generator_program;
    std::vector<std::string> generator_args;
    std::optional<std::string> source_artifact_name;
    std::optional<std::string> calibration_role;
    std::optional<std::string> validation_role;
    std::vector<std::string> allowed_backends;
    std::filesystem::path artifact_path;
    std::optional<std::string> artifact_sha256;
    std::string artifact_status;
    std::filesystem::path stdout_log_path;
    std::filesystem::path stderr_log_path;
};

QuantizationRecord LoadQuantizationRecord(sqlite3* db, const std::string& quantization_name) {
    SqliteStatement statement(
        db,
        "SELECT "
        "  q.id, q.name, q.generator_program, q.generator_args_json, q.source_artifact_name, "
        "  q.calibration_role, q.validation_role, q.allowed_backends_json, "
        "  a.id, a.path, a.artifact_sha256, a.status, a.stdout_log_path, a.stderr_log_path "
        "FROM quantization q "
        "JOIN artifact a ON a.quantization_id = q.id "
        "WHERE q.name = ?");
    statement.BindText(1, quantization_name);
    if (!statement.StepRow()) {
        throw std::runtime_error("Unknown quantization: " + quantization_name);
    }

    return QuantizationRecord{
        statement.ColumnInt64(0),
        statement.ColumnInt64(8),
        statement.ColumnText(1),
        statement.ColumnText(2),
        ordered_json::parse(statement.ColumnText(3)).get<std::vector<std::string>>(),
        statement.ColumnIsNull(4) ? std::nullopt : std::optional<std::string>(statement.ColumnText(4)),
        statement.ColumnIsNull(5) ? std::nullopt : std::optional<std::string>(statement.ColumnText(5)),
        statement.ColumnIsNull(6) ? std::nullopt : std::optional<std::string>(statement.ColumnText(6)),
        ordered_json::parse(statement.ColumnText(7)).get<std::vector<std::string>>(),
        std::filesystem::path(statement.ColumnText(9)),
        statement.ColumnIsNull(10) ? std::nullopt : std::optional<std::string>(statement.ColumnText(10)),
        statement.ColumnText(11),
        std::filesystem::path(statement.ColumnText(12)),
        std::filesystem::path(statement.ColumnText(13)),
    };
}

std::vector<std::string> DatasetPathsForRole(sqlite3* db, const std::string& role) {
    SqliteStatement statement(
        db,
        "SELECT source_path FROM dataset WHERE role = ? ORDER BY name");
    statement.BindText(1, role);
    std::vector<std::string> paths;
    while (statement.StepRow()) {
        paths.push_back(statement.ColumnText(0));
    }
    return paths;
}

std::vector<std::string> ExpandArgumentToken(
    const std::string& token,
    const std::filesystem::path& scratchpad_root,
    const std::string& source_path,
    const std::string& dest_path,
    const std::vector<std::string>& calibration_paths,
    const std::vector<std::string>& validation_paths) {
    constexpr std::string_view kSrcMarker = "${SRC}";
    constexpr std::string_view kDestMarker = "${DEST}";
    constexpr std::string_view kScratchpadMarker = "${SCRATCHPAD}";
    constexpr std::string_view kCalibrationMarker = "${CALIBRATION_TSVS}";
    constexpr std::string_view kValidationMarker = "${VALIDATION_TSVS}";

    auto replace_all = [](std::string value, std::string_view needle, const std::string& replacement) {
        size_t pos = 0;
        while ((pos = value.find(needle.data(), pos, needle.size())) != std::string::npos) {
            value.replace(pos, needle.size(), replacement);
            pos += replacement.size();
        }
        return value;
    };

    std::vector<std::string> expanded = {replace_all(
        replace_all(
            replace_all(token, kSrcMarker, source_path),
            kDestMarker, dest_path),
        kScratchpadMarker, scratchpad_root.string())};

    auto expand_multi = [&](std::string_view marker, const std::vector<std::string>& values) {
        if (token.find(marker.data(), 0, marker.size()) == std::string::npos) {
            return expanded;
        }
        if (values.empty()) {
            throw std::runtime_error("Generator placeholder expands to an empty dataset list: " + token);
        }
        std::vector<std::string> multi;
        multi.reserve(values.size());
        for (const std::string& value : values) {
            multi.push_back(replace_all(expanded.front(), marker, value));
        }
        return multi;
    };

    if (token.find(kCalibrationMarker.data(), 0, kCalibrationMarker.size()) != std::string::npos) {
        return expand_multi(kCalibrationMarker, calibration_paths);
    }
    if (token.find(kValidationMarker.data(), 0, kValidationMarker.size()) != std::string::npos) {
        return expand_multi(kValidationMarker, validation_paths);
    }
    return expanded;
}

void UpdateArtifactState(
    sqlite3* db,
    sqlite3_int64 artifact_id,
    const MaterializationInfo& info,
    const std::filesystem::path& stdout_log_path,
    const std::filesystem::path& stderr_log_path) {
    SqliteStatement statement(
        db,
        "UPDATE artifact "
        "SET artifact_sha256 = ?, size_bytes = ?, status = ?, stdout_log_path = ?, stderr_log_path = ?, materialized_at = ? "
        "WHERE id = ?");
    if (info.sha256.empty()) {
        statement.BindNull(1);
    } else {
        statement.BindText(1, info.sha256);
    }
    statement.BindInt64(2, info.size_bytes);
    statement.BindText(3, info.status);
    statement.BindText(4, stdout_log_path.string());
    statement.BindText(5, stderr_log_path.string());
    if (info.status == kStatusMaterialized) {
        statement.BindText(6, CurrentTimestamp());
    } else {
        statement.BindNull(6);
    }
    statement.BindInt64(7, artifact_id);
    statement.StepDone();
}

std::string BackendOptionName(nli::SessionBackend backend) {
    return nli::SessionBackendOptionName(backend);
}

void EnsureBackendAllowed(const QuantizationRecord& quantization, nli::SessionBackend backend) {
    const std::string backend_name = BackendOptionName(backend);
    if (std::find(
            quantization.allowed_backends.begin(),
            quantization.allowed_backends.end(),
            backend_name) == quantization.allowed_backends.end()) {
        throw std::runtime_error(
            "Quantization '" + quantization.name + "' is not allowed on backend '" + backend_name + "'");
    }
}

std::string EnsureArtifactMaterialized(
    sqlite3* db,
    const std::filesystem::path& scratchpad_root,
    const QuantizationRecord& quantization,
    bool force_regenerate,
    std::ostream& log,
    std::set<std::string>& in_progress) {
    if (!in_progress.insert(quantization.name).second) {
        throw std::runtime_error("Detected quantization generation cycle at: " + quantization.name);
    }

    auto cleanup = [&]() { in_progress.erase(quantization.name); };
    try {
        const MaterializationInfo current_state =
            InspectArtifact(quantization.artifact_path, quantization.artifact_sha256);
        if (!force_regenerate && current_state.status == kStatusMaterialized) {
            UpdateArtifactState(db, quantization.artifact_id, current_state, quantization.stdout_log_path, quantization.stderr_log_path);
            cleanup();
            return quantization.artifact_path.string();
        }

        std::string source_path;
        if (quantization.source_artifact_name) {
            const auto source_quantization = LoadQuantizationRecord(db, *quantization.source_artifact_name);
            source_path = EnsureArtifactMaterialized(
                db,
                scratchpad_root,
                source_quantization,
                false,
                log,
                in_progress);
        }

        if (quantization.generator_program.empty()) {
            MaterializationInfo failure = current_state;
            failure.status = current_state.status == kStatusInvalid ? std::string(kStatusInvalid) : std::string(kStatusMissing);
            UpdateArtifactState(db, quantization.artifact_id, failure, quantization.stdout_log_path, quantization.stderr_log_path);
            throw std::runtime_error(
                "Artifact is unavailable and has no generator command: " + quantization.name);
        }

        const std::vector<std::string> calibration_paths =
            quantization.calibration_role ? DatasetPathsForRole(db, *quantization.calibration_role) : std::vector<std::string>{};
        const std::vector<std::string> validation_paths =
            quantization.validation_role ? DatasetPathsForRole(db, *quantization.validation_role) : std::vector<std::string>{};

        std::vector<std::string> expanded_args;
        for (const std::string& token : quantization.generator_args) {
            for (const std::string& expanded_token : ExpandArgumentToken(
                     token,
                     scratchpad_root,
                     source_path,
                     quantization.artifact_path.string(),
                     calibration_paths,
                     validation_paths)) {
                expanded_args.push_back(ResolvePathArgument(expanded_token));
            }
        }

        const std::string program = ResolveProgramArgument(quantization.generator_program);
        log << "Generating artifact for " << quantization.name << "\n";
        log << "  program=" << program << "\n";
        const int exit_code = RunProcess(
            program,
            expanded_args,
            quantization.stdout_log_path,
            quantization.stderr_log_path);
        if (exit_code != 0) {
            MaterializationInfo failure;
            failure.status = std::string(kStatusFailed);
            UpdateArtifactState(db, quantization.artifact_id, failure, quantization.stdout_log_path, quantization.stderr_log_path);
            throw std::runtime_error(
                "Generator command failed for " + quantization.name + " with exit code " +
                std::to_string(exit_code));
        }

        const MaterializationInfo updated_state =
            InspectArtifact(quantization.artifact_path, std::nullopt);
        UpdateArtifactState(db, quantization.artifact_id, updated_state, quantization.stdout_log_path, quantization.stderr_log_path);
        if (updated_state.status != kStatusMaterialized) {
            throw std::runtime_error(
                "Generator did not produce a valid artifact for " + quantization.name);
        }

        cleanup();
        return quantization.artifact_path.string();
    } catch (...) {
        cleanup();
        throw;
    }
}

sqlite3_int64 RequireBackendId(sqlite3* db, nli::SessionBackend backend) {
    const std::string backend_name =
        backend == nli::SessionBackend::kCPU ? "CPU" : "CoreML";
    return RequireSingleId(
        db,
        "SELECT id FROM backend WHERE name = ?",
        backend_name,
        "Unknown backend");
}

sqlite3_int64 RequireDatasetId(sqlite3* db, const std::string& dataset_name) {
    return RequireSingleId(
        db,
        "SELECT id FROM dataset WHERE name = ?",
        dataset_name,
        "Unknown dataset");
}

sqlite3_int64 EnsureEvaluationRun(
    sqlite3* db,
    sqlite3_int64 artifact_id,
    sqlite3_int64 backend_id,
    sqlite3_int64 dataset_id,
    const ordered_json& command_json,
    bool force_rerun) {
    SqliteStatement lookup(
        db,
        "SELECT id FROM evaluation_run WHERE artifact_id = ? AND backend_id = ? AND dataset_id = ?");
    lookup.BindInt64(1, artifact_id);
    lookup.BindInt64(2, backend_id);
    lookup.BindInt64(3, dataset_id);

    sqlite3_int64 run_id = 0;
    if (lookup.StepRow()) {
        run_id = lookup.ColumnInt64(0);
        lookup.Reset();
    } else {
        lookup.Reset();
        SqliteStatement insert(
            db,
            "INSERT INTO evaluation_run "
            "    (artifact_id, backend_id, dataset_id, command_json, status, started_at, finished_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)");
        insert.BindInt64(1, artifact_id);
        insert.BindInt64(2, backend_id);
        insert.BindInt64(3, dataset_id);
        insert.BindText(4, command_json.dump());
        insert.BindText(5, std::string(kRunStatusPending));
        insert.BindNull(6);
        insert.BindNull(7);
        insert.StepDone();
        run_id = sqlite3_last_insert_rowid(db);
    }

    if (force_rerun) {
        SqliteStatement delete_rows(
            db,
            "DELETE FROM evaluation WHERE evaluation_run_id = ?");
        delete_rows.BindInt64(1, run_id);
        delete_rows.StepDone();
    }

    SqliteStatement update(
        db,
        "UPDATE evaluation_run SET command_json = ?, status = ?, started_at = ?, finished_at = NULL "
        "WHERE id = ?");
    update.BindText(1, command_json.dump());
    update.BindText(2, std::string(kRunStatusRunning));
    update.BindText(3, CurrentTimestamp());
    update.BindInt64(4, run_id);
    update.StepDone();

    return run_id;
}

void MarkEvaluationRunStatus(sqlite3* db, sqlite3_int64 run_id, std::string_view status) {
    SqliteStatement update(
        db,
        "UPDATE evaluation_run SET status = ?, finished_at = ? WHERE id = ?");
    update.BindText(1, std::string(status));
    update.BindText(2, CurrentTimestamp());
    update.BindInt64(3, run_id);
    update.StepDone();
}

std::vector<DatasetRowRecord> MissingDatasetRows(
    sqlite3* db,
    sqlite3_int64 dataset_id,
    sqlite3_int64 evaluation_run_id) {
    SqliteStatement statement(
        db,
        "SELECT dr.id, dr.premise, dr.hypothesis "
        "FROM dataset_row dr "
        "LEFT JOIN evaluation e "
        "    ON e.dataset_row_id = dr.id AND e.evaluation_run_id = ? "
        "WHERE dr.dataset_id = ? AND e.id IS NULL "
        "ORDER BY dr.row_idx");
    statement.BindInt64(1, evaluation_run_id);
    statement.BindInt64(2, dataset_id);

    std::vector<DatasetRowRecord> rows;
    while (statement.StepRow()) {
        rows.push_back(DatasetRowRecord{
            statement.ColumnInt64(0),
            statement.ColumnText(1),
            statement.ColumnText(2),
        });
    }
    return rows;
}

void InsertEvaluationRow(
    sqlite3* db,
    sqlite3_int64 evaluation_run_id,
    sqlite3_int64 dataset_row_id,
    const nli::NliLogits& logits) {
    SqliteStatement statement(
        db,
        "INSERT OR IGNORE INTO evaluation "
        "    (evaluation_run_id, dataset_row_id, entailment_logit, neutral_logit, contradiction_logit, predicted_label) "
        "VALUES (?, ?, ?, ?, ?, ?)");
    statement.BindInt64(1, evaluation_run_id);
    statement.BindInt64(2, dataset_row_id);
    statement.BindDouble(3, logits.entailment);
    statement.BindDouble(4, logits.neutral);
    statement.BindDouble(5, logits.contradiction);
    statement.BindText(6, std::string(nli::PredictedLabel(logits)));
    statement.StepDone();
}

std::filesystem::path SentencePiecePathForScratchpad(const std::filesystem::path& scratchpad_root) {
    return scratchpad_root / "models" / "mdeberta" / "spm.model";
}

void RunStudyEvaluationInternal(
    const nli::StudyRunCommandLineOptions& options,
    const nli::StudyPredictorFactory& predictor_factory,
    std::ostream& log,
    bool allow_reference_recursion) {
    const std::filesystem::path scratchpad_root = ResolveAbsolutePath(options.scratchpad_root);
    const std::filesystem::path db_path = scratchpad_root / "db.sqlite3";
    if (!std::filesystem::exists(db_path)) {
        throw std::runtime_error("Study database not found: " + db_path.string());
    }

    SqliteDatabase db(db_path);
    const QuantizationRecord quantization = LoadQuantizationRecord(db.handle(), options.quantization_name);
    EnsureBackendAllowed(quantization, options.backend);

    if (allow_reference_recursion && quantization.name != "reference") {
        nli::StudyRunCommandLineOptions reference_options = options;
        reference_options.quantization_name = "reference";
        reference_options.force_regenerate = false;
        reference_options.force_rerun = false;
        RunStudyEvaluationInternal(reference_options, predictor_factory, log, false);
    }

    sqlite3_int64 run_id = 0;
    try {
        std::set<std::string> in_progress;
        const std::string artifact_path = EnsureArtifactMaterialized(
            db.handle(),
            scratchpad_root,
            quantization,
            options.force_regenerate,
            log,
            in_progress);

        const sqlite3_int64 dataset_id = RequireDatasetId(db.handle(), options.dataset_name);
        const sqlite3_int64 backend_id = RequireBackendId(db.handle(), options.backend);
        const ordered_json command_json = {
            {"quantization", options.quantization_name},
            {"backend", BackendOptionName(options.backend)},
            {"dataset", options.dataset_name},
            {"artifact_path", artifact_path},
        };

        run_id = EnsureEvaluationRun(
            db.handle(),
            quantization.artifact_id,
            backend_id,
            dataset_id,
            command_json,
            options.force_rerun);

        const auto missing_rows = MissingDatasetRows(db.handle(), dataset_id, run_id);
        if (!missing_rows.empty()) {
            auto predictor = predictor_factory(
                artifact_path,
                SentencePiecePathForScratchpad(scratchpad_root).string(),
                options.backend,
                log);
            if (options.backend == nli::SessionBackend::kCoreML &&
                (predictor->ActualBackend() != nli::SessionBackend::kCoreML ||
                 predictor->UsedFallback())) {
                throw std::runtime_error(
                    "Requested backend 'coreml' but session fell back to '" +
                    BackendOptionName(predictor->ActualBackend()) +
                    "'. Refusing to record fallback results as CoreML.");
            }
            for (const DatasetRowRecord& row : missing_rows) {
                const nli::NliLogits logits = predictor->PredictLogits(row.premise, row.hypothesis);
                InsertEvaluationRow(db.handle(), run_id, row.id, logits);
            }
        }

        MarkEvaluationRunStatus(db.handle(), run_id, kRunStatusCompleted);
    } catch (...) {
        if (run_id != 0) {
            try {
                MarkEvaluationRunStatus(db.handle(), run_id, kRunStatusFailed);
            } catch (...) {
            }
        }
        throw;
    }
}

}  // namespace

namespace nli {

StudyPredictorFactory DefaultStudyPredictorFactory() {
    return [](const std::string& model_path,
              const std::string& sentencepiece_path,
              SessionBackend backend,
              std::ostream& log) {
        return std::make_unique<OnnxStudyPredictor>(model_path, sentencepiece_path, backend, log);
    };
}

void InitializeStudyWorkspace(const StudyInitCommandLineOptions& options, std::ostream& log) {
    const std::filesystem::path scratchpad_root = ResolveAbsolutePath(options.scratchpad_root);
    const std::filesystem::path catalog_path = ResolveAbsolutePath(options.catalog_path);
    const std::filesystem::path db_path = scratchpad_root / "db.sqlite3";

    EnsureDirectoryLayout(scratchpad_root);
    if (options.force) {
        RemoveDatabaseFiles(db_path);
    }

    SqliteDatabase db(db_path);
    CreateSchema(db);
    SeedBackends(db);
    UpsertCatalogEntries(db, LoadStudyCatalog(catalog_path), scratchpad_root);
    ImportDatasets(db, scratchpad_root / "datasets");

    log << "Initialized study workspace at " << scratchpad_root << "\n";
}

void RunStudyEvaluation(const StudyRunCommandLineOptions& options, std::ostream& log) {
    RunStudyEvaluation(options, DefaultStudyPredictorFactory(), log);
}

void RunStudyEvaluation(
    const StudyRunCommandLineOptions& options,
    const StudyPredictorFactory& predictor_factory,
    std::ostream& log) {
    RunStudyEvaluationInternal(options, predictor_factory, log, true);
}

}  // namespace nli
