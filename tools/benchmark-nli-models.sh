#!/bin/bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

build_dir="${repo_root}/builddir"
eval_binary="${build_dir}/nli-eval"
input_dir="${repo_root}/benchmarks/nli"
pattern="*.tsv"
backend="cpu"
model_path="${repo_root}/models/mdeberta/onnx/model.onnx"
compare_model_path="${repo_root}/models/mdeberta/onnx/candidates/dynamic_qint8_matmul_exclude_suggested.onnx"
report_path=""
max_disagreements=""

usage() {
    cat <<EOF
Usage: tools/benchmark-nli-models.sh [options]

Run nli-eval across a directory of TSV benchmark files.

Defaults:
  input dir:      ${input_dir}
  file pattern:   ${pattern}
  backend:        ${backend}
  primary model:  ${model_path}
  compare model:  ${compare_model_path}
  eval binary:    ${eval_binary}

Options:
  --input-dir PATH         Directory containing TSV benchmark files.
  --pattern GLOB           Filename pattern inside the input directory.
  --backend NAME           Session backend to use (default: ${backend}).
  --model PATH             Primary model path.
  --compare-model PATH     Comparison model path.
  --eval-binary PATH       nli-eval binary to run.
  --max-disagreements N    Forward to nli-eval.
  --report PATH            Optional combined report output path.
  --help                   Show this help text.

Examples:
  tools/benchmark-nli-models.sh
  tools/benchmark-nli-models.sh --pattern 'mnli-*.tsv'
  tools/benchmark-nli-models.sh --report benchmarks/nli/full-report.txt
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input-dir)
            if [[ $# -lt 2 ]]; then
                echo "--input-dir requires a path argument" >&2
                exit 1
            fi
            input_dir="$2"
            shift 2
            ;;
        --pattern)
            if [[ $# -lt 2 ]]; then
                echo "--pattern requires a glob argument" >&2
                exit 1
            fi
            pattern="$2"
            shift 2
            ;;
        --backend)
            if [[ $# -lt 2 ]]; then
                echo "--backend requires a backend name" >&2
                exit 1
            fi
            backend="$2"
            shift 2
            ;;
        --model)
            if [[ $# -lt 2 ]]; then
                echo "--model requires a path argument" >&2
                exit 1
            fi
            model_path="$2"
            shift 2
            ;;
        --compare-model)
            if [[ $# -lt 2 ]]; then
                echo "--compare-model requires a path argument" >&2
                exit 1
            fi
            compare_model_path="$2"
            shift 2
            ;;
        --eval-binary)
            if [[ $# -lt 2 ]]; then
                echo "--eval-binary requires a path argument" >&2
                exit 1
            fi
            eval_binary="$2"
            shift 2
            ;;
        --max-disagreements)
            if [[ $# -lt 2 ]]; then
                echo "--max-disagreements requires a numeric argument" >&2
                exit 1
            fi
            max_disagreements="$2"
            shift 2
            ;;
        --report)
            if [[ $# -lt 2 ]]; then
                echo "--report requires a path argument" >&2
                exit 1
            fi
            report_path="$2"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        --*)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
        *)
            echo "Unexpected positional argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ ! -x "${eval_binary}" ]]; then
    echo "nli-eval binary not found or not executable: ${eval_binary}" >&2
    echo "Build it first with: tools/build.sh --target nli-eval" >&2
    exit 1
fi

if [[ ! -d "${input_dir}" ]]; then
    echo "Input directory does not exist: ${input_dir}" >&2
    exit 1
fi

if [[ ! -f "${model_path}" ]]; then
    echo "Primary model file does not exist: ${model_path}" >&2
    exit 1
fi

if [[ ! -f "${compare_model_path}" ]]; then
    echo "Compare model file does not exist: ${compare_model_path}" >&2
    exit 1
fi

if [[ -n "${report_path}" ]]; then
    mkdir -p "$(dirname "${report_path}")"
    : > "${report_path}"
    echo "Writing combined report to ${report_path}"
fi

tsv_files=()
while IFS= read -r tsv_path; do
    tsv_files+=("${tsv_path}")
done < <(find "${input_dir}" -maxdepth 1 -type f -name "${pattern}" -print | sort)

if [[ ${#tsv_files[@]} -eq 0 ]]; then
    echo "No TSV files matched ${input_dir}/${pattern}" >&2
    exit 1
fi

for tsv_path in "${tsv_files[@]}"; do
    header="== ${tsv_path} =="
    echo "${header}"
    if [[ -n "${report_path}" ]]; then
        echo "${header}" >> "${report_path}"
    fi

    command=(
        "${eval_binary}"
        -b "${backend}"
        --model "${model_path}"
        --compare-model "${compare_model_path}"
    )

    if [[ -n "${max_disagreements}" ]]; then
        command+=("--max-disagreements" "${max_disagreements}")
    fi

    command+=("${tsv_path}")

    run_output="$(mktemp)"
    trap 'rm -f "${run_output}"' EXIT
    "${command[@]}" > "${run_output}" 2>&1
    cat "${run_output}"
    if [[ -n "${report_path}" ]]; then
        cat "${run_output}" >> "${report_path}"
    fi
    rm -f "${run_output}"
    trap - EXIT

    echo
    if [[ -n "${report_path}" ]]; then
        echo >> "${report_path}"
    fi
done
