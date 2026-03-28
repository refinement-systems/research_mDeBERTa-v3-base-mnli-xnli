#!/bin/bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

model_id="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
base_url="https://huggingface.co/${model_id}/resolve/main"
output_dir="${repo_root}/models/mdeberta"
force=0

readonly default_files=(
    "spm.model"
    "onnx/model.onnx"
    "onnx/model_quantized.onnx"
)

usage() {
    cat <<EOF
Usage: tools/download-mdeberta-v3-base.sh [options] [FILE...]

Download files from https://huggingface.co/${model_id} into:
  ${output_dir}

Without positional FILE arguments, the script downloads:
  ${default_files[*]}

Options:
  --dir PATH   Override the output directory.
  --force      Re-download files even if they already exist locally.
  --help       Show this help text.

Examples:
  tools/download-mdeberta-v3-base.sh
  tools/download-mdeberta-v3-base.sh --force
  tools/download-mdeberta-v3-base.sh spm.model onnx/model.onnx
EOF
}

download_file() {
    local file_name="$1"
    local destination="${output_dir}/${file_name}"
    local temporary="${destination}.tmp"

    mkdir -p "$(dirname "${destination}")"

    if [[ -f "${destination}" && "${force}" -eq 0 ]]; then
        echo "Skipping ${file_name}; already exists at ${destination}"
        return
    fi

    echo "Downloading ${file_name} -> ${destination}"
    curl --fail --location --progress-bar \
        --output "${temporary}" \
        "${base_url}/${file_name}?download=1"
    mv "${temporary}" "${destination}"
}

files=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dir)
            if [[ $# -lt 2 ]]; then
                echo "--dir requires a path argument" >&2
                exit 1
            fi
            output_dir="$2"
            shift 2
            ;;
        --force)
            force=1
            shift
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
            files+=("$1")
            shift
            ;;
    esac
done

if [[ ${#files[@]} -eq 0 ]]; then
    files=("${default_files[@]}")
fi

for file_name in "${files[@]}"; do
    download_file "${file_name}"
done
