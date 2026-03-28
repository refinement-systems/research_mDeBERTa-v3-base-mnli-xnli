#!/bin/bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
legacy_commit="3339662"

tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/nli-legacy-qdebug.XXXXXX")"
cleanup() {
    rm -rf "${tmp_dir}"
}
trap cleanup EXIT

legacy_script="${tmp_dir}/debug-onnx-quantization.py"
git show "${legacy_commit}:tools/debug-onnx-quantization.py" > "${legacy_script}"
chmod +x "${legacy_script}"

exec "${repo_root}/.venv/bin/python" "${legacy_script}" "$@"
