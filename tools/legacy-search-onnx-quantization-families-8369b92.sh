#!/bin/bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
legacy_commit="8369b92"

tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/nli-legacy-family-search.XXXXXX")"
cleanup() {
    rm -rf "${tmp_dir}"
}
trap cleanup EXIT

legacy_script="${tmp_dir}/search-onnx-quantization-families.py"
git show "${legacy_commit}:tools/search-onnx-quantization-families.py" > "${legacy_script}"
chmod +x "${legacy_script}"

exec "${repo_root}/.venv/bin/python" "${legacy_script}" "$@"
