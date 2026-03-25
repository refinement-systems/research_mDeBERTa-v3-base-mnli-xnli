#!/bin/bash

set -euo pipefail

build_dir="${NLI_BUILDDIR:-builddir}"

if [ -e "${build_dir}" ] && [ ! -f "${build_dir}/CMakeCache.txt" ]; then
    echo "${build_dir} exists but is not a CMake build tree; remove it and rerun tools/test.sh" >&2
    exit 1
fi

if [ ! -f "${build_dir}/CMakeCache.txt" ]; then
    (set -x; cmake -S . -B "${build_dir}" -G Ninja \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DCMAKE_INSTALL_PREFIX="$(pwd)")
fi

if ! grep -q '^NLI_BUILD_TESTS:BOOL=ON$' "${build_dir}/CMakeCache.txt"; then
    (set -x; cmake -S . -B "${build_dir}" -DNLI_BUILD_TESTS=ON)
fi

(set -x; cmake --build "${build_dir}" --target nli_tests)
(set -x; exec ctest --test-dir "${build_dir}" --output-on-failure "$@")
