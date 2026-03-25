#!/bin/bash

if [ -e builddir ] && [ ! -f builddir/CMakeCache.txt ]; then
    echo "builddir exists but is not a CMake build tree; remove it and rerun tools/setup.sh" >&2
    exit 1
fi

(set -x; exec cmake -S . -B builddir -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_INSTALL_PREFIX="$(pwd)" \
    "$@")
