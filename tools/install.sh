#!/bin/bash

(set -x; exec cmake --install builddir --component runtime "$@")
