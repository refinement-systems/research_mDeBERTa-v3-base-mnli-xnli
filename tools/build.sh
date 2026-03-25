#!/bin/bash

(set -x; exec cmake --build builddir "$@")
