#!/usr/bin/env bash

# Build the AXCL PCIe application for an x86_64 host.
# The generated package is placed in build_axcl_x86/install by default.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${COSYVOICE_CPP_BUILD_DIR:-${SCRIPT_DIR}/build_axcl_x86}"
AXCL_VERSION="${AXCL_VERSION:-3.6.2}"
AXCL_ARCHIVE="axcl_${AXCL_VERSION}_x86.zip"
AXCL_URL="https://github.com/ZHEQIUSHUI/assets/releases/download/ax_${AXCL_VERSION}/${AXCL_ARCHIVE}"
JOBS="${COSYVOICE_CPP_JOBS:-$(nproc)}"

download_file() {
    local url="$1"
    local output="$2"
    if command -v curl >/dev/null 2>&1; then
        curl -fL --retry 10 --retry-all-errors --retry-delay 2 \
            --connect-timeout 20 --max-time 1200 -o "${output}" "${url}"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "${output}" "${url}"
    else
        echo "Error: curl or wget is required to download ${output}" >&2
        exit 1
    fi
}

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

if [ ! -d "axcl_${AXCL_VERSION}" ]; then
    if [ ! -f "${AXCL_ARCHIVE}" ]; then
        echo "Downloading AXCL SDK from ${AXCL_URL}"
        download_file "${AXCL_URL}" "${AXCL_ARCHIVE}"
    fi
    unzip -q "${AXCL_ARCHIVE}"
fi
AXCL_DIR="${PWD}/axcl_${AXCL_VERSION}"

# Optional extra CMake arguments, for example:
# COSYVOICE_CPP_CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Debug"
EXTRA_CMAKE_ARGS=()
if [ -n "${COSYVOICE_CPP_CMAKE_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    EXTRA_CMAKE_ARGS=(${COSYVOICE_CPP_CMAKE_ARGS})
fi

cmake -S "${SCRIPT_DIR}" -B "${PWD}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${PWD}/install" \
    -DAXCL_DIR="${AXCL_DIR}" \
    "${EXTRA_CMAKE_ARGS[@]}"
cmake --build "${PWD}" --parallel "${JOBS}"
cmake --install "${PWD}"
