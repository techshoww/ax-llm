#!/usr/bin/env bash

# Cross-compile the AX650/BSP application for aarch64.  BSP_MSP_DIR must be
# the absolute path to the BSP SDK's msp/out directory.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${COSYVOICE_CPP_BUILD_DIR:-${SCRIPT_DIR}/build}"
BSP_MSP_DIR="${BSP_MSP_DIR:-/home/lihongjie/AI-support/ax650n_bsp_sdk/msp/out/}"
JOBS="${COSYVOICE_CPP_JOBS:-$(nproc)}"
TOOLCHAIN_ARCHIVE="gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz"
TOOLCHAIN_DIR="${TOOLCHAIN_ARCHIVE%.tar.xz}"
TOOLCHAIN_URL="https://developer.arm.com/-/media/Files/downloads/gnu-a/9.2-2019.12/binrel/${TOOLCHAIN_ARCHIVE}"
OPENCV_ARCHIVE="libopencv-4.5.5-aarch64.zip"
OPENCV_DIR="${SCRIPT_DIR}/libopencv-4.5.5-aarch64"
OPENCV_URL="https://github.com/ZHEQIUSHUI/assets/releases/download/ax650/${OPENCV_ARCHIVE}"

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

if [ -z "${BSP_MSP_DIR}" ]; then
    echo "Error: set BSP_MSP_DIR to the BSP SDK msp/out directory" >&2
    exit 1
fi
if [ ! -f "${BSP_MSP_DIR}/lib/libax_sys.so" ]; then
    echo "Error: ${BSP_MSP_DIR}/lib/libax_sys.so is not available" >&2
    exit 1
fi

if ! command -v aarch64-none-linux-gnu-gcc >/dev/null 2>&1; then
    if [ ! -f "${SCRIPT_DIR}/${TOOLCHAIN_ARCHIVE}" ]; then
        echo "Downloading aarch64 toolchain from ${TOOLCHAIN_URL}"
        download_file "${TOOLCHAIN_URL}" "${SCRIPT_DIR}/${TOOLCHAIN_ARCHIVE}"
    fi
    if [ ! -d "${SCRIPT_DIR}/${TOOLCHAIN_DIR}" ]; then
        tar -xf "${SCRIPT_DIR}/${TOOLCHAIN_ARCHIVE}" -C "${SCRIPT_DIR}"
    fi
    export PATH="${SCRIPT_DIR}/${TOOLCHAIN_DIR}/bin:${PATH}"
fi

if ! command -v aarch64-none-linux-gnu-gcc >/dev/null 2>&1; then
    echo "Error: aarch64-none-linux-gnu-gcc not found" >&2
    exit 1
fi

if [ ! -d "${OPENCV_DIR}" ]; then
    if [ ! -f "${SCRIPT_DIR}/${OPENCV_ARCHIVE}" ]; then
        echo "Downloading aarch64 OpenCV from ${OPENCV_URL}"
        download_file "${OPENCV_URL}" "${SCRIPT_DIR}/${OPENCV_ARCHIVE}"
    fi
    unzip -q "${SCRIPT_DIR}/${OPENCV_ARCHIVE}" -d "${SCRIPT_DIR}"
fi

EXTRA_CMAKE_ARGS=()
if [ -n "${COSYVOICE_CPP_CMAKE_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    EXTRA_CMAKE_ARGS=(${COSYVOICE_CPP_CMAKE_ARGS})
fi

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${BUILD_DIR}/install" \
    -DBSP_MSP_DIR="${BSP_MSP_DIR}" \
    -DCMAKE_TOOLCHAIN_FILE="${SCRIPT_DIR}/toolchains/aarch64-none-linux-gnu.toolchain.cmake" \
    -DOpenCV_DIR="${OPENCV_DIR}/lib/cmake/opencv4" \
    "${EXTRA_CMAKE_ARGS[@]}"
cmake --build "${BUILD_DIR}" --parallel "${JOBS}"
cmake --install "${BUILD_DIR}"
