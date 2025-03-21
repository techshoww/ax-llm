#!/bin/bash

# BSP_MSP_DIR 这个变量使用*绝对路径*指定到 SDK 的msp/out目录，如下所示（根据自己的目录修改）
# 绝对路径 绝对路径 绝对路径 
BSP_MSP_DIR=~/AI-support/ax650n_bsp_sdk/msp/out/
echo "bsp dir: ${BSP_MSP_DIR}"
# 下面会简单判断 BSP 路径是否正确
if [ ! -d "${BSP_MSP_DIR}" ]; then
    echo "Error: ${BSP_MSP_DIR} is not a directory"
    exit 1
fi

if [ ! -f "${BSP_MSP_DIR}/lib/libax_sys.so" ]; then
    echo "Error: ${BSP_MSP_DIR}/lib/libax_sys.so is not a file"
    exit 1
fi

# build_dir 修改为自己想要的编译目录名称
build_dir=build
echo "build dir: ${build_dir}"
mkdir ${build_dir}
cd ${build_dir}

# 下载失败可以使用其他方式下载并放到在 $build_dir 目录，参考如下命令解压
URL="https://developer.arm.com/-/media/Files/downloads/gnu-a/9.2-2019.12/binrel/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz"
FOLDER="gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu"

aarch64-none-linux-gnu-gcc -v
if [ $? -ne 0 ]; then
    # Check if the file exists
    if [ ! -f "$FOLDER.tar.xz" ]; then
        # Download the file
        echo "Downloading $URL"
        wget "$URL" -O "$FOLDER.tar.xz"
    fi

    # Check if the folder exists
    if [ ! -d "$FOLDER" ]; then
        # Extract the file
        echo "Extracting $FOLDER.tar.xz"
        tar -xf "$FOLDER.tar.xz"
    fi

    export PATH=$PATH:$PWD/$FOLDER/bin/
    aarch64-none-linux-gnu-gcc -v
    if [ $? -ne 0 ]; then
        echo "Error: aarch64-none-linux-gnu-gcc not found"
        exit 1
    fi
fi

opencv_aarch64_url=https://github.com/ZHEQIUSHUI/assets/releases/download/ax650/libopencv-4.5.5-aarch64.zip
if [ ! -f "libopencv-4.5.5-aarch64.zip" ]; then
    # Download the file
    echo "Downloading $opencv_aarch64_url"
    wget "$opencv_aarch64_url" -O "libopencv-4.5.5-aarch64.zip"
else
    echo "libopencv-4.5.5-aarch64.zip already exists"
fi

# Check if the folder exists
if [ ! -d "libopencv-4.5.5-aarch64" ]; then
    # Extract the file
    echo "Extracting unzip libopencv-4.5.5-aarch64.zip"
    unzip libopencv-4.5.5-aarch64.zip
else
    echo "libopencv-4.5.5-aarch64 already exists"
fi


# 开始编译
cmake -DBSP_MSP_DIR=${BSP_MSP_DIR} -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-none-linux-gnu.toolchain.cmake -DOpenCV_DIR=$PWD/libopencv-4.5.5-aarch64/lib/cmake/opencv4 ..
make -j16
make install