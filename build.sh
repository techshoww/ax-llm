#!/bin/bash

# build_dir 修改为自己想要的编译目录名称
build_dir=build_axcl_x86
echo "build dir: ${build_dir}"
mkdir ${build_dir}
cd ${build_dir}

if [ ! -d /usr/include/axcl/ ]; then
    hf_endpoint=${HF_ENDPOINT:-"https://huggingface.co"}
    axcl_dir=axcl_dir
    echo "axcl not installed, install it in $axcl_dir, hf_endpoint: $hf_endpoint"
    axcl_deb_filename=axcl_host_x86_64_V3.10.2_20251111020143_NO5046.deb
    axcl_url=${hf_endpoint}/AXERA-TECH/AXCL/resolve/main/v3.10.2/$axcl_deb_filename
    mkdir $axcl_dir
    cd $axcl_dir
    if [ ! -f $axcl_deb_filename ]; then
        wget $axcl_url
    fi
    if [ ! -d ./usr ]; then
        dpkg-deb -R ./$axcl_deb_filename .
    fi
    cd ..
fi

# 开始编译
if [ ! -d /usr/include/axcl/ ]; then
    cmake \
        -DTOKENIZER_BUILD_TESTS=OFF \
        -DOpenCV_DIR=$PWD/$opencv_dir/lib/cmake/opencv4 \
        -DAXCL_DIR=$PWD/$axcl_dir/usr \
        ..
else
    cmake \
        -DTOKENIZER_BUILD_TESTS=OFF ..
fi
make -j16
make install