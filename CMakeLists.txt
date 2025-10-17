option(DEBUG "Enable debug compilation" OFF)

# set cmake_install_prefix path
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory")
endif()

message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")

# check if building type is not set
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build" FORCE)
endif()

# set(CMAKE_TOOLCHAIN_FILE ${CMAKE_SOURCE_DIR}/toolchains/aarch64-none-linux-gnu.toolchain.cmake)
cmake_minimum_required(VERSION 3.0)
project(LLAMA-AX650-CPP)

add_compile_options(-std=c++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" CMAKE_SYSTEM_PROCESSOR_LOWER)

# if(CMAKE_SYSTEM_PROCESSOR_LOWER STREQUAL "x86_64" OR
#    CMAKE_SYSTEM_PROCESSOR_LOWER STREQUAL "amd64" OR
#    CMAKE_SYSTEM_PROCESSOR_LOWER STREQUAL "x86" OR
#    CMAKE_SYSTEM_PROCESSOR_LOWER STREQUAL "i386" OR
#    CMAKE_SYSTEM_PROCESSOR_LOWER STREQUAL "i686")
#     set(SYSTEM_ARCH "x86")
#     message(STATUS "Detected x86/x86_64 architecture.")
# elseif(CMAKE_SYSTEM_PROCESSOR_LOWER STREQUAL "aarch64" OR
#        CMAKE_SYSTEM_PROCESSOR_LOWER STREQUAL "arm64")
#     set(SYSTEM_ARCH "aarch64")
#     message(STATUS "Detected aarch64 architecture.")
# else()
#     set(SYSTEM_ARCH "unknown")
#     message(WARNING "Unknown or unsupported system processor architecture: ${CMAKE_SYSTEM_PROCESSOR}")
# endif()


# if(SYSTEM_ARCH STREQUAL "x86")
#     include_directories(third_party/onnxruntime-linux-x64-1.23.0/include)
#     link_directories(third_party/onnxruntime-linux-x64-1.23.0/lib)
# elseif(SYSTEM_ARCH STREQUAL "aarch64") 
#     include_directories(third_party/onnxruntime-linux-aarch64-1.23.0/include)
#     link_directories(third_party/onnxruntime-linux-aarch64-1.23.0/lib)
# endif()

include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

# bsp
if(NOT BSP_MSP_DIR)
    # 判断 /soc/lib/libax_engine.so 是否存在，以确定是否为板端编译
    if(EXISTS /soc/lib/libax_engine.so)
        message(STATUS "Detected board,BSP_MSP_DIR = /soc")
        set(BSP_MSP_DIR /soc)
    else()
        set(BSP_MSP_DIR ${CMAKE_SOURCE_DIR}/bsp_msp_out/msp/out)
    endif()
endif()
message(STATUS "BSP_MSP_DIR = ${BSP_MSP_DIR}")
include_directories(${BSP_MSP_DIR}/include)
link_directories(${BSP_MSP_DIR}/lib)

# 添加 FLAGS 检查代码是否有明显 bug
include(overlook.cmake)

include_directories(src)
include_directories(src/runner)
include_directories(src/runner/utils)

if(NOT OpenCV_DIR)
    set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/third_party/libopencv-4.6-aarch64-none/lib/cmake/opencv4)
endif()

message(STATUS "OPENCV_DIR Path: ${OpenCV_DIR}")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

function(build_exec name main_source)
    add_executable(${name} ${main_source}
                    src/runner/ax_model_runner/ax_model_runner_ax650.cpp 
                    src/runner/BaseRunner.cpp
                    src/runner/utils/whisper.cpp
                    src/runner/utils/memory_utils.cpp 
                    src/runner/utils/cqdm.cpp
                    src/runner/Tokenizer/Tokenizer.cpp
                    src/runner/utils/files.cpp 
                    src/runner/utils/image_processor.cpp 
                    src/runner/utils/mrope.cpp
                    src/runner/utils/slice_3d.tpp
                    )
    if(DEBUG)
        message(STATUS "Debug mode is enabled for target: ${name}")
        target_compile_definitions(${name} PRIVATE DEBUG)
    endif()

    target_link_libraries(${name} ax_engine ax_interpreter ax_sys)
    target_link_libraries(${name} ${OpenCV_LIBS})
    target_link_libraries(${name} onnxruntime)
    install(TARGETS ${name} DESTINATION bin)
endfunction()

build_exec(main src/main.cpp)
# build_exec(main_api src/main_api.cpp)

file(GLOB RUN_SCRIPT "${CMAKE_SOURCE_DIR}/scripts/*.py" "${CMAKE_SOURCE_DIR}/scripts/*.sh")
install(FILES ${RUN_SCRIPT} DESTINATION bin/)

