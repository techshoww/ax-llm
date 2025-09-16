# set cross-compiled system type, it's better not use the type which cmake cannot recognized.
SET (CMAKE_SYSTEM_NAME Linux)
SET (CMAKE_SYSTEM_PROCESSOR aarch64)

# aarch64-linux-gnu-gcc DO NOT need to be installed, so make sure aarch64-linux-gnu-gcc and aarch64-linux-gnu-g++ can be found in $PATH:
SET (CMAKE_C_COMPILER   "aarch64-none-linux-gnu-gcc")
SET (CMAKE_CXX_COMPILER "aarch64-none-linux-gnu-g++")

# set searching rules for cross-compiler
SET (CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
SET (CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET (CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# if(NOT CMAKE_FIND_ROOT_PATH_MODE_PACKAGE)
#     set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
# endif()

set(CMAKE_C_FLAGS "-march=armv8-a")
set(CMAKE_CXX_FLAGS "-march=armv8-a")

# cache flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}" CACHE STRING "c flags")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "c++ flags")