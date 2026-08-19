cmake_minimum_required(VERSION 3.22)

# sm_120a is the development machine; the target is B200 (sm_100a). Override
# with -DCMAKE_CUDA_ARCHITECTURES=100a. The `a` suffix matters -- the e8m0
# intrinsics and the 32-byte vector path need the arch-specific target.
#
# This has to be decided *before* project(), which otherwise sets
# CMAKE_CUDA_ARCHITECTURES itself (to nvcc's default, sm_75) and makes a
# post-project `if(NOT DEFINED ...)` guard dead code.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "120a" CACHE STRING "CUDA architectures to build for")
endif()

project(rmsnorm_kernel_dev CUDA CXX)

set(CMAKE_CXX_STANDARD 20)

message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

include_directories(src/include ${CMAKE_CURRENT_SOURCE_DIR})

# One kernel variant per file; rmsnorm.cu holds only the dispatcher.
set(RMSNORM_SOURCES
    rmsnorm.cu
    src/kernels/rmsnorm_00_baseline.cu
    src/kernels/rmsnorm_01_loop_epi.cu
    src/kernels/rmsnorm_02_fixed_c.cu
    src/kernels/rmsnorm_03_fixed_32pt.cu
    src/kernels/rmsnorm_04_pers_32pt.cu
    src/kernels/rmsnorm_05_tma.cu
)

add_executable(bench ${RMSNORM_SOURCES} bench.cu)
target_compile_options(bench PUBLIC -lineinfo)

add_executable(test_rmsnorm ${RMSNORM_SOURCES} test.cu)
target_compile_options(test_rmsnorm PUBLIC -lineinfo)

enable_testing()
add_test(NAME rmsnorm COMMAND test_rmsnorm)

# The torch binding in csrc/ is what `pip install` compiles into
# prime_kernels.rmsnorm._C; building it here too keeps a compile error in it from
# only showing up at packaging time. It needs libtorch, so skip it when that is
# not available and let the kernel/test targets build regardless.
# `torch/extension.h` pulls in Python.h, so the headers of the interpreter that
# owns this libtorch are needed too; point cmake at it with
# -DPython_EXECUTABLE=/path/to/venv/bin/python.
find_package(Torch QUIET)
find_package(Python COMPONENTS Development.Module QUIET)
if(Torch_FOUND AND Python_FOUND)
    add_library(rmsnorm_torch MODULE csrc/torch_interface.cpp ${RMSNORM_SOURCES})
    target_link_libraries(rmsnorm_torch PRIVATE "${TORCH_LIBRARIES}" Python::Module)
else()
    message(STATUS "libtorch or the Python headers not found -- skipping the rmsnorm_torch target")
endif()
