cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)
project(project_irgpua LANGUAGES CXX CUDA)

set(RAFT_NVTX ON)
include(fetch_rapids.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-find)

rapids_cuda_init_architectures(tp_reduce)

rapids_cpm_init()
include(cmake/thirdparty/get_cccl.cmake)
include(cmake/thirdparty/get_rmm.cmake)
include(cmake/thirdparty/get_raft.cmake)

find_package(OpenMP REQUIRED)

# Set the CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconcepts")
set(CUDA_ARCHITECTURES "75")

include_directories(src)

add_executable(main_cpu  src/main.cu src/fix_cpu.cu)
target_compile_options(main_cpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
target_link_libraries(main_cpu PUBLIC OpenMP::OpenMP_CXX rmm::rmm raft::raft)

add_executable(main_gpu_hand src/utils.cuh src/fix_gpu_hand.cuh src/fix_gpu_hand.cu  src/main_gpu_hand.cu src/cuda_tools/cuda_error_checking.cuh src/cuda_tools/main_helper.hh)
target_compile_options(main_gpu_hand PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
target_link_libraries(main_gpu_hand PUBLIC OpenMP::OpenMP_CXX rmm::rmm raft::raft)
