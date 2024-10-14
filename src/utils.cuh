#include "cuda_tools/cuda_error_checking.cuh"

#include <raft/core/device_span.hpp>

#include <rmm/device_uvector.hpp>


template <typename T>
__global__
void kernel_reduce_baseline(raft::device_span<const T> buffer, raft::device_span<T> total)
{
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < buffer.size())
        atomicAdd(total.data(), buffer[id]);
}

void baseline_reduce(rmm::device_uvector<int>& buffer,
                     rmm::device_scalar<int>& total)
{
    constexpr int blocksize = 64;
    const int gridsize = (buffer.size() + blocksize - 1) / blocksize;

    kernel_reduce_baseline<int><<<gridsize, blocksize, 0, buffer.stream()>>>(
      raft::device_span<int>(buffer.data(), buffer.size()),
      raft::device_span<int>(total.data(), 1));

    CUDA_CHECK_ERROR(cudaStreamSynchronize(buffer.stream()));
}

template <typename T>
__global__
void kernel_block_scan(raft::device_span<T> buffer, raft::device_span<T> block_sums)
{
    extern __shared__ int sdata[];
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int tid = threadIdx.x;


    if (id < buffer.size()) {
        sdata[tid] = buffer[id];
    } else {
        sdata[tid] = 0;
    }
    __syncthreads();


    for (unsigned int i = 1; i < blockDim.x; i *= 2)
    {
        int temp = 0;
        if (tid >= i)
        {
            temp = sdata[tid - i];
        }
        __syncthreads();
        sdata[tid] += temp;
        __syncthreads();
    }


    if (id < buffer.size()) {
        buffer[id] = sdata[tid];
    }

    if (tid == blockDim.x - 1 && blockIdx.x < block_sums.size()) {
        block_sums[blockIdx.x] = sdata[tid];
    }
}

template <typename T>
__global__
void kernel_sum_block_sums(raft::device_span<T> block_sums)
{

    for (int i = 1; i < block_sums.size(); ++i)
    {
        block_sums[i] += block_sums[i - 1];
    }
}

template <typename T>
__global__
void kernel_update_blocks(raft::device_span<T> buffer, raft::device_span<T> block_sums)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (blockIdx.x > 0 && id < buffer.size()) {
        buffer[id] += block_sums[blockIdx.x - 1]; // Ajouter la somme du bloc précédent
    }
}

void your_scan(rmm::device_uvector<int>& buffer)
{
    constexpr int blocksize = 64;
    int gridsize = (buffer.size() + blocksize - 1) / blocksize;


    rmm::device_uvector<int> block_sums(gridsize, buffer.stream());


    kernel_block_scan<int><<<gridsize, blocksize, blocksize * sizeof(int), buffer.stream()>>>(
        raft::device_span<int>(buffer.data(), buffer.size()),
        raft::device_span<int>(block_sums.data(), block_sums.size()));

    CUDA_CHECK_ERROR(cudaStreamSynchronize(buffer.stream()));

    if (gridsize > 1) {
        kernel_sum_block_sums<int><<<1, 1, 0, buffer.stream()>>>(
            raft::device_span<int>(block_sums.data(), block_sums.size()));
        CUDA_CHECK_ERROR(cudaStreamSynchronize(buffer.stream()));
    }


    kernel_update_blocks<int><<<gridsize, blocksize, 0, buffer.stream()>>>(
        raft::device_span<int>(buffer.data(), buffer.size()),
        raft::device_span<int>(block_sums.data(), block_sums.size()));

    CUDA_CHECK_ERROR(cudaStreamSynchronize(buffer.stream()));
}

