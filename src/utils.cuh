#include "cuda_tools/cuda_error_checking.cuh"

#include <raft/core/device_span.hpp>

#include <rmm/device_uvector.hpp>

#include <array>
#include <cmath>

__global__ void histogram_kernel(const raft::device_span<int> input, raft::device_span<int> histogram, int image_size)
{
    extern __shared__ int sdata[];

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;

    // Réinitialiser l'histogramme partagé à zéro
    if (tid < 256)
    {
        sdata[tid] = 0;
    }
    __syncthreads();

    // Compter les occurrences des pixels dans la mémoire partagée
    if (idx < image_size)
    {
        atomicAdd(&sdata[input[idx]], 1);
    }
    __syncthreads();

    // Transférer les résultats non nuls de sdata vers histogram en mémoire globale
    if (tid < 256 && sdata[tid] > 0)
    {
        atomicAdd(&histogram[tid], sdata[tid]);
    }
}

__global__ void cumulative_histogram_kernel(const raft::device_span<int> histogram, raft::device_span<int> cdf)
{
    extern __shared__ int shared_data[];

    int idx = threadIdx.x;

    // Charger les valeurs d'histogramme dans la mémoire partagée
    shared_data[idx] = histogram[idx];
    __syncthreads();

    // Calculer la somme cumulée
    for (int offset = 1; offset < blockDim.x; offset *= 2)
    {
        int temp = 0;
        if (idx >= offset)
            temp = shared_data[idx - offset];
        __syncthreads();
        shared_data[idx] += temp;
        __syncthreads();
    }

    // Écrire les résultats cumulés dans le tableau CDF
    if (idx < 256)
    {
        cdf[idx] = shared_data[idx];
    }
}

__global__ void equalize_kernel(const raft::device_span<int> input, raft::device_span<int> output,
                                const raft::device_span<int> cdf, int cdf_min, int image_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < image_size)
    {
        // Normalisation du pixel après égalisation de l'histogramme
        float normalized_value = ((cdf[input[idx]] - cdf_min) / static_cast<float>(image_size - cdf_min)) * 255.0f;

        // Limitation de la valeur dans la plage [0, 255]
        output[idx] = min(max(roundf(normalized_value), 0.0f), 255.0f);

        // Pour debug : afficher si la valeur est en dehors des bornes
        if (output[idx] < 0 || output[idx] > 255)
        {
            printf("Valeur hors plage à %d: valeur = %d, cdf_min = %d, cdf_input_idx = %d\n",
                   idx, output[idx], cdf_min, cdf[input[idx]]);
        }
    }
}



__global__ void build_predicate_kernel(raft::device_span<int> buffer, raft::device_span<int> predicate, int garbage_val, int size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size)
    {
        predicate[idx] = (buffer[idx] != garbage_val) ? 1 : 0;
    }
}

__global__ void scatter_kernel(raft::device_span<int> buffer, raft::device_span<int> predicate, int size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < size && buffer[idx] != -27)
    {
        int tmp = buffer[idx];
        __syncthreads();
        buffer[predicate[idx]] = tmp;
    }
}

 __global__ void apply_map_kernel(raft::device_span<int> buffer, int size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size)
    {
        if (idx % 4 == 0)
            buffer[idx] += 1;
        else if (idx % 4 == 1)
            buffer[idx] -= 5;
        else if (idx % 4 == 2)
            buffer[idx] += 3;
        else if (idx % 4 == 3)
            buffer[idx] -= 8;

    }
}

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
void kernel_block_scan_inclusif(raft::device_span<T> buffer, raft::device_span<T> block_sums)
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
void kernel_block_scan_exlusif(raft::device_span<T> buffer, raft::device_span<T> block_sums)
{
    extern __shared__ int sdata[];
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int tid = threadIdx.x;

    sdata[tid] = (id > 0) ? buffer[id - 1] : 0;

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

void your_scan(rmm::device_uvector<int>& buffer, bool exclusif)
{
    constexpr int blocksize = 256;
    int gridsize = (buffer.size() + blocksize - 1) / blocksize;


    rmm::device_uvector<int> block_sums(gridsize, buffer.stream());
    if (exclusif == true){
        kernel_block_scan_exlusif<int><<<gridsize, blocksize, blocksize * sizeof(int), buffer.stream()>>>(
            raft::device_span<int>(buffer.data(), buffer.size()),
            raft::device_span<int>(block_sums.data(), block_sums.size()));
    }
    else{
        kernel_block_scan_inclusif<int><<<gridsize, blocksize, blocksize * sizeof(int), buffer.stream()>>>(
            raft::device_span<int>(buffer.data(), buffer.size()),
            raft::device_span<int>(block_sums.data(), block_sums.size()));
    }

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
