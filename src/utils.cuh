#include "cuda_tools/cuda_error_checking.cuh"

#include <raft/core/device_span.hpp>

#include <rmm/device_uvector.hpp>

#include <array>
#include <cmath>

__global__ void histogram_kernel(const raft::device_span<int> input, raft::device_span<int> histogram, int image_size)
{
    __shared__ int sdata[256];

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

//TODO : Utiliser dans la fonction fix_image_gpu_hand et changer equalize_kernel
__global__ void find_cdf_min_kernel(const raft::device_span<int> histogram, raft::device_span<int> cdf) {
    int tid = threadIdx.x;
    extern __shared__ int shared_min[];

    // Initialiser la valeur partagée avec un maximum élevé
    if (tid == 0) {
        shared_min[0] = INT_MAX;
    }
    __syncthreads();

    // Trouver le premier élément non nul dans le histogramme
    if (histogram[tid] != 0) {
        atomicMin(&shared_min[0], histogram[tid]);
    }
    __syncthreads();

    // Copier le résultat final dans cdf[0] par le thread 0
    if (tid == 0) {
        cdf[0] = shared_min[0];
    }
} //! Ce kernel trouvera le premier élément non nul et le stockera dans cdf[0]


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



// __global__ void build_predicate_kernel(raft::device_span<int> buffer, raft::device_span<int> predicate, int garbage_val, int size)
// {
//     //? Shared memory
//     //? mettre le predicate a 0 en dehors du kernel -> n'ecrire que pour mettre des 1
//     int idx = blockDim.x * blockIdx.x + threadIdx.x;
//     if (idx < size)
//     {
//         predicate[idx] = (buffer[idx] != garbage_val) ? 1 : 0;
//     }
// }

__global__ void build_predicate_kernel(raft::device_span<int> buffer, raft::device_span<int> predicate, int garbage_val, int size)
{
    extern __shared__ int shared_buffer[];
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;

    // Charger les données dans la mémoire partagée
    if (idx < size)
    {
        shared_buffer[tid] = buffer[idx];
        __syncthreads();
        if (shared_buffer[tid] != garbage_val)
        {
            predicate[idx] = 1;
        }
    }
}

//TODO : NsightCompute -> A voir si cest mieux d'avoir le Memset a 0 avant ou pas car performance a peu pres pareil
// __global__ void build_predicate_kernel(raft::device_span<int> buffer, raft::device_span<int> predicate, int garbage_val, int size)
// {
//     int idx = blockDim.x * blockIdx.x + threadIdx.x;
//     int tid = threadIdx.x;
//     extern __shared__ int shared_predicate[];
//     if (idx < size) {
//         shared_buffer[tid] = buffer[idx];
//         shared_predicate[tid] = (shared_buffer[tid] != garbage_val) ? 1 : 0;
//         __syncthreads();
//         if (tid == 0) {
//             for (int i = 0; i < blockDim.x; ++i) {
//                 if ((blockIdx.x * blockDim.x + i) < size) {
//                     predicate[blockIdx.x * blockDim.x + i] = shared_predicate[i];
//                 }
//             }
//         }
//     }
// }


// __global__ void scatter_kernel(raft::device_span<int> buffer, raft::device_span<int> predicate, int size)
// {

//     //? Shared memory ?

//     int idx = blockDim.x * blockIdx.x + threadIdx.x;

//     if (idx < size) {
//         int tmp = buffer[idx];

//         if (tmp != -27) {
//             __syncthreads();
//             buffer[predicate[idx]] = tmp;
//         }
//     }
// }

//TODO : NsightCompute -> Analyse de la performance
__global__ void scatter_kernel(raft::device_span<int> buffer, raft::device_span<int> predicate, int size) {
    extern __shared__ int shared_buffer[];
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;

    // Charger les données de buffer dans la mémoire partagée, tout en vérifiant les limites
    if (idx < size) {
        shared_buffer[tid] = buffer[idx];
    }
    __syncthreads();

    if (idx < size && shared_buffer[tid] != -27) {
        int new_position = predicate[idx];

        // Vérification des limites de new_position
        if (new_position >= 0 && new_position < size) {
            // Utiliser atomicCAS ou une autre technique pour éviter les écritures concurrentes si nécessaire
            atomicExch(&buffer[new_position], shared_buffer[tid]);
        }
    }
}

//  __global__ void apply_map_kernel(raft::device_span<int> buffer, int size)
// {
//     /*
//     matrice :
//     | 1  -5 |
//     | 3  -8 |
//     */
//     int idx = blockDim.x * blockIdx.x + threadIdx.x;
//     if (idx < size)
//     {
//         if (idx % 4 == 0)
//             buffer[idx] += 1;
//         else if (idx % 4 == 1)
//             buffer[idx] -= 5;
//         else if (idx % 4 == 2)
//             buffer[idx] += 3;
//         else if (idx % 4 == 3)
//             buffer[idx] -= 8;
//     }
// }

__global__ void apply_map_kernel(raft::device_span<int> buffer, int size) {
    // Charger la matrice dans la mémoire partagée
    __shared__ int matrix[4];  // Matrice 1D : [1, -5, 3, -8]
    if (threadIdx.x < 4) {
        // Chaque thread de l'index 0 à 3 charge un élément de la matrice
        int values[4] = {1, -5, 3, -8};
        matrix[threadIdx.x] = values[threadIdx.x];
    }
    __syncthreads();

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        // Appliquer l'opération en utilisant l'élément de la matrice
        buffer[idx] += matrix[idx % 4];
    }
}



__device__ void warp_reduce_tot(int *sdata, int tid, int block_size) {
    if (block_size >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (block_size >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (block_size >= 128) { if (tid < 64)  { sdata[tid] += sdata[tid + 64]; }  __syncthreads(); }
    if (block_size >= 64)  { if (tid < 32)  { sdata[tid] += sdata[tid + 32]; }  __syncthreads(); }
    if (block_size >= 32)  { if (tid < 16)  { sdata[tid] += sdata[tid + 16]; }  __syncthreads(); }
    if (block_size >= 16)  { if (tid < 8)   { sdata[tid] += sdata[tid + 8]; }   __syncthreads(); }
    if (block_size >= 8)   { if (tid < 4)   { sdata[tid] += sdata[tid + 4]; }   __syncthreads(); }
    if (block_size >= 4)   { if (tid < 2)   { sdata[tid] += sdata[tid + 2]; }   __syncthreads(); }
    if (block_size >= 2)   { if (tid < 1)   { sdata[tid] += sdata[tid + 1]; }   __syncthreads(); }
}


template <typename T>
__global__
void kernel_your_reduce_grid_stride_loop(raft::device_span<const T> buffer, raft::device_span<T> total)
{
    extern __shared__ int sdata[];
    const unsigned int BLOCK_SIZE = blockDim.x;

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * BLOCK_SIZE * 2 + threadIdx.x;
    unsigned int grid_size = BLOCK_SIZE * gridDim.x * 2;


    // Initialisation locale du tableau partagé
    sdata[tid] = 0;
    __syncthreads();

    // Utilisation de la grid-stride loop pour traiter plusieurs éléments
    for (unsigned int idx = i; idx < buffer.size(); idx += grid_size) {
        sdata[tid] += buffer[idx] + ((idx + BLOCK_SIZE < buffer.size())? buffer[idx + BLOCK_SIZE] : 0);
    }
    __syncthreads();

    // Unrolled everything here
    warp_reduce_tot(sdata, tid, BLOCK_SIZE);
    __syncthreads();

    // Le thread 0 écrit le résultat partiel dans le tableau total
    if (tid == 0) total[blockIdx.x] = sdata[0];
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
        // kernel_block_scan_inclusif<int><<<1, blocksize, blocksize * sizeof(int), buffer.stream()>>>(
        //     raft::device_span<int>(block_sums.data(), block_sums.size()),
        //     raft::device_span<int>(block_sums.data(), block_sums.size()));

        //? Changement
        kernel_sum_block_sums<int><<<1, 1, 0, buffer.stream()>>>(
            raft::device_span<int>(block_sums.data(), block_sums.size()));
        CUDA_CHECK_ERROR(cudaStreamSynchronize(buffer.stream()));
    }


    kernel_update_blocks<int><<<gridsize, blocksize, 0, buffer.stream()>>>(
        raft::device_span<int>(buffer.data(), buffer.size()),
        raft::device_span<int>(block_sums.data(), block_sums.size()));

    CUDA_CHECK_ERROR(cudaStreamSynchronize(buffer.stream()));
}
