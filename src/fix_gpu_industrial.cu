#include "fix_gpu_industrial.cuh"
#include "utils.cuh"
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/count.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/fill.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <cub/cub.cuh>

struct transform_op
{
    __host__ __device__
    int operator()(const thrust::tuple<int, int>& input)
    {
        int value = thrust::get<0>(input);
        int index = thrust::get<1>(input);

        if (index % 4 == 0)
            return value + 1;
        else if (index % 4 == 1)
            return value - 5;
        else if (index % 4 == 2)
            return value + 3;
        else
            return value - 8;
    }
};

void fix_image_gpu_industrial(Image& to_fix)
{
    const int image_size = to_fix.width * to_fix.height;
    const int garbage_val = -27;

    // Créer un stream CUDA
    cudaStream_t stream;
    CUDA_CHECK_ERROR(cudaStreamCreate(&stream));

    // Copier le buffer de `to_fix` vers un `thrust::device_vector`
    thrust::device_vector<int> d_buffer(to_fix.buffer, to_fix.buffer + to_fix.size());

    // Faire une copie profonde de d_buffer pour le scatter final
    thrust::device_vector<int> d_buffer_copy = d_buffer;

    // Créer le vecteur de prédicat
    thrust::device_vector<int> predicate(to_fix.size());

    // Générer le vecteur de prédicat
    thrust::transform(thrust::cuda::par.on(stream), d_buffer.begin(), d_buffer.end(), predicate.begin(),
                      [garbage_val] __device__ (int val) { return val != garbage_val ? 1 : 0; });

    // Effectuer un scan exclusif pour calculer les indices des éléments valides
    thrust::exclusive_scan(thrust::cuda::par.on(stream), predicate.begin(), predicate.end(), predicate.begin(), 0);

    // Appliquer `thrust::scatter_if` pour redistribuer les éléments
    thrust::scatter_if(thrust::cuda::par.on(stream),
                       d_buffer_copy.begin(), d_buffer_copy.end(),  // Données à redistribuer
                       predicate.begin(),                            // Position cible
                       d_buffer_copy.begin(),                            // Conditions de filtrage
                       d_buffer.begin(),                             // Sortie
                       [garbage_val] __device__ (int val) { return val != garbage_val; }); // Filtrer les valeurs

    // Transformer les valeurs de d_buffer en fonction de l'index modulo 4
    thrust::transform(thrust::cuda::par.on(stream),
                      thrust::make_zip_iterator(thrust::make_tuple(d_buffer.begin(), thrust::counting_iterator<int>(0))),
                      thrust::make_zip_iterator(thrust::make_tuple(d_buffer.end(), thrust::counting_iterator<int>(image_size))),
                      d_buffer.begin(),
                      transform_op());

    // Synchroniser le stream pour s'assurer que toutes les opérations sont terminées
    CUDA_CHECK_ERROR(cudaStreamSynchronize(stream));

    // Copier le résultat final dans le buffer d'origine
    CUDA_CHECK_ERROR(cudaMemcpy(to_fix.buffer, thrust::raw_pointer_cast(d_buffer.data()),
                                image_size * sizeof(int), cudaMemcpyDeviceToHost));

    // Détruire le stream
    CUDA_CHECK_ERROR(cudaStreamDestroy(stream));
}


/*
bool check_predicate(const thrust::device_vector<int>& predicate_gpu, const std::vector<int>& predicate_cpu) {
    // Vérifier que les tailles des deux vecteurs sont identiques
    if (predicate_gpu.size() != predicate_cpu.size()) {
        std::cerr << "Size mismatch: GPU predicate size (" << predicate_gpu.size()
                  << ") vs CPU predicate size (" << predicate_cpu.size() << ")" << std::endl;
        return false;
    }

    // Créer un vecteur hôte pour recevoir les données du GPU
    std::vector<int> predicate_gpu_host(predicate_gpu.size());

    // Copier les données du GPU vers le vecteur hôte
    CUDA_CHECK_ERROR(cudaMemcpy(predicate_gpu_host.data(), thrust::raw_pointer_cast(predicate_gpu.data()),
                                predicate_gpu.size() * sizeof(int), cudaMemcpyDeviceToHost));

    // Comparer les deux vecteurs élément par élément
    for (std::size_t i = 0; i < predicate_cpu.size(); ++i) {
        if (predicate_gpu_host[i] != predicate_cpu[i]) {
            std::cerr << "Mismatch at index " << i << ": CPU(" << predicate_cpu[i]
                      << ") vs GPU(" << predicate_gpu_host[i] << ")" << std::endl;
            return false;
        }
    }

    // Si toutes les valeurs sont identiques, les vecteurs sont identiques
    return true;
}

// Appeler le kernel de scatter pour réorganiser les éléments
/*scatter_kernel<<<(to_fix.size() + 255) / 256, 256, 0, d_buffer.stream()>>>(
    raft::device_span<int>(d_buffer.data(), d_buffer.size()),
    raft::device_span<int>(predicate.data(), predicate.size()),
    to_fix.size());
CUDA_CHECK_ERROR(cudaStreamSynchronize(d_buffer.stream()));

// Réaliser l’égalisation d’histogramme
thrust::device_vector<int> histogram(256, 0);
thrust::device_vector<int> d_image(d_buffer.data(), d_buffer.data() + image_size);
thrust::sort(d_image.begin(), d_image.end());
thrust::counting_iterator<int> search_begin(0);
thrust::upper_bound(d_image.begin(), d_image.end(), search_begin, search_begin + 256, histogram.begin());

// Calculer le CDF
thrust::exclusive_scan(histogram.begin(), histogram.end(), histogram.begin());

// Trouver cdf_min
//! error avec copy_if
/*int cdf_min;
thrust::copy_if(histogram.begin(), histogram.end(), thrust::make_counting_iterator(1), &cdf_min,
                [] __device__(int x) { return x != 0; });

// Appliquer l’égalisation
equalize_kernel<<<(image_size + 255) / 256, 256, 0, d_buffer.stream()>>>(
    raft::device_span<int>(d_buffer.data(), d_buffer.size()),
    raft::device_span<int>(d_buffer.data(), d_buffer.size()),
    raft::device_span<int>(histogram.data().get(), 256),
    cdf_min, image_size);*/

//CUDA_CHECK_ERROR(cudaMemcpy(to_fix.buffer, d_buffer.data(), image_size * sizeof(int), cudaMemcpyDeviceToHost));

