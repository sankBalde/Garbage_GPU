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

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/scatter.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <iostream>

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
    const int num_bins = 256;  // Nombre de niveaux de gris pour l'histogramme
    const int min_val = 0;     // Valeur minimale pour l'histogramme
    const int max_val = 255;   // Valeur maximale pour l'histogramme

    // Créer un stream CUDA
    cudaStream_t stream;
    CUDA_CHECK_ERROR(cudaStreamCreate(&stream));

    // Copier le buffer de `to_fix` vers un `thrust::device_vector`
    thrust::device_vector<int> d_buffer(to_fix.buffer, to_fix.buffer + to_fix.size());
    thrust::device_vector<int> d_buffer_copy = d_buffer; // Copie profonde pour le scatter final

    // Générer le vecteur de prédicat
    thrust::device_vector<int> predicate(to_fix.size());
    thrust::transform(thrust::cuda::par.on(stream), d_buffer.begin(), d_buffer.end(), predicate.begin(),
                      [garbage_val] __device__ (int val) { return val != garbage_val ? 1 : 0; });
    thrust::exclusive_scan(thrust::cuda::par.on(stream), predicate.begin(), predicate.end(), predicate.begin(), 0);

    thrust::scatter_if(thrust::cuda::par.on(stream),
                       d_buffer_copy.begin(), d_buffer_copy.end(),  // Données à redistribuer
                       predicate.begin(),                            // Position cible
                       d_buffer_copy.begin(),                            // Conditions de filtrage
                       d_buffer.begin(),                             // Sortie
                       [garbage_val] __device__ (int val) { return val != garbage_val; }); // Filtrer les valeurs


    // Transformer les valeurs de `d_buffer` en fonction de l'index modulo 4
    thrust::transform(thrust::cuda::par.on(stream),
                      thrust::make_zip_iterator(thrust::make_tuple(d_buffer.begin(), thrust::counting_iterator<int>(0))),
                      thrust::make_zip_iterator(thrust::make_tuple(d_buffer.end(), thrust::counting_iterator<int>(image_size))),
                      d_buffer.begin(),
                      transform_op());

    // Allocations pour le calcul de l'histogramme avec CUB
    thrust::device_vector<int> histogram(num_bins, 0);
    int *d_histogram = thrust::raw_pointer_cast(histogram.data());

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes, thrust::raw_pointer_cast(d_buffer.data()), d_histogram, num_bins, min_val, max_val + 1, image_size, stream);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes, thrust::raw_pointer_cast(d_buffer.data()), d_histogram, num_bins, min_val, max_val + 1, image_size, stream);

    // Calcul du CDF pour l'égalisation
    thrust::device_vector<float> cdf(num_bins);
    thrust::inclusive_scan(thrust::cuda::par.on(stream), histogram.begin(), histogram.end(), cdf.begin());

    // Normaliser le CDF pour obtenir les valeurs d'égalisation
    float min_cdf;
    cudaMemcpy(&min_cdf, thrust::raw_pointer_cast(cdf.data()), sizeof(float), cudaMemcpyDeviceToHost);
    thrust::transform(thrust::cuda::par.on(stream), cdf.begin(), cdf.end(), cdf.begin(), [min_cdf, image_size] __device__ (float c) {
        return (c - min_cdf) / (image_size - min_cdf) * 255.0f;
    });

    // Remapper les valeurs de `d_buffer` pour appliquer l'égalisation
    thrust::transform(thrust::cuda::par.on(stream), d_buffer.begin(), d_buffer.end(), d_buffer.begin(), [d_cdf = thrust::raw_pointer_cast(cdf.data())] __device__ (int val) {
        return static_cast<int>(d_cdf[val]);
    });

    // Copier le résultat final dans le buffer d'origine
    CUDA_CHECK_ERROR(cudaMemcpy(to_fix.buffer, thrust::raw_pointer_cast(d_buffer.data()), image_size * sizeof(int), cudaMemcpyDeviceToHost));

    // Libérer les ressources
    //cudaFree(d_temp_storage);
    CUDA_CHECK_ERROR(cudaStreamDestroy(stream));
}
