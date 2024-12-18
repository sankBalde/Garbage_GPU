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

#include <raft/common/nvtx.hpp>

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
    raft::common::nvtx::range fun_scope("Industrial Fix Image");
    const int image_size = to_fix.width * to_fix.height;
    const int garbage_val = -27;
    const int num_bins = 256;
    const int min_val = 0;
    const int max_val = 255;

    // Créer un stream CUDA
    cudaStream_t stream;
    CUDA_CHECK_ERROR(cudaStreamCreate(&stream));

    // Copier le buffer de `to_fix` vers un `thrust::device_vector`
    thrust::device_vector<int> d_buffer(to_fix.buffer, to_fix.buffer + to_fix.size());
    thrust::device_vector<int> d_buffer_copy = d_buffer;

    // Générer le vecteur de prédicat
    thrust::device_vector<int> predicate(to_fix.size());
    raft::common::nvtx::push_range("Build Predicate");
    thrust::transform(thrust::cuda::par.on(stream), d_buffer.begin(), d_buffer.end(), predicate.begin(),
                      [garbage_val] __device__ (int val) { return val != garbage_val ? 1 : 0; });
    raft::common::nvtx::pop_range();
    raft::common::nvtx::push_range("Exclusive Scan");       
    thrust::exclusive_scan(thrust::cuda::par.on(stream), predicate.begin(), predicate.end(), predicate.begin(), 0);
    raft::common::nvtx::pop_range();

    raft::common::nvtx::push_range("Scatter");
    thrust::scatter_if(thrust::cuda::par.on(stream),
                       d_buffer_copy.begin(), d_buffer_copy.end(),
                       predicate.begin(),
                       d_buffer_copy.begin(),
                       d_buffer.begin(),
                       [garbage_val] __device__ (int val) { return val != garbage_val; });
    raft::common::nvtx::pop_range();

    // Transformer les valeurs de `d_buffer` en fonction de l'index modulo 4
    raft::common::nvtx::push_range("Apply Map Kernel");
    thrust::transform(thrust::cuda::par.on(stream),
                      thrust::make_zip_iterator(thrust::make_tuple(d_buffer.begin(), thrust::counting_iterator<int>(0))),
                      thrust::make_zip_iterator(thrust::make_tuple(d_buffer.end(), thrust::counting_iterator<int>(image_size))),
                      d_buffer.begin(),
                      transform_op());
    raft::common::nvtx::pop_range();

    // Allocation et calcul de l'histogramme avec CUB
    thrust::device_vector<int> histogram(num_bins, 0);
    int *d_histogram = thrust::raw_pointer_cast(histogram.data());

    size_t temp_storage_bytes = 0;
    raft::common::nvtx::push_range("Histogram Kernel");
    cub::DeviceHistogram::HistogramEven(nullptr, temp_storage_bytes, thrust::raw_pointer_cast(d_buffer.data()), d_histogram, num_bins, min_val, max_val + 1, image_size, stream);
    thrust::device_vector<char> temp_storage(temp_storage_bytes);
    cub::DeviceHistogram::HistogramEven(thrust::raw_pointer_cast(temp_storage.data()), temp_storage_bytes, thrust::raw_pointer_cast(d_buffer.data()), d_histogram, num_bins, min_val, max_val + 1, image_size, stream);
    raft::common::nvtx::pop_range();

    // Calcul du CDF pour l'égalisation
    thrust::device_vector<float> cdf(num_bins);
    raft::common::nvtx::push_range("Inclusive Scan");
    thrust::inclusive_scan(thrust::cuda::par.on(stream), histogram.begin(), histogram.end(), cdf.begin());
    raft::common::nvtx::pop_range();

    float min_cdf;
    CUDA_CHECK_ERROR(cudaMemcpy(&min_cdf, thrust::raw_pointer_cast(cdf.data()), sizeof(float), cudaMemcpyDeviceToHost));
    raft::common::nvtx::push_range("Histogram Equalize");
    thrust::transform(thrust::cuda::par.on(stream), cdf.begin(), cdf.end(), cdf.begin(), [min_cdf, image_size] __device__ (float c) {
        return (c - min_cdf) / (image_size - min_cdf) * 255.0f;
    });

    thrust::transform(thrust::cuda::par.on(stream), d_buffer.begin(), d_buffer.end(), d_buffer.begin(), [d_cdf = thrust::raw_pointer_cast(cdf.data())] __device__ (int val) {
        return static_cast<int>(d_cdf[val]);
    });
    raft::common::nvtx::pop_range();

    CUDA_CHECK_ERROR(cudaMemcpy(to_fix.buffer, thrust::raw_pointer_cast(d_buffer.data()), image_size * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaStreamDestroy(stream));
}
