#include "fix_gpu_hand.cuh"
#include "utils.cuh"
#include <raft/common/nvtx.hpp>

void fix_image_gpu_hand_old(Image& to_fix)
{
    const int image_size = to_fix.width * to_fix.height;
    int block_size = 256;
    int grid_size_non_garbage = (image_size + block_size - 1) / block_size;
    int grid_size_avec_garbage = (to_fix.size() + block_size - 1) / block_size;

    // Allocation sur GPU pour d_buffer
    rmm::device_uvector<int> d_buffer(to_fix.size(), rmm::cuda_stream_default);

    CUDA_CHECK_ERROR(cudaMemcpy(d_buffer.data(), to_fix.buffer, to_fix.size() * sizeof(int), cudaMemcpyHostToDevice));

    // Allocation sur GPU pour predicate
    rmm::device_uvector<int> predicate(to_fix.size(), d_buffer.stream());

    constexpr int garbage_val = -27;

    // Lancement du kernel avec les données sur GPU
    build_predicate_kernel<<<grid_size_avec_garbage, block_size, 0, d_buffer.stream()>>>(
        raft::device_span<int>(d_buffer.data(), d_buffer.size()),
        raft::device_span<int>(predicate.data(), predicate.size()),
        garbage_val, to_fix.size());
    CUDA_CHECK_ERROR(cudaGetLastError());


    CUDA_CHECK_ERROR(cudaStreamSynchronize(d_buffer.stream()));

    // Appel de your_scan pour effectuer un scan exclusif
    your_scan(predicate, true);

    CUDA_CHECK_ERROR(cudaStreamSynchronize(d_buffer.stream()));

    // Lancement du kernel de scatter
    scatter_kernel<<<grid_size_avec_garbage, block_size, 0, d_buffer.stream()>>>(
        raft::device_span<int>(d_buffer.data(), d_buffer.size()),
        raft::device_span<int>(predicate.data(), predicate.size()),
        to_fix.size());

    CUDA_CHECK_ERROR(cudaStreamSynchronize(d_buffer.stream()));

     apply_map_kernel<<<grid_size_non_garbage, block_size, 0, d_buffer.stream()>>>(
        raft::device_span<int>(d_buffer.data(), d_buffer.size()),
         image_size);
    CUDA_CHECK_ERROR(cudaStreamSynchronize(d_buffer.stream()));

    // #3 Histogram equalization

    // Histogram

    //! GPU

    // // Allocation pour l'histogramme et le CDF
    rmm::device_uvector<int> histogram(256, d_buffer.stream());
    rmm::device_uvector<int> cdf(256, rmm::cuda_stream_default);

    // // Lancement du kernel pour calculer l'histogramme
     histogram_kernel<<<grid_size_avec_garbage, block_size, 0, d_buffer.stream()>>>(
         raft::device_span<int>(d_buffer.data(), d_buffer.size()),
         raft::device_span<int>(histogram.data(), histogram.size()),
         image_size);

    CUDA_CHECK_ERROR(cudaStreamSynchronize(histogram.stream()));
    CUDA_CHECK_ERROR(cudaStreamSynchronize(d_buffer.stream()));

    your_scan(histogram, false);

    CUDA_CHECK_ERROR(cudaStreamSynchronize(d_buffer.stream()));

    std::vector<int> histogram_host(256);
    CUDA_CHECK_ERROR(cudaMemcpy(histogram_host.data(), histogram.data(), histogram.size() * sizeof(int), cudaMemcpyDeviceToHost));

    // // Trouver le premier élément non nul dans l'histogramme
     int cdf_min = 0;
     for (int i = 1; i < 256; ++i)
     {
         if (histogram_host[i] != 0)
         {
             cdf_min = histogram_host[i];
             break;
         }
     }

    // // Appliquer l'égalisation de l'histogramme
     equalize_kernel<<<grid_size_non_garbage, block_size, 0, d_buffer.stream()>>>(
         raft::device_span<int>(d_buffer.data(), d_buffer.size()),
         raft::device_span<int>(d_buffer.data(), d_buffer.size()),  // Réutilisation de d_buffer pour stocker le résultat
         raft::device_span<int>(histogram.data(), histogram.size()),
         cdf_min, image_size);

    // // Synchronisation pour assurer la fin de l'exécution
   CUDA_CHECK_ERROR(cudaStreamSynchronize(d_buffer.stream()));
   CUDA_CHECK_ERROR(cudaMemcpy(to_fix.buffer, d_buffer.data(), image_size * sizeof(int), cudaMemcpyDeviceToHost));
}


void fix_image_gpu_hand(Image& to_fix)
{
    raft::common::nvtx::range fun_scope("Fix Image GPU Hand");
    const int image_size = to_fix.width * to_fix.height;
    int block_size = 256;
    int grid_size_non_garbage = (image_size + block_size - 1) / block_size;
    int grid_size_avec_garbage = (to_fix.size() + block_size - 1) / block_size;

    rmm::device_uvector<int> d_buffer(to_fix.size(), rmm::cuda_stream_default);

    // Copie to_fix sur GPU
    CUDA_CHECK_ERROR(cudaMemcpy(d_buffer.data(), to_fix.buffer, to_fix.size() * sizeof(int), cudaMemcpyHostToDevice));

    rmm::device_uvector<int> predicate(to_fix.size(), d_buffer.stream());
    
//! 1 - GARBAGE VALUES (-27)

    // Initialiser predicate à zéro
    CUDA_CHECK_ERROR(cudaMemset(predicate.data(), 0, predicate.size() * sizeof(int)));

    constexpr int garbage_val = -27;
    raft::common::nvtx::push_range("Build Predicate Kernel");
    build_predicate_kernel<<<grid_size_avec_garbage, block_size, 0, d_buffer.stream()>>>(
        raft::device_span<int>(d_buffer.data(), d_buffer.size()),
        raft::device_span<int>(predicate.data(), predicate.size()),
        garbage_val, to_fix.size());
    CUDA_CHECK_ERROR(cudaGetLastError());
    CUDA_CHECK_ERROR(cudaStreamSynchronize(d_buffer.stream()));
    raft::common::nvtx::pop_range();

    // Scan exclusif
    raft::common::nvtx::push_range("Scan Exclusif Predicate");
    your_scan(predicate, true);
    raft::common::nvtx::pop_range();

    raft::common::nvtx::push_range("Scatter Kernel");
    scatter_kernel<<<grid_size_avec_garbage, block_size, 0, d_buffer.stream()>>>(
        raft::device_span<int>(d_buffer.data(), d_buffer.size()),
        raft::device_span<int>(predicate.data(), predicate.size()),
        to_fix.size());
    CUDA_CHECK_ERROR(cudaStreamSynchronize(d_buffer.stream()));
    raft::common::nvtx::pop_range();

//! 2 - MAP
    raft::common::nvtx::push_range("Apply Map Kernel");
    apply_map_kernel<<<grid_size_non_garbage, block_size, 0, d_buffer.stream()>>>(
        raft::device_span<int>(d_buffer.data(), d_buffer.size()),
        image_size);
    CUDA_CHECK_ERROR(cudaStreamSynchronize(d_buffer.stream()));
    raft::common::nvtx::pop_range();

//! 3 - HISTOGRAM EGALISATION

    rmm::device_uvector<int> histogram(256, d_buffer.stream());
    rmm::device_uvector<int> cdf(256, rmm::cuda_stream_default);

    // Initialise

    CUDA_CHECK_ERROR(cudaMemset(histogram.data(), 0, histogram.size() * sizeof(int)));
    const int histogram_sharedMem = 256 * sizeof(int);

    raft::common::nvtx::push_range("Histogram Kernel");
    histogram_kernel<<<grid_size_avec_garbage, block_size, histogram_sharedMem, d_buffer.stream()>>>(
        raft::device_span<int>(d_buffer.data(), d_buffer.size()),
        raft::device_span<int>(histogram.data(), histogram.size()),
        image_size);

    CUDA_CHECK_ERROR(cudaStreamSynchronize(d_buffer.stream()));
    raft::common::nvtx::pop_range();

    raft::common::nvtx::push_range("Scan Inclusif Kernel");
    your_scan(histogram, false);    // Scan inclusif
    raft::common::nvtx::pop_range();

    std::vector<int> histogram_host(256);
    CUDA_CHECK_ERROR(cudaMemcpy(histogram_host.data(), histogram.data(), histogram.size() * sizeof(int), cudaMemcpyDeviceToHost));

    auto first_none_zero = std::find_if(histogram_host.begin(), histogram_host.end(), [](auto v) { return v != 0; });

    const int cdf_min = *first_none_zero;

    raft::common::nvtx::push_range("Equalize Kernel");
    equalize_kernel<<<grid_size_non_garbage, block_size, 0, d_buffer.stream()>>>(
        raft::device_span<int>(d_buffer.data(), d_buffer.size()),
        raft::device_span<int>(d_buffer.data(), d_buffer.size()),
        raft::device_span<int>(histogram.data(), histogram.size()),
        cdf_min, image_size);

    CUDA_CHECK_ERROR(cudaStreamSynchronize(d_buffer.stream()));
    raft::common::nvtx::pop_range();
    CUDA_CHECK_ERROR(cudaMemcpy(to_fix.buffer, d_buffer.data(), image_size * sizeof(int), cudaMemcpyDeviceToHost));
}


void your_reduce(rmm::device_uvector<int>& buffer,
                 rmm::device_scalar<int>& total)
{
    constexpr int blocksize = 256;
    int gridsize = (buffer.size() + blocksize - 1) / blocksize;

    int shared_memory_size = blocksize * sizeof(int);

    if (gridsize == 1) {
        kernel_your_reduce_grid_stride_loop<int><<<gridsize, blocksize, shared_memory_size, buffer.stream()>>>(
            raft::device_span<const int>(buffer.data(), buffer.size()),
            raft::device_span<int>(total.data(), 1));
        CUDA_CHECK_ERROR(cudaStreamSynchronize(buffer.stream()));
        return;
    }

//! Si on a plus de 64 valeurs

    rmm::device_uvector<int> partial_sums(gridsize, buffer.stream());
    rmm::device_uvector<int> partial_sums_bis(gridsize, buffer.stream());

    kernel_your_reduce_grid_stride_loop<int><<<gridsize, blocksize, shared_memory_size, buffer.stream()>>>(
        raft::device_span<const int>(buffer.data(), buffer.size()),
        raft::device_span<int>(partial_sums.data(), partial_sums.size()));
    CUDA_CHECK_ERROR(cudaStreamSynchronize(buffer.stream()));

    int cpt = 0;
    
    while (gridsize > 1) {

        gridsize = (gridsize + blocksize - 1) / blocksize;

        if (cpt % 2 == 0) {
            kernel_your_reduce_grid_stride_loop<int><<<gridsize, blocksize, shared_memory_size, buffer.stream()>>>(
                raft::device_span<const int>(partial_sums.data(), partial_sums.size()),
                raft::device_span<int>(partial_sums_bis.data(), partial_sums_bis.size()));
            CUDA_CHECK_ERROR(cudaStreamSynchronize(buffer.stream()));

            partial_sums_bis.resize(gridsize, buffer.stream());
        } else {
            kernel_your_reduce_grid_stride_loop<int><<<gridsize, blocksize, shared_memory_size, buffer.stream()>>>(
                raft::device_span<const int>(partial_sums_bis.data(), partial_sums_bis.size()),
                raft::device_span<int>(partial_sums.data(), partial_sums.size()));
            CUDA_CHECK_ERROR(cudaStreamSynchronize(buffer.stream()));

            partial_sums.resize(gridsize, buffer.stream());
        }

        cpt += 1;
    }

    kernel_your_reduce_grid_stride_loop<int><<<1, blocksize, shared_memory_size, buffer.stream()>>>(
        raft::device_span<int>(partial_sums.data(), partial_sums.size()),
        raft::device_span<int>(total.data(), 1));
    CUDA_CHECK_ERROR(cudaStreamSynchronize(buffer.stream()));
}