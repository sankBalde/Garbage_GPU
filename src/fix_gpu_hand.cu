#include "fix_gpu_hand.cuh"
#include "utils.cuh"


void fix_image_gpu_hand(Image& to_fix)
{
    const int image_size = to_fix.width * to_fix.height;
    int block_size = 256;
    int grid_size_non_garbage = (image_size + block_size - 1) / block_size;
    int grid_size_avec_garbage = (to_fix.size() + block_size - 1) / block_size;

    // Allocation sur GPU pour d_buffer
    rmm::device_uvector<int> d_buffer(image_size, rmm::cuda_stream_default);

    // Si to_fix.buffer est en mémoire hôte (CPU), il faut le copier sur le GPU.
    // Si to_fix.buffer est déjà sur le GPU, tu peux le faire directement.
    CUDA_CHECK_ERROR(cudaMemcpy(d_buffer.data(), to_fix.buffer, image_size * sizeof(int), cudaMemcpyHostToDevice));

    // Allocation sur GPU pour predicate
    rmm::device_uvector<int> predicate(to_fix.size(), rmm::cuda_stream_default);

    // Lancement du kernel avec les données sur GPU
    build_predicate_kernel<<<grid_size_avec_garbage, block_size, 0, d_buffer.stream()>>>(
        raft::device_span<int>(d_buffer.data(), d_buffer.size()),
        raft::device_span<int>(predicate.data(), predicate.size()),
        -27, to_fix.size());

    // Synchronisation pour assurer la fin de l'exécution
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
    // Allocation pour l'histogramme et le CDF
    rmm::device_uvector<int> histogram(256, d_buffer.stream());
    //rmm::device_uvector<int> cdf(256, rmm::cuda_stream_default);

    // Lancement du kernel pour calculer l'histogramme
    histogram_kernel<<<grid_size_avec_garbage, block_size, 0, d_buffer.stream()>>>(
        raft::device_span<int>(d_buffer.data(), d_buffer.size()),
        raft::device_span<int>(histogram.data(), histogram.size()),
        image_size);

    your_scan(histogram, false);


    // Trouver le premier élément non nul dans le CDF
    std::vector<int> histogram_host(256);
    CUDA_CHECK_ERROR(cudaMemcpy(histogram_host.data(), histogram.data(), histogram.size() * sizeof(int), cudaMemcpyDeviceToHost));

    // Trouver le premier élément non nul dans l'histogramme
    int cdf_min = 0;
    for (int i = 1; i < 256; ++i)
    {
        if (histogram_host[i] != 0)
        {
            cdf_min = histogram_host[i];
            break;
        }
    }

    // Appliquer l'égalisation de l'histogramme
    equalize_kernel<<<grid_size_non_garbage, block_size, 0, d_buffer.stream()>>>(
        raft::device_span<int>(d_buffer.data(), d_buffer.size()),
        raft::device_span<int>(d_buffer.data(), d_buffer.size()),  // Réutilisation de d_buffer pour stocker le résultat
        raft::device_span<int>(histogram.data(), histogram.size()),
        cdf_min, image_size);

    // Synchronisation pour assurer la fin de l'exécution
    CUDA_CHECK_ERROR(cudaStreamSynchronize(d_buffer.stream()));
    CUDA_CHECK_ERROR(cudaMemcpy(to_fix.buffer, d_buffer.data(), image_size * sizeof(int), cudaMemcpyDeviceToHost));
}
