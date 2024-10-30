#include "fix_gpu_industrial.cuh"
#include "utils.cuh"


bool check_predicate(const rmm::device_uvector<int>& predicate_gpu, const std::vector<int>& predicate_cpu) {
    // Vérifier que les tailles des deux vecteurs sont identiques
    if (predicate_gpu.size() != predicate_cpu.size()) {
        std::cerr << "Size mismatch: GPU predicate size (" << predicate_gpu.size()
                  << ") vs CPU predicate size (" << predicate_cpu.size() << ")" << std::endl;
        return false;
    }

    // Créer un vecteur hôte pour recevoir les données du GPU
    std::vector<int> predicate_gpu_host(predicate_gpu.size());

    // Copier les données du GPU vers le vecteur hôte
    CUDA_CHECK_ERROR(cudaMemcpy(predicate_gpu_host.data(), predicate_gpu.data(),
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



void fix_image_gpu_industrial(Image& to_fix)
{
    const int image_size = to_fix.width * to_fix.height;
    int block_size = 256;
    int grid_size_non_garbage = (image_size + block_size - 1) / block_size;
    int grid_size_avec_garbage = (to_fix.size() + block_size - 1) / block_size;

    // Allocation sur GPU pour d_buffer
    rmm::device_uvector<int> d_buffer(to_fix.size(), rmm::cuda_stream_default);

    // Si to_fix.buffer est en mémoire hôte (CPU), il faut le copier sur le GPU.
    // Si to_fix.buffer est déjà sur le GPU, tu peux le faire directement.
    CUDA_CHECK_ERROR(cudaMemcpy(d_buffer.data(), to_fix.buffer, to_fix.size() * sizeof(int), cudaMemcpyHostToDevice));

    // Allocation sur GPU pour predicate
    rmm::device_uvector<int> predicate(to_fix.size(), d_buffer.stream());
    CUDA_CHECK_ERROR(cudaMemset(predicate.data(), 0, predicate.size() * sizeof(int)));
    constexpr int garbage_val = -27;

    // Lancement du kernel avec les données sur GPU
    build_predicate_kernel<<<grid_size_avec_garbage, block_size, 0, d_buffer.stream()>>>(
        raft::device_span<int>(d_buffer.data(), d_buffer.size()),
        raft::device_span<int>(predicate.data(), predicate.size()),
        garbage_val, to_fix.size());    // TODO : Missmatch
    CUDA_CHECK_ERROR(cudaGetLastError());


    CUDA_CHECK_ERROR(cudaStreamSynchronize(d_buffer.stream()));

    /*std::vector<int> predicate_CPU(to_fix.size(), 0);

    for (int i = 0; i < to_fix.size(); ++i) {
        if (to_fix.buffer[i] != garbage_val)
            predicate_CPU[i] = 1;
    }*/



    // Synchronisation pour assurer la fin de l'exécution

    // Appel de your_scan pour effectuer un scan exclusif
    your_scan(predicate, true);
    //std::inclusive_scan(predicate_CPU.begin(), predicate_CPU.end(), predicate_CPU.begin(), 0);

    //check_predicate(predicate, predicate_CPU);

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



    //! Mettre le d_buffer dans le to_fix -> Format CPU
    //CUDA_CHECK_ERROR(cudaMemcpy(to_fix.buffer, d_buffer.data(), image_size * sizeof(int), cudaMemcpyDeviceToHost));


    // #3 Histogram equalization

    // Histogram


    //! GPU

    // // Allocation pour l'histogramme et le CDF
    rmm::device_uvector<int> histogram(256, d_buffer.stream());
    rmm::device_uvector<int> cdf(256, rmm::cuda_stream_default);

    //CUDA_CHECK_ERROR(cudaMemset(histogram.data(), 0, histogram.size() * sizeof(int)));
    //CUDA_CHECK_ERROR(cudaMemset(cdf.data(), 0, cdf.size() * sizeof(int)));

    // // Lancement du kernel pour calculer l'histogramme
     histogram_kernel<<<grid_size_avec_garbage, block_size, 0, d_buffer.stream()>>>(
         raft::device_span<int>(d_buffer.data(), d_buffer.size()),
         raft::device_span<int>(histogram.data(), histogram.size()),
         image_size);
    CUDA_CHECK_ERROR(cudaStreamSynchronize(histogram.stream()));
    CUDA_CHECK_ERROR(cudaStreamSynchronize(d_buffer.stream()));
    your_scan(histogram, false);
    CUDA_CHECK_ERROR(cudaStreamSynchronize(d_buffer.stream()));

    // // Trouver le premier élément non nul dans le CDF
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
