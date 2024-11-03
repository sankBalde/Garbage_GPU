#include "image.hh"
#include "pipeline.hh"
#include "fix_cpu.cuh"
#include "cuda_tools/main_helper.hh"

#include <vector>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <filesystem>
#include <numeric>

#include "fix_gpu_hand.cuh"
// #include "utils.cuh"

#include <raft/common/nvtx.hpp>

#include <rmm/mr/device/pool_memory_resource.hpp>

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[])
{

    raft::common::nvtx::range fun_scope("Main Function");
    // RMM Setup
    raft::common::nvtx::push_range("RMM Setup");
    auto memory_resource = make_pool();
    rmm::mr::set_current_device_resource(memory_resource.get());
    raft::common::nvtx::pop_range();

    // -- Pipeline initialization

    std::cout << "File loading..." << std::endl;

    // - Get file paths

    raft::common::nvtx::push_range("Get file paths");
    using recursive_directory_iterator = std::filesystem::recursive_directory_iterator;
    std::vector<std::string> filepaths;
    for (const auto& dir_entry : recursive_directory_iterator("../images"))
        filepaths.emplace_back(dir_entry.path());
    raft::common::nvtx::pop_range();
    // - Init pipeline object
    raft::common::nvtx::push_range("Init pipeline object");
    Pipeline pipeline(filepaths);

    // -- Main loop containing image retring from pipeline and fixing

    const int nb_images = pipeline.images.size();
    std::vector<Image> images(nb_images);

    // - One CPU thread is launched for each image
    raft::common::nvtx::pop_range();
    std::cout << "Done, starting compute" << std::endl;

    #pragma omp parallel for
    for (int i = 0; i < nb_images; ++i)
    {
        // TODO : make it GPU compatible (aka faster)
        // You will need to copy images one by one on the GPU
        // You can store the images the way you want on the GPU
        // But you should treat the pipeline as a pipeline :
        // You *must not* copy all the images and only then do the computations
        // You must get the image from the pipeline as they arrive and launch computations right away
        // There are still ways to speeds this process of course
        raft::common::nvtx::push_range("Image processing");
        images[i] = pipeline.get_image(i);
        fix_image_gpu_hand(images[i]);
        raft::common::nvtx::pop_range();
    }

    std::cout << "Done with compute, starting stats" << std::endl;

    // -- All images are now fixed : compute stats (total then sort)

    // - First compute the total of each image

    // TODO : make it GPU compatible (aka faster)
    // You can use multiple CPU threads for your GPU version using openmp or not
    // Up to you :)
    #pragma omp parallel for
    for (int i = 0; i < nb_images; ++i)
    {
        auto& image = images[i];
        const int image_size = image.width * image.height;
        // image.to_sort.total = std::reduce(image.buffer, image.buffer + image_size, 0);

        // Initialiser un buffer sur le GPU pour chaque image
        rmm::device_uvector<int> buffer(image_size, rmm::cuda_stream_default);
        rmm::device_scalar<int> total(0, rmm::cuda_stream_default);

        // Copier les données de l'image du CPU vers le GPU
        raft::common::nvtx::push_range("Copy Image Buffer to GPU");
        cudaMemcpy(buffer.data(), image.buffer, image_size * sizeof(int), cudaMemcpyHostToDevice);
        raft::common::nvtx::pop_range();

        // TODO: Remplir le buffer avec l'image
        raft::common::nvtx::push_range("Reduce");
        your_reduce(buffer, total);
        raft::common::nvtx::pop_range();

        // Copier le résultat total du GPU vers le CPU
        int result;
        cudaMemcpy(&result, total.data(), sizeof(int), cudaMemcpyDeviceToHost);

        // Assigner le résultat dans l'image
        image.to_sort.total = result;

    }

    // - All totals are known, sort images accordingly (OPTIONAL)
    // Moving the actual images is too expensive, sort image indices instead
    // Copying to an id array and sort it instead

    // TODO OPTIONAL : for you GPU version you can store it the way you want
    // But just like the CPU version, moving the actual images while sorting will be too slow
    using ToSort = Image::ToSort;
    std::vector<ToSort> to_sort(nb_images);
    std::generate(to_sort.begin(), to_sort.end(), [n = 0, images] () mutable
    {
        return images[n++].to_sort;
    });

    // TODO OPTIONAL : make it GPU compatible (aka faster)
    std::sort(to_sort.begin(), to_sort.end(), [](ToSort a, ToSort b) {
        return a.total < b.total;
    });

    // TODO : Test here that you have the same results
    // You can compare visually and should compare image vectors values and "total" values
    // If you did the sorting, check that the ids are in the same order
    for (int i = 0; i < nb_images; ++i)
    {
        std::cout << "Image #" << images[i].to_sort.id << " total : " << images[i].to_sort.total << std::endl;
        std::ostringstream oss;
        oss << "Image#" << images[i].to_sort.id << ".pgm";
        std::string str = oss.str();
        images[i].write(str);
    }

    std::cout << "Done, the internet is safe now :)" << std::endl;

    // Cleaning
    // TODO : Don't forget to update this if you change allocation style
    for (int i = 0; i < nb_images; ++i)
        free(images[i].buffer);

    return 0;
}
