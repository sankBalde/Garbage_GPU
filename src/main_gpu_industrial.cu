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

#include "fix_gpu_industrial.cuh"
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

#include <rmm/mr/device/pool_memory_resource.hpp>

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[])
{

    // RMM Setup
    auto memory_resource = make_pool();
    rmm::mr::set_current_device_resource(memory_resource.get());

    // -- Pipeline initialization

    std::cout << "File loading..." << std::endl;

    // - Get file paths

    using recursive_directory_iterator = std::filesystem::recursive_directory_iterator;
    std::vector<std::string> filepaths;
    for (const auto& dir_entry : recursive_directory_iterator("../images"))
        filepaths.emplace_back(dir_entry.path());

    // - Init pipeline object

    Pipeline pipeline(filepaths);

    // -- Main loop containing image retring from pipeline and fixing

    const int nb_images = pipeline.images.size();
    std::vector<Image> images(nb_images);

    // - One CPU thread is launched for each image

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
        images[i] = pipeline.get_image(i);
        fix_image_gpu_industrial(images[i]);
    }

    std::cout << "Done with compute, starting stats" << std::endl;

    // -- All images are now fixed : compute stats (total then sort)

    // - First compute the total of each image

    // TODO : make it GPU compatible (aka faster)
    // You can use multiple CPU threads for your GPU version using openmp or not
    // Up to you :)
    thrust::device_vector<int> totals(nb_images);
    #pragma omp parallel for
    for (int i = 0; i < nb_images; ++i)
    {
        const int image_size = images[i].width * images[i].height;

        thrust::device_vector<int> d_image(images[i].buffer, images[i].buffer + images[i].size());
        totals[i] = thrust::reduce(thrust::device, d_image.begin(), d_image.end(), 0);
    }

    // Tri des images par total
    thrust::device_vector<int> ids(nb_images);
    thrust::sequence(ids.begin(), ids.end());
    thrust::sort_by_key(totals.begin(), totals.end(), ids.begin());

    // Affichage des résultats
    for (int i = 0; i < nb_images; ++i)
    {
        int sorted_id = ids[i];
        std::cout << "Image #" << images[sorted_id].to_sort.id << " total : " << totals[sorted_id] << std::endl;
        std::ostringstream oss;
        oss << "Image#" << images[sorted_id].to_sort.id << ".pgm";
        images[sorted_id].write(oss.str());
    }

    std::cout << "Done, the internet is safe now :)" << std::endl;

    return 0;
}
