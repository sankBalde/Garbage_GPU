#pragma once

#include <cmath>
#include <tuple>

#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <raft/core/handle.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

template <typename Tuple>
constexpr auto tuple_length(Tuple)
{
    return std::tuple_size_v<Tuple>;
}

auto make_async()
{
    return std::make_shared<rmm::mr::cuda_async_memory_resource>();
}
auto make_pool()
{
    // Allocate 0.05 Go
    size_t initial_pool_size = std::pow(2, 26);
    return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(
        make_async(),
        initial_pool_size);
}

bool parse_arguments(int argc, char* argv[])
{
    bool bench_nsight = false;
    for (int i = 1; i < argc; i++)
    {
        if (argv[i] == std::string_view("--no-check"))
        {
            //Fixture::no_check = true;
            std::swap(argv[i], argv[--argc]);
        }
        // Set iteration number to 1 not to mess with nsight
        if (argv[i] == std::string_view("--bench-nsight"))
        {
            bench_nsight = true;
            std::swap(argv[i], argv[--argc]);
        }
    }

    return bench_nsight;
}
