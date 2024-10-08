# Use CPM to find or clone CCCL
function(find_and_configure_cccl)
        include(${rapids-cmake-dir}/cpm/cccl.cmake)
endfunction()

find_and_configure_cccl()
