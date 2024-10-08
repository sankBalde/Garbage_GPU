function(find_and_configure_rmm)
    include(${rapids-cmake-dir}/cpm/rmm.cmake)
endfunction()

find_and_configure_rmm()
