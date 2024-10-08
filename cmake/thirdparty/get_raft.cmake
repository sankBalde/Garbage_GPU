function(find_and_configure_raft)
    set(oneValueArgs VERSION FORK PINNED_TAG CLONE_ON_PIN)
    cmake_parse_arguments(PKG "" "${oneValueArgs}" "" ${ARGN})

    rapids_cpm_find(raft ${PKG_VERSION}
        GLOBAL_TARGETS raft::raft
        CPM_ARGS
        GIT_REPOSITORY https://github.com/${PKG_FORK}/raft.git
        GIT_TAG ${PKG_PINNED_TAG}
        SOURCE_SUBDIR cpp
        OPTIONS
            "BUILD_TESTS OFF"
            "BUILD_BENCH OFF"
            "RAFT_COMPILE_LIBRARY OFF"
    )

    if(raft_ADDED)
        message(VERBOSE "Using RAFT located in ${raft_SOURCE_DIR}")
    else()
        message(VERBOSE "Using RAFT located in ${raft_DIR}")
    endif()
endfunction()

# Change pinned tag and fork here to test a commit in CI
# To use a different RAFT locally, set the CMake variable
# RPM_raft_SOURCE=/path/to/local/raft
find_and_configure_raft(VERSION 24.08
    FORK rapidsai
    PINNED_TAG branch-24.08
    CLONE_ON_PIN ON
)
