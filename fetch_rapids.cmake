# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/TP_REDUCE_RAPIDS.cmake)
  file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-24.08/RAPIDS.cmake
      ${CMAKE_CURRENT_BINARY_DIR}/TP_REDUCE_RAPIDS.cmake
  )
endif()
include(${CMAKE_CURRENT_BINARY_DIR}/TP_REDUCE_RAPIDS.cmake)
