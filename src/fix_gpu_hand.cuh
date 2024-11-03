#pragma once

#include <rmm/device_uvector.hpp>
#include <rmm/device_scalar.hpp>
#include "image.hh"

void fix_image_gpu_hand(Image& to_fix);
void your_reduce(rmm::device_uvector<int>& buffer,
                 rmm::device_scalar<int>& total);