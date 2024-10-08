#pragma once

#include <rmm/device_uvector.hpp>
#include <rmm/device_scalar.hpp>
#include "image.hh"

void fix_image_gpu_hand(Image& to_fix);