#pragma once

#include <rmm/device_uvector.hpp>
#include <rmm/device_scalar.hpp>
#include "image.hh"

void fix_image_gpu_industrial(Image& to_fix);