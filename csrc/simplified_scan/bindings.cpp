#include "simplified_scan.h"
#include <torch/python.h>
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwd", &simplified_scan_fwd, "Simplified scan forward");
    m.def("bwd", &simplified_scan_bwd, "Simplified scan backward");
}
