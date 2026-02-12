#include "selective_scan.h"
#include <torch/python.h>
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwd", &selective_scan_fwd, "Selective scan forward");
    m.def("bwd", &selective_scan_bwd, "Selective scan backward");
}
