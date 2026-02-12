#include "../selective_scan_backward.cuh"

template void selective_scan_bwd_cuda<float, complex_t, DiscretizationMethod::MAMBA>(SSMParamsBwd &params, cudaStream_t stream);
