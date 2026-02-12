#include "../simplified_scan_backward.cuh"

template void simplified_scan_bwd_cuda<DiscretizationMethod::BILINEAR>(SSMParamsBwd &params, cudaStream_t stream);
