#include "../simplified_scan_backward.cuh"

template void simplified_scan_bwd_cuda<DiscretizationMethod::ZOH>(SSMParamsBwd &params, cudaStream_t stream);
