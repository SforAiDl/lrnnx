#include "../selective_scan_backward.cuh"

template void selective_scan_bwd_cuda<float, float,  DiscretizationMethod::RGLRU>(SSMParamsBwd &params, cudaStream_t stream);
