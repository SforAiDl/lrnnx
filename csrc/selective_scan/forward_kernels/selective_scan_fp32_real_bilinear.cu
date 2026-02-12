#include "../selective_scan.cuh"

template void selective_scan_fwd_cuda<float, float,  DiscretizationMethod::BILINEAR>(SSMParamsBase  &params, cudaStream_t stream);
