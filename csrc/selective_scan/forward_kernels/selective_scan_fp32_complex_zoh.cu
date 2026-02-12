#include "../selective_scan.cuh"

template void selective_scan_fwd_cuda<float, complex_t, DiscretizationMethod::ZOH>(SSMParamsBase &params, cudaStream_t stream);
