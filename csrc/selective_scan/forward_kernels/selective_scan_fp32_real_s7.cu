#include "../selective_scan.cuh"

template void selective_scan_fwd_cuda<float, float,  DiscretizationMethod::S7>(SSMParamsBase &params, cudaStream_t stream);
