#include "../simplified_scan.cuh"

template void simplified_scan_fwd_cuda<DiscretizationMethod::ZOH>(SSMParamsBase &params, cudaStream_t stream);
