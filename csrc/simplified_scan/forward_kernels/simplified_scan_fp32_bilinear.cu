#include "../simplified_scan.cuh"

template void simplified_scan_fwd_cuda<DiscretizationMethod::BILINEAR>(SSMParamsBase &params, cudaStream_t stream);
