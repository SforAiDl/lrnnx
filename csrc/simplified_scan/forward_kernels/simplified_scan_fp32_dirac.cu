#include "../simplified_scan.cuh"

template void simplified_scan_fwd_cuda<DiscretizationMethod::DIRAC>(SSMParamsBase &params, cudaStream_t stream);
