#include "../selective_scan.cuh"

template void selective_scan_fwd_cuda<float, float,  DiscretizationMethod::DIRAC>(SSMParamsBase &params, cudaStream_t stream);
