#include "../selective_scan.cuh"

template void selective_scan_fwd_cuda<float, float,  DiscretizationMethod::RGLRU>(SSMParamsBase &params, cudaStream_t stream);
