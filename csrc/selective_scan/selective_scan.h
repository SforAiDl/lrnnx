
// Lambda trick to select pre complied paths 
// Refs: https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h
// and https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h

#pragma once

#include "../common.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
// Discretization methods for selective scan
////////////////////////////////////////////////////////////////////////////////////////////////////

enum class DiscretizationMethod {
    ZOH,      
    BILINEAR,
    DIRAC,
    MAMBA,
    S7,
    RGLRU
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct SSMParamsBase {
    using index_t = uint32_t;

    int batch, dim, seqlen, dstate, n_groups, n_chunks;
    int dim_ngroups_ratio;
    bool is_variable_B;
    bool is_variable_C;
    bool is_complex_input;

    bool delta_softplus;

    index_t A_d_stride;
    index_t A_dstate_stride;
    index_t B_batch_stride;
    index_t B_d_stride;
    index_t B_dstate_stride;
    index_t B_group_stride;
    index_t C_batch_stride;
    index_t C_d_stride;
    index_t C_dstate_stride;
    index_t C_group_stride;
    index_t u_batch_stride;
    index_t u_d_stride;
    index_t delta_batch_stride;
    index_t delta_d_stride;
    index_t deltaA_batch_stride;
    index_t deltaA_d_stride;
    index_t z_batch_stride;
    index_t z_d_stride;
    index_t out_batch_stride;
    index_t out_d_stride;
    index_t out_z_batch_stride;
    index_t out_z_d_stride;

    // common data pointers
    void *__restrict__ A_ptr;
    void *__restrict__ B_ptr;
    void *__restrict__ C_ptr;
    void *__restrict__ D_ptr;
    void *__restrict__ u_ptr;
    void *__restrict__ delta_ptr;
    void *__restrict__ deltaA_ptr;
    void *__restrict__ delta_bias_ptr;
    void *__restrict__ out_ptr;
    void *__restrict__ x_ptr;
    void *__restrict__ z_ptr;
    void *__restrict__ out_z_ptr;

    // discretization method
    DiscretizationMethod d_method;
};

struct SSMParamsBwd: public SSMParamsBase {
    index_t dout_batch_stride;
    index_t dout_d_stride;
    index_t dA_d_stride;
    index_t dA_dstate_stride;
    index_t dB_batch_stride;
    index_t dB_group_stride;
    index_t dB_d_stride;
    index_t dB_dstate_stride;
    index_t dC_batch_stride;
    index_t dC_group_stride;
    index_t dC_d_stride;
    index_t dC_dstate_stride;
    index_t du_batch_stride;
    index_t du_d_stride;
    index_t dz_batch_stride;
    index_t dz_d_stride;
    index_t ddelta_batch_stride;
    index_t ddelta_d_stride;
    index_t ddeltaA_batch_stride;
    index_t ddeltaA_d_stride;

    //  ommon data pointers
    void *__restrict__ dout_ptr;
    void *__restrict__ dA_ptr;
    void *__restrict__ dB_ptr;
    void *__restrict__ dC_ptr;
    void *__restrict__ dD_ptr;
    void *__restrict__ du_ptr;
    void *__restrict__ dz_ptr;
    void *__restrict__ ddelta_ptr;
    void *__restrict__ ddeltaA_ptr;
    void *__restrict__ ddelta_bias_ptr;
};

///////////////////////////////////////////////////////////////////////////////////////////
// exporting functions for python bindings
std::vector<at::Tensor> selective_scan_fwd(const at::Tensor &u, const at::Tensor &delta,
                  const at::Tensor &A, const at::Tensor &B, const at::Tensor &C,
                  const c10::optional<at::Tensor> &D_,
                  const c10::optional<at::Tensor> &z_,
                  const c10::optional<at::Tensor> &delta_bias_,
                  const c10::optional<at::Tensor> &deltaA_,
                  bool delta_softplus,
                  const std::string& discretization_method_str);

    
std::vector<at::Tensor> selective_scan_bwd(const at::Tensor &u, const at::Tensor &delta,
                  const at::Tensor &A, const at::Tensor &B, const at::Tensor &C,
                  const c10::optional<at::Tensor> &D_,
                  const c10::optional<at::Tensor> &z_,
                  const c10::optional<at::Tensor> &delta_bias_,
                  const c10::optional<at::Tensor> &deltaA_,
                  const at::Tensor &dout,
                  const c10::optional<at::Tensor> &x_,
                  const c10::optional<at::Tensor> &out_,
                  c10::optional<at::Tensor> &dz_,
                  bool delta_softplus,
                  bool recompute_out_z,
                  const std::string& discretization_method_str);

////////////////////////////////////////////////////////////////////////////////////////////////////////////
