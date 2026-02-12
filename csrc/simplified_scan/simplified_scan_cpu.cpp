/******************************************************************************
 * Simplified Scan CPU Interface
 * 
 * CPU-side interface for the simplified scan CUDA kernel.
 * This is the S5-style scan with complex-valued inputs.
 ******************************************************************************/

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/python.h>
#include <vector>

#include "simplified_scan.h"

#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")

#define DISPATCH_DISCRETIZATION(D_METHOD, CONST_NAME, ...)                          \
    if (D_METHOD == DiscretizationMethod::ZOH) {                                    \
        constexpr auto CONST_NAME = DiscretizationMethod::ZOH;                      \
        __VA_ARGS__();                                                              \
    } else if (D_METHOD == DiscretizationMethod::BILINEAR) {                        \
        constexpr auto CONST_NAME = DiscretizationMethod::BILINEAR;                 \
        __VA_ARGS__();                                                              \
    }  else if (D_METHOD == DiscretizationMethod::DIRAC) {                          \
        constexpr auto CONST_NAME = DiscretizationMethod::DIRAC;                    \
        __VA_ARGS__();                                                              \
    } else {                                                                        \
        AT_ERROR("Discretization method not implemented");                          \
    }



template<DiscretizationMethod DMethod>
void simplified_scan_fwd_cuda(SSMParamsBase &params, cudaStream_t stream);

template<DiscretizationMethod DMethod>
void simplified_scan_bwd_cuda(SSMParamsBwd &params, cudaStream_t stream);

void set_ssm_params_fwd(SSMParamsBase &params,
                        // sizes
                        const size_t batch,
                        const size_t dim,
                        const size_t seqlen,
                        const size_t dstate,
                        const size_t n_groups,
                        const size_t n_chunks,
                        // device pointers
                        const at::Tensor u,
                        const at::Tensor delta,
                        const at::Tensor A,
                        const at::Tensor B,
                        const at::Tensor C,
                        const at::Tensor out,
                        void* deltaA_ptr,
                        void* x_ptr,
                        DiscretizationMethod d_method) {

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.batch = batch;
    params.dim = dim;
    params.seqlen = seqlen;
    params.dstate = dstate;
    params.n_groups = n_groups;
    params.n_chunks = n_chunks;
    params.dim_ngroups_ratio = dim / n_groups;

    params.d_method = d_method;

    // Set the pointers and strides.
    params.u_ptr = u.data_ptr();
    params.delta_ptr = delta.data_ptr();
    params.A_ptr = A.data_ptr();
    params.B_ptr = B.data_ptr();
    params.C_ptr = C.data_ptr();
    params.out_ptr = out.data_ptr();
    params.x_ptr = x_ptr;
    // All stride are in elements, not bytes.
    params.A_d_stride = A.stride(0);
    params.A_dstate_stride = A.stride(1);
    params.B_d_stride = B.stride(0);
    params.B_dstate_stride = B.stride(1);
    params.C_d_stride = C.stride(0);
    params.C_dstate_stride = C.stride(1);
    params.u_batch_stride = u.stride(0);
    params.u_d_stride = u.stride(1);
    params.delta_batch_stride = delta.stride(0);
    params.delta_d_stride = delta.stride(1);
    // deltaA strides (same layout as delta)
    params.deltaA_ptr = deltaA_ptr;
    params.deltaA_batch_stride = delta.stride(0);  // Same layout as delta
    params.deltaA_d_stride = delta.stride(1);      // Same layout as delta
    params.out_batch_stride = out.stride(0);
    params.out_d_stride = out.stride(1);
}

void set_ssm_params_bwd(SSMParamsBwd &params,
                        // sizes
                        const size_t batch,
                        const size_t dim,
                        const size_t seqlen,
                        const size_t dstate,
                        const size_t n_groups,
                        const size_t n_chunks,
                        // device pointers
                        const at::Tensor u,
                        const at::Tensor delta,
                        const at::Tensor A,
                        const at::Tensor B,
                        const at::Tensor C,
                        const at::Tensor out,
                        void* deltaA_ptr,
                        void* x_ptr,
                        const at::Tensor dout,
                        const at::Tensor du,
                        const at::Tensor ddelta,
                        const at::Tensor ddeltaA,
                        const at::Tensor dA,
                        const at::Tensor dB,
                        const at::Tensor dC,
                        bool has_deltaA,
                        DiscretizationMethod d_method) {
    set_ssm_params_fwd(params, batch, dim, seqlen, dstate, n_groups, n_chunks,
                       u, delta, A, B, C, out, deltaA_ptr, x_ptr, d_method);

    // Set the pointers and strides.
    params.dout_ptr = dout.data_ptr();
    params.du_ptr = du.data_ptr();
    params.dA_ptr = dA.data_ptr();
    params.dB_ptr = dB.data_ptr();
    params.dC_ptr = dC.data_ptr();
    params.ddelta_ptr = ddelta.data_ptr();
    // All stride are in elements, not bytes.
    params.dout_batch_stride = dout.stride(0);
    params.dout_d_stride = dout.stride(1);
    params.dA_d_stride = dA.stride(0);
    params.dA_dstate_stride = dA.stride(1);
    params.dB_d_stride = dB.stride(0);
    params.dB_batch_stride = dB.stride(0);
    params.dB_group_stride = dB.stride(1);
    params.dB_dstate_stride = dB.stride(1);
    params.dC_d_stride = dC.stride(0);
    params.dC_batch_stride = dC.stride(0);
    params.dC_group_stride = dC.stride(1);
    params.dC_dstate_stride = dC.stride(1);
    params.du_batch_stride = du.stride(0);
    params.du_d_stride = du.stride(1);
    params.ddelta_batch_stride = ddelta.stride(0);
    params.ddelta_d_stride = ddelta.stride(1);
    if (has_deltaA) {
        params.ddeltaA_ptr = ddeltaA.data_ptr();
        params.ddeltaA_batch_stride = ddeltaA.stride(0);
        params.ddeltaA_d_stride = ddeltaA.stride(1);
    } else {
        params.ddeltaA_ptr = nullptr;
    }
}

std::vector<at::Tensor> simplified_scan_fwd(const at::Tensor &u, const at::Tensor &delta,
                  const at::Tensor &A, const at::Tensor &B, const at::Tensor &C,
                  const c10::optional<at::Tensor> &deltaA_,
                  const std::string& discretization_method_str) {
    auto input_type = u.scalar_type();
    auto weight_type = A.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::ComplexFloat);
    TORCH_CHECK(weight_type == at::ScalarType::ComplexFloat);

    DiscretizationMethod d_method;
    if (discretization_method_str == "zoh") {
        d_method = DiscretizationMethod::ZOH;
    } else if (discretization_method_str == "bilinear") {
        d_method = DiscretizationMethod::BILINEAR;
    } else if (discretization_method_str == "dirac") {
        d_method = DiscretizationMethod::DIRAC;
    } 
    else {
        TORCH_CHECK(false, "Discretization method not supported");
    }

    TORCH_CHECK(u.is_cuda());
    TORCH_CHECK(delta.is_cuda());
    TORCH_CHECK(A.is_cuda());
    TORCH_CHECK(B.is_cuda());
    TORCH_CHECK(C.is_cuda());

    TORCH_CHECK(u.stride(-1) == 1 || u.size(-1) == 1);
    TORCH_CHECK(delta.stride(-1) == 1 || delta.size(-1) == 1);

    const auto sizes = u.sizes();
    const int batch_size = sizes[0];
    const int dim = sizes[1];
    const int seqlen = sizes[2];
    const int dstate = A.size(1);
    const int n_groups = 1;

    TORCH_CHECK(dstate <= 256, "simplified_scan only supports state dimension <= 256");

    CHECK_SHAPE(u, batch_size, dim, seqlen);
    CHECK_SHAPE(delta, batch_size, dim, seqlen);
    CHECK_SHAPE(A, dim, dstate);
    CHECK_SHAPE(B, dim, dstate);
    CHECK_SHAPE(C, dim, dstate);

    const int n_chunks = (seqlen + 2048 - 1) / 2048;
    at::Tensor out = torch::empty_like(u);
    at::Tensor x;
    x = torch::empty({batch_size, dim, n_chunks, dstate * 2}, 
                     u.options().dtype(at::ScalarType::ComplexFloat));

    SSMParamsBase params;
    set_ssm_params_fwd(params, batch_size, dim, seqlen, dstate, n_groups, n_chunks,
                       u, delta, A, B, C, out,
                       deltaA_.has_value() ? deltaA_.value().data_ptr() : nullptr,
                       x.data_ptr(),
                       d_method);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{u.device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    DISPATCH_DISCRETIZATION(params.d_method, DMethod, [&] {
        simplified_scan_fwd_cuda<DMethod>(params, stream);
    });

    return {out, x};
}

std::vector<at::Tensor> simplified_scan_bwd(const at::Tensor &u, const at::Tensor &delta,
                  const at::Tensor &A, const at::Tensor &B, const at::Tensor &C,
                  const c10::optional<at::Tensor> &deltaA_,
                  const at::Tensor &dout,
                  const c10::optional<at::Tensor> &x_,
                  const std::string& discretization_method_str) {
    auto input_type = u.scalar_type();
    auto weight_type = A.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::ComplexFloat);
    TORCH_CHECK(weight_type == at::ScalarType::ComplexFloat);

    DiscretizationMethod d_method;
    if (discretization_method_str == "zoh") {
        d_method = DiscretizationMethod::ZOH;
    } else if (discretization_method_str == "bilinear") {
        d_method = DiscretizationMethod::BILINEAR;
    } else if (discretization_method_str == "dirac") {
        d_method = DiscretizationMethod::DIRAC;
    } 
    else {
        TORCH_CHECK(false, "Discretization method not supported");
    }

    TORCH_CHECK(u.is_cuda());
    TORCH_CHECK(delta.is_cuda());
    TORCH_CHECK(A.is_cuda());
    TORCH_CHECK(B.is_cuda());
    TORCH_CHECK(C.is_cuda());
    TORCH_CHECK(dout.is_cuda());

    TORCH_CHECK(u.stride(-1) == 1 || u.size(-1) == 1);
    TORCH_CHECK(delta.stride(-1) == 1 || delta.size(-1) == 1);
    TORCH_CHECK(dout.stride(-1) == 1 || dout.size(-1) == 1);

    const auto sizes = u.sizes();
    const int batch_size = sizes[0];
    const int dim = sizes[1];
    const int seqlen = sizes[2];
    const int dstate = A.size(1);
    const int n_groups = 1;

    TORCH_CHECK(dstate <= 256, "simplified_scan only supports state dimension <= 256");

    // deltaA is optional
    const bool has_deltaA = deltaA_.has_value();
    if (deltaA_.has_value()) {
        auto deltaA = deltaA_.value();
        TORCH_CHECK(deltaA.scalar_type() == at::ScalarType::Float);
        TORCH_CHECK(deltaA.is_cuda());
        TORCH_CHECK(deltaA.stride(-1) == 1 || deltaA.size(-1) == 1);
        CHECK_SHAPE(deltaA, batch_size, dim, seqlen);
    }

    CHECK_SHAPE(u, batch_size, dim, seqlen);
    CHECK_SHAPE(delta, batch_size, dim, seqlen);
    CHECK_SHAPE(A, dim, dstate);
    CHECK_SHAPE(B, dim, dstate);
    CHECK_SHAPE(C, dim, dstate);
    CHECK_SHAPE(dout, batch_size, dim, seqlen);

    const int n_chunks = (seqlen + 2048 - 1) / 2048;
    if (n_chunks > 1) { TORCH_CHECK(x_.has_value()); }
    if (x_.has_value()) {
        auto x = x_.value();
        TORCH_CHECK(x.scalar_type() == weight_type);
        TORCH_CHECK(x.is_cuda());
        TORCH_CHECK(x.is_contiguous());
        CHECK_SHAPE(x, batch_size, dim, n_chunks, 2 * dstate);
    }

    at::Tensor du = torch::empty_like(u);
    at::Tensor ddelta = torch::empty_like(delta);
    at::Tensor dA = torch::zeros_like(A);
    at::Tensor dB = torch::zeros_like(B);
    at::Tensor dC = torch::zeros_like(C);
    at::Tensor ddeltaA;
    if (deltaA_.has_value()) { ddeltaA = torch::empty_like(deltaA_.value()); }

    at::Tensor out = torch::empty_like(u);

    SSMParamsBwd params;
    set_ssm_params_bwd(params, batch_size, dim, seqlen, dstate, n_groups, n_chunks,
                       u, delta, A, B, C, out,
                       deltaA_.has_value() ? deltaA_.value().data_ptr() : nullptr,
                       x_.has_value() ? x_.value().data_ptr() : nullptr,
                       dout, du, ddelta, ddeltaA, dA, dB, dC,
                       has_deltaA,
                       d_method);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{u.device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    DISPATCH_DISCRETIZATION(params.d_method, DMethod, [&] {
        simplified_scan_bwd_cuda<DMethod>(params, stream);
    });

    std::vector<at::Tensor> result = {du, ddelta, dA, dB, dC};
    if (has_deltaA) { result.push_back(ddeltaA); }
    return result;
}
