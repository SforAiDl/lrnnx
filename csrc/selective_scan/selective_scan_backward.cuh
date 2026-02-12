/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 * Modified by SAiDL.
 ******************************************************************************/

#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK
#include <ATen/cuda/Atomic.cuh>  // For atomicAdd on complex

#ifndef USE_ROCM
    #include <cub/block/block_load.cuh>
    #include <cub/block/block_store.cuh>
    #include <cub/block/block_scan.cuh>
    #include <cub/block/block_reduce.cuh>
#else
    #include <hipcub/hipcub.hpp>
    namespace cub = hipcub;
#endif

#include <cstdio>
#include "selective_scan.h"
#include "../reverse_scan.cuh"

template<typename scalar_t> __device__ __forceinline__ scalar_t conj(scalar_t x);
template<> __device__ __forceinline__ float conj<float>(float x) { return x; }
template<> __device__ __forceinline__ complex_t conj<complex_t>(complex_t x) { return std::conj(x); }


template<int kNThreads_, int kNItems_, bool kIsEvenLen_, bool kIsVariableB_, bool kIsVariableC_,
         bool kDeltaSoftplus_, bool kHasZ_, typename input_t_, typename weight_t_, DiscretizationMethod DMethod_>
struct Selective_Scan_bwd_kernel_traits {
    static_assert(kNItems_ % 4 == 0);
    using input_t = input_t_;
    using weight_t = weight_t_;
    static constexpr int kNThreads = kNThreads_;
    static constexpr int kNItems = kNItems_;
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 2 || kNBytes == 4);
    static constexpr int kNElts = kNBytes == 4 ? 4 : constexpr_min(8, kNItems);
    static_assert(kNItems % kNElts == 0);
    static constexpr int kNLoads = kNItems / kNElts;
    static constexpr bool kIsComplex = std::is_same_v<weight_t, complex_t>;
    static constexpr bool kIsEvenLen = kIsEvenLen_;
    static constexpr bool kIsVariableB = kIsVariableB_;
    static constexpr bool kIsVariableC = kIsVariableC_;
    static constexpr bool kDeltaSoftplus = kDeltaSoftplus_;
    static constexpr bool kHasZ = kHasZ_;

    static constexpr DiscretizationMethod DMethod = DMethod_;
    // Setting MinBlocksPerMP to be 3 (instead of 2) for 128 threads with float improves occupancy.
    // For complex this would lead to massive register spilling, so we keep it at 2.
    static constexpr int kMinBlocks = kNThreads == 128 && !kIsComplex ? 3 : 2;
    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    using scan_t = std::conditional_t<!kIsComplex, float2, float4>;
    using BlockLoadT = cub::BlockLoad<input_t, kNThreads, kNItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadVecT = cub::BlockLoad<vec_t, kNThreads, kNLoads, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadWeightT = cub::BlockLoad<input_t, kNThreads, !kIsComplex ? kNItems : kNItems * 2, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadWeightVecT = cub::BlockLoad<vec_t, kNThreads, !kIsComplex ? kNLoads : kNLoads * 2, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockStoreT = cub::BlockStore<input_t, kNThreads, kNItems, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockStoreVecT = cub::BlockStore<vec_t, kNThreads, kNLoads, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    // using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_RAKING_MEMOIZE>;
    using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_RAKING>;
    // using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_WARP_SCANS>;
    using BlockReverseScanT = BlockReverseScan<scan_t, kNThreads>;
    using BlockReduceT = cub::BlockReduce<scan_t, kNThreads>;
    using BlockReduceFloatT = cub::BlockReduce<float, kNThreads>;
    using BlockReduceComplexT = cub::BlockReduce<complex_t, kNThreads>;
    using BlockExchangeT = cub::BlockExchange<float, kNThreads, !kIsComplex ? kNItems : kNItems * 2>;

    static constexpr int kSmemIOSize = custom_max({sizeof(typename BlockLoadT::TempStorage),
                                                    sizeof(typename BlockLoadVecT::TempStorage),
                                                    (int(kIsVariableB) + int(kIsVariableC)) * sizeof(typename BlockLoadWeightT::TempStorage),
                                                    (int(kIsVariableB) + int(kIsVariableC)) * sizeof(typename BlockLoadWeightVecT::TempStorage),
                                                    sizeof(typename BlockStoreT::TempStorage),
                                                    sizeof(typename BlockStoreVecT::TempStorage)});
    static constexpr int kSmemExchangeSize = (int(kIsVariableB) + int(kIsVariableC)) * sizeof(typename BlockExchangeT::TempStorage);
    static constexpr int kSmemReduceSize = sizeof(typename BlockReduceT::TempStorage);
    static constexpr int kSmemSize = kSmemIOSize + kSmemExchangeSize + kSmemReduceSize + sizeof(typename BlockScanT::TempStorage) + sizeof(typename BlockReverseScanT::TempStorage);
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads, Ktraits::kMinBlocks)
void selective_scan_bwd_kernel(SSMParamsBwd params) {
    constexpr bool kIsComplex = Ktraits::kIsComplex;
    constexpr bool kIsVariableB = Ktraits::kIsVariableB;
    constexpr bool kIsVariableC = Ktraits::kIsVariableC;
    constexpr bool kDeltaSoftplus = Ktraits::kDeltaSoftplus;
    constexpr bool kHasZ = Ktraits::kHasZ;
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kNItems = Ktraits::kNItems;
    constexpr DiscretizationMethod DMethod = Ktraits::DMethod;
    using input_t = typename Ktraits::input_t;
    using weight_t = typename Ktraits::weight_t;
    using scan_t = typename Ktraits::scan_t;

    // Shared memory.
    extern __shared__ char smem_[];
    // cast to lvalue reference of expected type
    // char *smem_loadstorescan = smem_ + 2 * MAX_DSTATE * sizeof(weight_t);
    // auto& smem_load = reinterpret_cast<typename BlockLoadT::TempStorage&>(smem_ + 2 * MAX_DSTATE * sizeof(weight_t));
    // auto& smem_load = reinterpret_cast<typename BlockLoadT::TempStorage&>(smem_loadstorescan);
    auto& smem_load = reinterpret_cast<typename Ktraits::BlockLoadT::TempStorage&>(smem_);
    auto& smem_load_weight = reinterpret_cast<typename Ktraits::BlockLoadWeightT::TempStorage&>(smem_);
    auto& smem_load_weight1 = *reinterpret_cast<typename Ktraits::BlockLoadWeightT::TempStorage*>(smem_ + sizeof(typename Ktraits::BlockLoadWeightT::TempStorage));
    auto& smem_store = reinterpret_cast<typename Ktraits::BlockStoreT::TempStorage&>(smem_);
    auto& smem_exchange = *reinterpret_cast<typename Ktraits::BlockExchangeT::TempStorage*>(smem_ + Ktraits::kSmemIOSize);
    auto& smem_exchange1 = *reinterpret_cast<typename Ktraits::BlockExchangeT::TempStorage*>(smem_ + Ktraits::kSmemIOSize + sizeof(typename Ktraits::BlockExchangeT::TempStorage));
    auto& smem_reduce = *reinterpret_cast<typename Ktraits::BlockReduceT::TempStorage*>(reinterpret_cast<char *>(&smem_exchange) + Ktraits::kSmemExchangeSize);
    auto& smem_reduce_float = *reinterpret_cast<typename Ktraits::BlockReduceFloatT::TempStorage*>(&smem_reduce);
    auto& smem_reduce_complex = *reinterpret_cast<typename Ktraits::BlockReduceComplexT::TempStorage*>(&smem_reduce);
    auto& smem_scan = *reinterpret_cast<typename Ktraits::BlockScanT::TempStorage*>(reinterpret_cast<char *>(&smem_reduce) + Ktraits::kSmemReduceSize);
    auto& smem_reverse_scan = *reinterpret_cast<typename Ktraits::BlockReverseScanT::TempStorage*>(reinterpret_cast<char *>(&smem_scan) + sizeof(typename Ktraits::BlockScanT::TempStorage));
    weight_t *smem_delta_a = reinterpret_cast<weight_t *>(smem_ + Ktraits::kSmemSize);
    scan_t *smem_running_postfix = reinterpret_cast<scan_t *>(smem_delta_a + 2 * MAX_DSTATE + kNThreads);
    weight_t *smem_da = reinterpret_cast<weight_t *>(smem_running_postfix + MAX_DSTATE);
    weight_t *smem_dbc = reinterpret_cast<weight_t *>(smem_da + MAX_DSTATE);

    const int batch_id = blockIdx.x;
    const int dim_id = blockIdx.y;
    const int group_id = dim_id / (params.dim_ngroups_ratio);
    input_t *u = reinterpret_cast<input_t *>(params.u_ptr) + batch_id * params.u_batch_stride
        + dim_id * params.u_d_stride;
    input_t *delta = reinterpret_cast<input_t *>(params.delta_ptr) + batch_id * params.delta_batch_stride
        + dim_id * params.delta_d_stride;
    input_t *deltaA = params.deltaA_ptr != nullptr 
        ? reinterpret_cast<input_t *>(params.deltaA_ptr) + batch_id * params.deltaA_batch_stride
            + dim_id * params.deltaA_d_stride 
        : nullptr;
    input_t *dout = reinterpret_cast<input_t *>(params.dout_ptr) + batch_id * params.dout_batch_stride
        + dim_id * params.dout_d_stride;
    weight_t *A = reinterpret_cast<weight_t *>(params.A_ptr) + dim_id * params.A_d_stride;
    weight_t *B = reinterpret_cast<weight_t *>(params.B_ptr) + dim_id * params.B_d_stride;
    input_t *Bvar = reinterpret_cast<input_t *>(params.B_ptr) + batch_id * params.B_batch_stride + group_id * params.B_group_stride;
    weight_t *C = reinterpret_cast<weight_t *>(params.C_ptr) + dim_id * params.C_d_stride;
    input_t *Cvar = reinterpret_cast<input_t *>(params.C_ptr) + batch_id * params.C_batch_stride + group_id * params.C_group_stride;
    weight_t *dA = reinterpret_cast<weight_t *>(params.dA_ptr) + dim_id * params.dA_d_stride;
    weight_t *dB = reinterpret_cast<weight_t *>(params.dB_ptr)
        + (!kIsVariableB ? dim_id * params.dB_d_stride : batch_id * (!kIsComplex ? params.dB_batch_stride : params.dB_batch_stride / 2) + group_id * params.dB_group_stride);
    weight_t *dC = reinterpret_cast<weight_t *>(params.dC_ptr)
        + (!kIsVariableC ? dim_id * params.dC_d_stride : batch_id * (!kIsComplex ? params.dC_batch_stride : params.dC_batch_stride / 2) + group_id * params.dC_group_stride);
    float *dD = params.dD_ptr == nullptr ? nullptr : reinterpret_cast<float *>(params.dD_ptr) + dim_id;
    float D_val = params.D_ptr == nullptr ? 0 : reinterpret_cast<float *>(params.D_ptr)[dim_id];
    float *ddelta_bias = params.ddelta_bias_ptr == nullptr ? nullptr : reinterpret_cast<float *>(params.ddelta_bias_ptr) + dim_id;
    float delta_bias = params.delta_bias_ptr == nullptr ? 0 : reinterpret_cast<float *>(params.delta_bias_ptr)[dim_id];
    scan_t *x = params.x_ptr == nullptr
        ? nullptr
        : reinterpret_cast<scan_t *>(params.x_ptr) + (batch_id * params.dim + dim_id) * (params.n_chunks) * params.dstate;
    float dD_val = 0;
    float ddelta_bias_val = 0;

    constexpr int kChunkSize = kNThreads * kNItems;
    u += (params.n_chunks - 1) * kChunkSize;
    delta += (params.n_chunks - 1) * kChunkSize;
    // load deltaA if provided
    if (deltaA != nullptr) {
        deltaA += (params.n_chunks - 1) * kChunkSize;
    }
    dout += (params.n_chunks - 1) * kChunkSize;
    Bvar += (params.n_chunks - 1) * kChunkSize * (!kIsComplex ? 1 : 2);
    Cvar += (params.n_chunks - 1) * kChunkSize * (!kIsComplex ? 1 : 2);
    for (int chunk = params.n_chunks - 1; chunk >= 0; --chunk) {
        input_t u_vals[kNItems];
        input_t delta_vals_load[kNItems];
        input_t deltaA_vals_load[kNItems];
        input_t dout_vals_load[kNItems];
        __syncthreads();
        load_input<Ktraits>(u, u_vals, smem_load, params.seqlen - chunk * kChunkSize);
        u -= kChunkSize;
        __syncthreads();
        load_input<Ktraits>(delta, delta_vals_load, smem_load, params.seqlen - chunk * kChunkSize);
        // will reload delta at the same location if kDeltaSoftplus
        if constexpr (!kDeltaSoftplus) { delta -= kChunkSize; }
        if (deltaA != nullptr) {
            __syncthreads();
            load_input<Ktraits>(deltaA, deltaA_vals_load, smem_load, params.seqlen - chunk * kChunkSize);
            deltaA -= kChunkSize;
        }
        __syncthreads();
        load_input<Ktraits>(dout, dout_vals_load, smem_load, params.seqlen - chunk * kChunkSize);
        dout -= kChunkSize;

        float dout_vals[kNItems], delta_vals[kNItems], deltaA_vals[kNItems];
        #pragma unroll
        for (int i = 0; i < kNItems; ++i) {
            dout_vals[i] = float(dout_vals_load[i]);
            delta_vals[i] = float(delta_vals_load[i]) + delta_bias;
            if constexpr (kDeltaSoftplus) {
                delta_vals[i] = delta_vals[i] <= 20.f ? log1pf(expf(delta_vals[i])) : delta_vals[i];
            }
            if (deltaA != nullptr) {
                deltaA_vals[i] = float(deltaA_vals_load[i]);
            }
        }

        if constexpr (kHasZ) {
            input_t *z = reinterpret_cast<input_t *>(params.z_ptr) + batch_id * params.z_batch_stride
                + dim_id * params.z_d_stride + chunk * kChunkSize;
            input_t *out = reinterpret_cast<input_t *>(params.out_ptr) + batch_id * params.out_batch_stride
                + dim_id * params.out_d_stride + chunk * kChunkSize;
            input_t *dz = reinterpret_cast<input_t *>(params.dz_ptr) + batch_id * params.dz_batch_stride
                + dim_id * params.dz_d_stride + chunk * kChunkSize;
            input_t z_vals[kNItems], out_vals[kNItems];
            __syncthreads();
            load_input<Ktraits>(z, z_vals, smem_load, params.seqlen - chunk * kChunkSize);
            __syncthreads();
            load_input<Ktraits>(out, out_vals, smem_load, params.seqlen - chunk * kChunkSize);
            float dz_vals[kNItems], z_silu_vals[kNItems];
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                float z_val = z_vals[i];
                float z_sigmoid_val = 1.0f / (1.0f + expf(-z_val));
                z_silu_vals[i] = z_val * z_sigmoid_val;
                dz_vals[i] = dout_vals[i] * float(out_vals[i]) * z_sigmoid_val
                             * (1.0f + z_val * (1.0f - z_sigmoid_val));
                dout_vals[i] *= z_silu_vals[i];
            }
            __syncthreads();
            store_output<Ktraits>(dz, dz_vals, smem_store, params.seqlen - chunk * kChunkSize);
            if (params.out_z_ptr != nullptr) {  // Recompute and store out_z
                float out_z_vals[kNItems];
                #pragma unroll
                for (int i = 0; i < kNItems; ++i) { out_z_vals[i] = float(out_vals[i]) * z_silu_vals[i]; }
                // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
                    // printf("out_val=%f, z_silu_val = %f, out_z_val = %f\n", float(out_vals[0]), z_silu_vals[0], out_z_vals[0]);
                // }
                input_t *out_z = reinterpret_cast<input_t *>(params.out_z_ptr) + batch_id * params.out_z_batch_stride
                    + dim_id * params.out_z_d_stride + chunk * kChunkSize;
                __syncthreads();
                store_output<Ktraits>(out_z, out_z_vals, smem_store, params.seqlen - chunk * kChunkSize);
            }
        }

        float du_vals[kNItems];
        #pragma unroll
        for (int i = 0; i < kNItems; ++i) { du_vals[i] = D_val * dout_vals[i]; }
        #pragma unroll
        for (int i = 0; i < kNItems; ++i) { dD_val += dout_vals[i] * float(u_vals[i]); }

        float ddelta_vals[kNItems] = {0};
        float ddeltaA_vals[kNItems] = {0};
        __syncthreads();
        for (int state_idx = 0; state_idx < params.dstate; ++state_idx) {
            const weight_t A_val = A[state_idx * params.A_dstate_stride];
            // multiply the real part of A with LOG2E so we can use exp2f instead of expf.
            weight_t A_scaled;
            constexpr float kLog2e = M_LOG2E;
            if constexpr (!kIsComplex) {
                A_scaled = A_val * kLog2e;
            } else {
                A_scaled = complex_t(A_val.real_ * kLog2e, A_val.imag_);
            }
            weight_t B_val, C_val;
            weight_t B_vals[kNItems], C_vals[kNItems];
            if constexpr (!kIsVariableB) {
                B_val = B[state_idx * params.B_dstate_stride];
            } else {
                load_weight<Ktraits>(Bvar + state_idx * params.B_dstate_stride, B_vals,
                    smem_load_weight, (params.seqlen - chunk * kChunkSize) * (!kIsComplex ? 1 : 2));
            }
            if constexpr (!kIsVariableC) {
                C_val = C[state_idx * params.C_dstate_stride];
            } else {
                auto &smem_load_weight_C = !kIsVariableB ? smem_load_weight : smem_load_weight1;
                load_weight<Ktraits>(Cvar + state_idx * params.C_dstate_stride, C_vals,
                    smem_load_weight_C, (params.seqlen - chunk * kChunkSize) * (!kIsComplex ? 1 : 2));
            }
            // const weight_t A_val = smem_a[state_idx];
            scan_t thread_data[kNItems], thread_reverse_data[kNItems];
            if constexpr (!kIsComplex) {
                #pragma unroll
                for (int i = 0; i < kNItems; ++i) {
                    // use deltaA for A discretization if provided, otherwise use delta
                    const float delta_for_A = (deltaA != nullptr) ? deltaA_vals[i] : delta_vals[i];
                    float delta_a_exp;
                    if constexpr (DMethod == DiscretizationMethod::ZOH){
                        const float A_del_for_A = A_val * delta_for_A;
                        delta_a_exp = expf(A_del_for_A);
                        const float A_del_for_B = A_val * delta_vals[i];
                        const float B_tilde = expm1f(A_del_for_B) / A_val;
                        const float B_u_disc = (!kIsVariableB ? 1.0f : B_vals[i]) * float(u_vals[i]) * B_tilde;
                        thread_data[i] = make_float2(delta_a_exp, B_u_disc);
                    } else if constexpr (DMethod == DiscretizationMethod::BILINEAR){
                        const float a_half_for_A = delta_for_A * 0.5f * A_val;
                        delta_a_exp = (1.0f + a_half_for_A) / (1.0f - a_half_for_A);
                        const float a_half_for_B = delta_vals[i] * 0.5f * A_val;
                        const float den_inv_for_B = 1.0f / (1.0f - a_half_for_B);
                        thread_data[i] = make_float2(delta_a_exp, !kIsVariableB ? delta_vals[i] * float(u_vals[i]) * den_inv_for_B : delta_vals[i] * float(u_vals[i]) * B_vals[i] * den_inv_for_B);
                    } else if constexpr (DMethod == DiscretizationMethod::DIRAC){
                        delta_a_exp = exp2f(delta_for_A * A_scaled);
                        thread_data[i] = make_float2(delta_a_exp, !kIsVariableB ? float(u_vals[i]) : float(u_vals[i]) * B_vals[i]);
                    } else if constexpr (DMethod == DiscretizationMethod::MAMBA){
                        delta_a_exp = exp2f(delta_for_A * A_scaled);
                        thread_data[i] = make_float2(delta_a_exp, !kIsVariableB ? delta_vals[i] * float(u_vals[i]) : delta_vals[i] * float(u_vals[i]) * B_vals[i]);
                    } else if constexpr (DMethod == DiscretizationMethod::S7) {
                        const float A_tv = delta_vals[i];
                        const float A_sq_half = A_tv * A_tv + 0.5f;
                        delta_a_exp = 1.0f - 1.0f / A_sq_half;
                        thread_data[i] = make_float2(delta_a_exp, !kIsVariableB ? float(u_vals[i]) : float(u_vals[i]) * B_vals[i]);
                    } else if constexpr (DMethod == DiscretizationMethod::RGLRU) {
                        static_assert(!kIsComplex, "RG LRU discretization only supports real weights");
                        const float A_base = A_val;
                        const float gate = delta_vals[i];  // pre-computed c * gate
                        const float A_bar = powf(A_base, gate);
                        const float sqrt_term = sqrtf(1.0f - A_bar * A_bar);
                        delta_a_exp = A_bar;
                        const float B_u_term = sqrt_term * float(u_vals[i]);
                        thread_data[i] = make_float2(A_bar, B_u_term);
                    }

                    if (i == 0) {
                        smem_delta_a[threadIdx.x == 0 ? state_idx + (chunk % 2) * MAX_DSTATE : threadIdx.x + 2 * MAX_DSTATE] = delta_a_exp;
                    } else {
                        thread_reverse_data[i - 1].x = delta_a_exp;
                    }
                    thread_reverse_data[i].y = dout_vals[i] *
                        (!kIsVariableC
                         ? (!kIsVariableB ? B_val * C_val : C_val)
                         : (!kIsVariableB ? B_val * C_vals[i] : C_vals[i]));
                }
                __syncthreads();
                thread_reverse_data[kNItems - 1].x = threadIdx.x == kNThreads - 1
                    ? (chunk == params.n_chunks - 1 ? 1.f : smem_delta_a[state_idx + ((chunk + 1) % 2) * MAX_DSTATE])
                    : smem_delta_a[threadIdx.x + 1 + 2 * MAX_DSTATE];

                scan_t running_prefix = chunk > 0 && threadIdx.x % 32 == 0 ? x[(chunk - 1) * params.dstate + state_idx] : make_float2(1.f, 0.f);
                SSMScanPrefixCallbackOp<weight_t> prefix_op(running_prefix);
                typename Ktraits::BlockScanT(smem_scan).InclusiveScan(
                    thread_data, thread_data, SSMScanOp<weight_t>(), prefix_op
                );
                scan_t running_postfix = chunk < params.n_chunks - 1 && threadIdx.x % 32 == 0 ? smem_running_postfix[state_idx] : make_float2(1.f, 0.f);
                SSMScanPrefixCallbackOp<weight_t> postfix_op(running_postfix);
                typename Ktraits::BlockReverseScanT(smem_reverse_scan).InclusiveReverseScan(
                    thread_reverse_data, thread_reverse_data, SSMScanOp<weight_t>(), postfix_op
                );
                if (threadIdx.x == 0) { smem_running_postfix[state_idx] = postfix_op.running_prefix; }

                // in bilinear mode, use h_prev from the scan to avoid instability when A_bar ~= 0
                float bilinear_h_prev_for_i0 = 0.0f;
                if constexpr (DMethod == DiscretizationMethod::BILINEAR) {
                    // reuse the smem_delta_a inter-thread buffer (entries [2*MAX_DSTATE, 2*MAX_DSTATE+kNThreads))
                    float *smem_h_last = reinterpret_cast<float *>(smem_delta_a + 2 * MAX_DSTATE);
                    smem_h_last[threadIdx.x] = thread_data[kNItems - 1].y;
                    __syncthreads();
                    if (threadIdx.x > 0) {
                        bilinear_h_prev_for_i0 = smem_h_last[threadIdx.x - 1];
                    } else {
                        bilinear_h_prev_for_i0 = (chunk > 0) ? x[(chunk - 1) * params.dstate + state_idx].y : 0.0f;
                    }
                }

                weight_t dA_val = 0, dBC_val = 0;
                weight_t dB_vals[kNItems], dC_vals[kNItems];
                #pragma unroll
                for (int i = 0; i < kNItems; ++i) {
                    const float h_i = thread_data[i].y;
                    const float u_val = float(u_vals[i]);
                    const float dx = thread_reverse_data[i].y;
                    // Use deltaA for A discretization if provided, otherwise use delta
                    const float delta_for_A = (deltaA != nullptr) ? deltaA_vals[i] : delta_vals[i];
                    
                    if constexpr (DMethod == DiscretizationMethod::ZOH) {
                        const float A_del_for_A = A_val * delta_for_A;
                        const float disc_A = expf(A_del_for_A);
                        const float A_del_for_B = A_val * delta_vals[i];
                        const float disc_A_for_B = expf(A_del_for_B);
                        const float K = expm1f(A_del_for_B) / A_val;
                        const float dx_b = !kIsVariableB ? dx : dx * B_vals[i];
                        du_vals[i] += dx_b * K;
                        const float a = h_i - (!kIsVariableB ? float(u_vals[i]) * K: float(u_vals[i]) * B_vals[i] * K);
                        if (deltaA != nullptr) {
                            ddeltaA_vals[i] += dx * A_val * a;
                            ddelta_vals[i] += float(u_vals[i]) * dx_b * disc_A_for_B;
                        } else {
                            ddelta_vals[i] += dx * A_val * a + float(u_vals[i]) * dx_b * disc_A_for_B;
                        }
                        dA_val += dx * delta_for_A * a + dx_b * float(u_vals[i]) * (delta_vals[i] * disc_A_for_B - K) / A_val;

                        if constexpr (kIsVariableB) {
                            dB_vals[i] = dx * float(u_vals[i]) * K;
                        }
                    } else if constexpr (DMethod == DiscretizationMethod::BILINEAR) {
                        const float v_for_A = 0.5f * delta_for_A * A_val;
                        const float one_minus_v_for_A = 1.0f - v_for_A;
                        const float den_inv_sq_for_A = 1.0f / (one_minus_v_for_A * one_minus_v_for_A);
                        const float v_for_B = 0.5f * delta_vals[i] * A_val;
                        const float den_inv_for_B = 1.0f / (1.0f - v_for_B);
                        const float B_u_term = (!kIsVariableB ? 1.0f : B_vals[i]) * den_inv_for_B * delta_vals[i] * u_val;

                        const float a_prev = (i > 0) ? thread_data[i - 1].y : bilinear_h_prev_for_i0;

                        const float dx_b_delta_inv = (!kIsVariableB ? 1.0f : B_vals[i]) * delta_vals[i] * den_inv_for_B * dx;
                        du_vals[i] += dx_b_delta_inv;

                        const float dDelta_input = B_u_term * den_inv_for_B / delta_vals[i];
                        if (deltaA != nullptr) {
                            const float dDeltaA_trans = a_prev * A_val * den_inv_sq_for_A;
                            ddeltaA_vals[i] += dx * dDeltaA_trans;
                            ddelta_vals[i] += dx * dDelta_input;
                        } else {
                            const float dDelta_trans = a_prev * A_val * den_inv_sq_for_A;
                            ddelta_vals[i] += dx * (dDelta_input + dDelta_trans);
                        }

                        const float dA_trans = a_prev * delta_for_A * den_inv_sq_for_A;
                        const float dA_input = B_u_term * delta_vals[i] * 0.5f * den_inv_for_B;
                        dA_val += dx * (dA_trans + dA_input);

                        if constexpr (kIsVariableB) {
                            dB_vals[i] = dx * delta_vals[i] * float(u_vals[i]) * den_inv_for_B;
                        }
                    } else if constexpr (DMethod == DiscretizationMethod::DIRAC) {
                        const float dx_b = !kIsVariableB ? dx : dx * B_vals[i];
                        const float a = h_i - (!kIsVariableB ? float(u_vals[i]) : float(u_vals[i]) * B_vals[i]);
                        du_vals[i] += dx_b;
                        if (deltaA != nullptr) {
                            ddeltaA_vals[i] += dx * A_val * a;
                        } else {
                            ddelta_vals[i] += dx * A_val * a;
                        }
                        dA_val += dx * delta_for_A * a;
                        if constexpr (kIsVariableB) {
                            dB_vals[i] = dx * float(u_vals[i]);
                        }
                    } else if constexpr (DMethod == DiscretizationMethod::MAMBA){
                        const float ddelta_u = !kIsVariableB ? dx : dx * B_vals[i];
                        du_vals[i] += ddelta_u * delta_vals[i];
                        const float a = h_i - (!kIsVariableB ? delta_vals[i] * u_val : delta_vals[i] * u_val * B_vals[i]);
                        ddelta_vals[i] += ddelta_u * u_val;
                        if (deltaA != nullptr) {
                            ddeltaA_vals[i] += dx * A_val * a;
                        } else {
                            ddelta_vals[i] += dx * A_val * a;
                        }
                        dA_val += dx * delta_for_A * a;
                        if constexpr (kIsVariableB) {
                            dB_vals[i] = dx * delta_vals[i] * u_val;
                        }
                    } else if constexpr (DMethod == DiscretizationMethod::S7) {
                        const float dx_b = !kIsVariableB ? dx : dx * B_vals[i];
                        const float a = h_i - (!kIsVariableB ? float(u_val) : float(u_val) * B_vals[i]);
                        du_vals[i] += dx_b;
                        const float A_tv = delta_vals[i];
                        const float A_sq_half = A_tv * A_tv + 0.5f;
                        const float dA_bar_dA = 2.0f * A_tv / (A_sq_half * A_sq_half);
                        const float A_bar = 1.0f - 1.0f / A_sq_half;
                        ddelta_vals[i] += dx * a * dA_bar_dA / A_bar;
                        if constexpr (kIsVariableB) {
                            dB_vals[i] = dx * float(u_vals[i]);
                        }
                    } else if constexpr (DMethod == DiscretizationMethod::RGLRU) {
                        static_assert(!kIsComplex, "RG LRU discretization only supports real weights");
                        
                        const float A_base = A_val;
                        const float gate = delta_vals[i]; 
                        const float A_bar = powf(A_base, gate);
                        const float sqrt_term = sqrtf(1.0f - A_bar * A_bar);
                        
                        du_vals[i] += dx * sqrt_term;
                        
                        // for padded (gate==0) positions, force zero gradient and avoid 0/0.
                        if (gate > 0.0f) {
                            const float dA_bar = dx * (h_i * sqrt_term - u_val) / (A_bar * sqrt_term);
                            ddelta_vals[i] += dA_bar * A_bar * logf(A_base);
                            dA_val += dA_bar * gate * A_bar / A_base;
                        }
                    }
                    if constexpr (!kIsVariableB || !kIsVariableC) {
                        if constexpr (!kIsVariableB) {
                            dBC_val += dout_vals[i] * (!kIsVariableC ? h_i : h_i * C_vals[i]);
                        } else {
                            dBC_val += dout_vals[i] * h_i;
                        }
                    }
                    if constexpr (kIsVariableC) {
                        dC_vals[i] = dout_vals[i] * (!kIsVariableB ? h_i * B_val : h_i);
                    }
                }
                // block-exchange to make the atomicAdd's coalesced, otherwise they're much slower
                if constexpr (kIsVariableB || kIsVariableC) {
                    if constexpr (kIsVariableB) {
                        typename Ktraits::BlockExchangeT(smem_exchange).BlockedToStriped(dB_vals, dB_vals);
                    }
                    if constexpr (kIsVariableC) {
                        auto &smem_exchange_C = !kIsVariableB ? smem_exchange : smem_exchange1;
                        typename Ktraits::BlockExchangeT(smem_exchange_C).BlockedToStriped(dC_vals, dC_vals);
                    }
                    const int seqlen_remaining = params.seqlen - chunk * kChunkSize - threadIdx.x;
                    weight_t *dB_cur = dB + state_idx * params.dB_dstate_stride + chunk * kChunkSize + threadIdx.x;
                    weight_t *dC_cur = dC + state_idx * params.dC_dstate_stride + chunk * kChunkSize + threadIdx.x;
                    #pragma unroll
                    for (int i = 0; i < kNItems; ++i) {
                        if (i * kNThreads < seqlen_remaining) {
                            if constexpr (kIsVariableB) { gpuAtomicAdd(dB_cur + i * kNThreads, dB_vals[i]); }
                            if constexpr (kIsVariableC) { gpuAtomicAdd(dC_cur + i * kNThreads, dC_vals[i]); }
                        }
                    }
                }
                if constexpr (!kIsVariableB || !kIsVariableC) {
                    float2 dA_dBC_val = make_float2(dA_val, dBC_val);
                    dA_dBC_val = typename Ktraits::BlockReduceT(smem_reduce).Sum(dA_dBC_val);
                    dA_val = dA_dBC_val.x;
                    if (threadIdx.x == 0) {
                        smem_dbc[state_idx] = chunk == params.n_chunks - 1 ? dA_dBC_val.y : dA_dBC_val.y + smem_dbc[state_idx];
                    }
                } else {
                    dA_val = typename Ktraits::BlockReduceFloatT(smem_reduce_float).Sum(dA_val);
                }
                if (threadIdx.x == 0) {
                    smem_da[state_idx] = chunk == params.n_chunks - 1 ? dA_val : dA_val + smem_da[state_idx];
                }
            } else {
                #pragma unroll
                for (int i = 0; i < kNItems; ++i) {
                    const float delta_for_A = (deltaA != nullptr) ? deltaA_vals[i] : delta_vals[i];
                    complex_t delta_a_exp;
                    if constexpr (DMethod == DiscretizationMethod::ZOH) {
                        const complex_t A_del_for_A = delta_for_A * A_val;
                        delta_a_exp = cexpf(A_del_for_A);
                        const complex_t A_del_for_B = delta_vals[i] * A_val;
                        const complex_t delta_a_exp_for_B = cexpf(A_del_for_B);
                        const complex_t discrete_B = (delta_a_exp_for_B - complex_t(1.0f, 0.f)) / A_val;
                        const weight_t B_u_val = !kIsVariableB
                                                 ? discrete_B * float(u_vals[i])
                                                 : discrete_B * B_vals[i] * float(u_vals[i]);
                        thread_data[i] = make_float4(delta_a_exp.real_, delta_a_exp.imag_, B_u_val.real_, B_u_val.imag_);
                    } else if constexpr (DMethod == DiscretizationMethod::BILINEAR) {
                        const complex_t a_half_for_A = 0.5f * delta_for_A * A_val;
                        const complex_t inv_term_for_A = complex_t(1.f, 0.f) / (complex_t(1.f, 0.f) - a_half_for_A);
                        delta_a_exp = (complex_t(1.f, 0.f) + a_half_for_A) * inv_term_for_A;
                        const complex_t a_half_for_B = 0.5f * delta_vals[i] * A_val;
                        const complex_t inv_term_for_B = complex_t(1.f, 0.f) / (complex_t(1.f, 0.f) - a_half_for_B);
                        const complex_t discrete_B = delta_vals[i] * inv_term_for_B;

                        const weight_t B_delta_u_val = !kIsVariableB
                                                    ? discrete_B * float(u_vals[i])
                                                    : discrete_B * B_vals[i] * float(u_vals[i]);
                        thread_data[i] = make_float4(delta_a_exp.real_, delta_a_exp.imag_, B_delta_u_val.real_, B_delta_u_val.imag_);
                    } else if constexpr (DMethod == DiscretizationMethod::DIRAC) {
                        delta_a_exp = cexp2f(delta_for_A * A_scaled);
                        const weight_t B_u_val = !kIsVariableB
                                                    ? float(u_vals[i])
                                                    : B_vals[i] * float(u_vals[i]);
                        thread_data[i] = make_float4(delta_a_exp.real_, delta_a_exp.imag_, B_u_val.real_, B_u_val.imag_);
                    } else if constexpr (DMethod == DiscretizationMethod::MAMBA) {
                        delta_a_exp = cexp2f(delta_for_A * A_scaled);
                        const weight_t B_delta_u_val = !kIsVariableB
                                                    ? delta_vals[i] * float(u_vals[i])
                                                    : B_vals[i] * (delta_vals[i] * float(u_vals[i]));
                        thread_data[i] = make_float4(delta_a_exp.real_, delta_a_exp.imag_, B_delta_u_val.real_, B_delta_u_val.imag_);
                    } else if constexpr (DMethod == DiscretizationMethod::S7) {
                        static_assert(!kIsComplex, "S7 discretization only supports real weights");
                    } else if constexpr (DMethod == DiscretizationMethod::RGLRU) {
                        static_assert(!kIsComplex, "RG LRU discretization only supports real weights");
                    }
                    if (i == 0) {
                        smem_delta_a[threadIdx.x == 0 ? state_idx + (chunk % 2) * MAX_DSTATE : threadIdx.x + 2 * MAX_DSTATE] = delta_a_exp;
                    } else {
                        thread_reverse_data[i - 1].x = delta_a_exp.real_;
                        thread_reverse_data[i - 1].y = -delta_a_exp.imag_;
                    }
                    complex_t dout_BC = 2 * dout_vals[i]
                        * conj(!kIsVariableC
                                ? (!kIsVariableB ? B_val * C_val : C_val)
                                : (!kIsVariableB ? B_val * C_vals[i] : C_vals[i]));
                    thread_reverse_data[i].z = dout_BC.real_;
                    thread_reverse_data[i].w = dout_BC.imag_;
                }
                __syncthreads();
                complex_t delta_a_exp = threadIdx.x == kNThreads - 1
                    ? (chunk == params.n_chunks - 1 ? 1.f : smem_delta_a[state_idx + ((chunk + 1) % 2) * MAX_DSTATE])
                    : smem_delta_a[threadIdx.x + 1 + 2 * MAX_DSTATE];
                thread_reverse_data[kNItems - 1].x = delta_a_exp.real_;
                thread_reverse_data[kNItems - 1].y = -delta_a_exp.imag_;
                // initialize running total
                scan_t running_prefix = chunk > 0 && threadIdx.x % 32 == 0 ? x[(chunk - 1) * params.dstate + state_idx] : make_float4(1.f, 0.f, 0.f, 0.f);
                SSMScanPrefixCallbackOp<weight_t> prefix_op(running_prefix);
                typename Ktraits::BlockScanT(smem_scan).InclusiveScan(
                    thread_data, thread_data, SSMScanOp<weight_t>(), prefix_op
                );
                scan_t running_postfix = chunk < params.n_chunks - 1 && threadIdx.x % 32 == 0 ? smem_running_postfix[state_idx] : make_float4(1.f, 0.f, 0.f, 0.f);
                SSMScanPrefixCallbackOp<weight_t> postfix_op(running_postfix);
                typename Ktraits::BlockReverseScanT(smem_reverse_scan).InclusiveReverseScan(
                    thread_reverse_data, thread_reverse_data, SSMScanOp<weight_t>(), postfix_op
                );
                if (threadIdx.x == 0) { smem_running_postfix[state_idx] = postfix_op.running_prefix; }
                weight_t dA_val = 0, dBC_val = 0;
                weight_t dB_vals[kNItems], dC_vals[kNItems];
                #pragma unroll
                for (int i = 0; i < kNItems; ++i) {
                    complex_t x = complex_t(thread_data[i].z, thread_data[i].w);
                    complex_t dx = complex_t(thread_reverse_data[i].z, thread_reverse_data[i].w);
                    if constexpr (!kIsVariableB || !kIsVariableC) {
                        if constexpr (!kIsVariableB) {
                            dBC_val += (2 * dout_vals[i]) * conj(!kIsVariableC ? x : x * C_vals[i]);
                        } else {
                            dBC_val += (2 * dout_vals[i]) * conj(x);
                        }
                    }
                    const complex_t h_i = complex_t(thread_data[i].x, thread_data[i].y);
                    const float u_val = float(u_vals[i]);
                    const complex_t ONE(1.f, 0.f);
                    const float delta_for_A = (deltaA != nullptr) ? deltaA_vals[i] : delta_vals[i];
                    
                    if constexpr (DMethod == DiscretizationMethod::ZOH) {
                        const complex_t A_del_for_A = delta_for_A * A_val;
                        const complex_t E_for_A = cexpf(A_del_for_A);
                        
                        const complex_t A_del_for_B = delta_vals[i] * A_val;
                        const complex_t E_for_B = cexpf(A_del_for_B);

                        const complex_t Q = (E_for_B - complex_t(1.0f, 0.f)) / A_val;

                        const complex_t B_u = !kIsVariableB
                                              ? complex_t(float(u_vals[i]), 0.f)
                                              : B_vals[i] * float(u_vals[i]);

                        const complex_t I_term = Q * B_u;
                        const complex_t a = x - I_term;
                        const complex_t A_conj = conj(A_val);
                        const complex_t a_conj = conj(a);
                        const complex_t E_for_B_conj = conj(E_for_B);
                        const complex_t B_u_conj = conj(B_u);
                        const complex_t I_term_conj = conj(I_term);
                        const complex_t Q_conj = conj(Q);
                        if (deltaA != nullptr) {
                            const complex_t dDeltaA_trans = dx * A_conj * a_conj;
                            ddeltaA_vals[i] += dDeltaA_trans.real_;
                            
                            const complex_t dx_E_Bu_conj = dx * E_for_B_conj * B_u_conj;
                            ddelta_vals[i] += dx_E_Bu_conj.real_;
                        } else {
                            const complex_t dDelta_trans = dx * A_conj * a_conj;
                            const complex_t dx_E_Bu_conj = dx * E_for_B_conj * B_u_conj;
                            ddelta_vals[i] += (dDelta_trans + dx_E_Bu_conj).real_;
                        }

                        const complex_t dA_trans = delta_for_A * dx * a_conj;
                        const complex_t dx_E_Bu_conj = dx * E_for_B_conj * B_u_conj;
                        const complex_t numerator = delta_vals[i] * dx_E_Bu_conj - dx * I_term_conj;
                        const complex_t dA_input = numerator / A_conj;
                        dA_val += dA_trans + dA_input;

                        if constexpr (kIsVariableB) {
                            dB_vals[i] = dx * Q_conj * float(u_vals[i]);
                            du_vals[i] += (dx * Q_conj * conj(B_vals[i])).real_;
                        } else {
                            du_vals[i] += (dx * Q_conj).real_;
                        }
                    } else if constexpr (DMethod == DiscretizationMethod::BILINEAR) {
                        const complex_t a_half_for_A = 0.5f * delta_for_A * A_val;
                        const complex_t inv_term_for_A = complex_t(1.f, 0.f) / (complex_t(1.f, 0.f) - a_half_for_A);
                        const complex_t delta_a_exp = (complex_t(1.f, 0.f) + a_half_for_A) * inv_term_for_A;
                        
                        const complex_t a_half_for_B = 0.5f * delta_vals[i] * A_val;
                        const complex_t inv_term_for_B = complex_t(1.f, 0.f) / (complex_t(1.f, 0.f) - a_half_for_B);
                        
                        const complex_t B_u = !kIsVariableB
                                              ? complex_t(float(u_vals[i]), 0.f)
                                              : B_vals[i] * float(u_vals[i]);

                        const complex_t input_term = delta_vals[i] * inv_term_for_B * B_u;
                        const complex_t a = (x - input_term) / delta_a_exp;

                        const complex_t inv_term_sq_for_A = inv_term_for_A * inv_term_for_A;
                        const complex_t inv_term_sq_for_B = inv_term_for_B * inv_term_for_B;

                        const complex_t A_conj = conj(A_val);
                        const complex_t D_for_B_conj = conj(inv_term_for_B);
                        const complex_t B_u_conj = conj(B_u);
                        const complex_t a_conj = conj(a);

                        const complex_t dA_from_dynamics = a_conj * conj(inv_term_sq_for_A) * delta_for_A;
                        const complex_t dA_from_input = B_u_conj * conj(inv_term_sq_for_B) * (0.5f * delta_vals[i] * delta_vals[i]);
                        dA_val += dx * (dA_from_dynamics + dA_from_input);

                        if (deltaA != nullptr) {
                            const complex_t dDeltaA_dyn = dx * a_conj * A_conj * conj(inv_term_sq_for_A);
                            ddeltaA_vals[i] += dDeltaA_dyn.real_;
                            
                            const complex_t term1 = inv_term_for_B * B_u;
                            const complex_t term2 = delta_vals[i] * (0.5f * A_val * inv_term_sq_for_B) * B_u;
                            const complex_t dDelta_input = dx * conj(term1 + term2);
                            ddelta_vals[i] += dDelta_input.real_;
                        } else {
                            const complex_t dDelta_dyn = dx * a_conj * A_conj * conj(inv_term_sq_for_A);
                            const complex_t term1 = inv_term_for_B * B_u;
                            const complex_t term2 = delta_vals[i] * (0.5f * A_val * inv_term_sq_for_B) * B_u;
                            const complex_t dDelta_input = dx * conj(term1 + term2);
                            ddelta_vals[i] += (dDelta_dyn + dDelta_input).real_;
                        }

                        const complex_t d_input_common = dx * D_for_B_conj * delta_vals[i];
                        if constexpr (kIsVariableB) {
                            dB_vals[i] = d_input_common * float(u_vals[i]);
                            du_vals[i] += (d_input_common * conj(B_vals[i])).real_;
                        } else {
                            du_vals[i] += d_input_common.real_;
                        }
                    } else if constexpr (DMethod == DiscretizationMethod::DIRAC) {
                        const float delta_for_A = (deltaA != nullptr) ? deltaA_vals[i] : delta_vals[i];
                        const complex_t a_conj = conj(x - (!kIsVariableB ? float(u_vals[i]) : float(u_vals[i]) * B_vals[i]));
                        du_vals[i] += kIsVariableB ? (dx * conj(B_vals[i])).real_ : dx.real_;
                        if (deltaA != nullptr) {
                            ddeltaA_vals[i] += (dx * conj(A_val) * a_conj).real_;
                        } else {
                            ddelta_vals[i] += (dx * conj(A_val) * a_conj).real_;
                        }
                        dA_val += delta_for_A * dx * a_conj;
                        if constexpr (kIsVariableB) {
                            dB_vals[i] = dx * float(u_vals[i]);
                        }
                    } else if constexpr (DMethod == DiscretizationMethod::MAMBA) {
                        const float delta_for_A = (deltaA != nullptr) ? deltaA_vals[i] : delta_vals[i];
                        float ddelta_u = !kIsVariableB ? dx.real_ : (dx * conj(B_vals[i])).real_;
                        const complex_t a_conj = conj(x - (!kIsVariableB ? delta_vals[i] * float(u_vals[i]) : delta_vals[i] * float(u_vals[i]) * B_vals[i]));
                        du_vals[i] += ddelta_u * delta_vals[i];
                        ddelta_vals[i] += ddelta_u * float(u_vals[i]);
                        if (deltaA != nullptr) {
                            ddeltaA_vals[i] += (dx * conj(A_val) * a_conj).real_;
                        } else {
                            ddelta_vals[i] += (dx * conj(A_val) * a_conj).real_;
                        }
                        dA_val += delta_for_A * dx * a_conj;
                        if constexpr (kIsVariableB) {
                            dB_vals[i] = dx * delta_vals[i] * float(u_vals[i]);
                        }
                    } else if constexpr (DMethod == DiscretizationMethod::S7) {
                        static_assert(!kIsComplex, "S7 discretization only supports real weights");
                    }

                    if constexpr (kIsVariableC) {
                        dC_vals[i] = (2 * dout_vals[i]) * conj(!kIsVariableB ? x * B_val : x);
                    }
                }
                // block-exchange to make the atomicAdd's coalesced, otherwise they're much slower
                if constexpr (kIsVariableB || kIsVariableC) {
                    float dB_vals_f[kNItems * 2], dC_vals_f[kNItems * 2];
                    if constexpr (kIsVariableB) {
                        #pragma unroll
                        for (int i = 0; i < kNItems; ++i) {
                            dB_vals_f[i * 2] = dB_vals[i].real_;
                            dB_vals_f[i * 2 + 1] = dB_vals[i].imag_;
                        }
                        typename Ktraits::BlockExchangeT(smem_exchange).BlockedToStriped(dB_vals_f, dB_vals_f);
                    }
                    if constexpr (kIsVariableC) {
                        #pragma unroll
                        for (int i = 0; i < kNItems; ++i) {
                            dC_vals_f[i * 2] = dC_vals[i].real_;
                            dC_vals_f[i * 2 + 1] = dC_vals[i].imag_;
                        }
                        auto &smem_exchange_C = !kIsVariableB ? smem_exchange : smem_exchange1;
                        typename Ktraits::BlockExchangeT(smem_exchange_C).BlockedToStriped(dC_vals_f, dC_vals_f);
                    }
                    const int seqlen_remaining = (params.seqlen - chunk * kChunkSize) * 2 - threadIdx.x;
                    float *dB_cur = reinterpret_cast<float *>(dB) + state_idx * params.dB_dstate_stride + chunk * kChunkSize * 2 + threadIdx.x;
                    float *dC_cur = reinterpret_cast<float *>(dC) + state_idx * params.dC_dstate_stride + chunk * kChunkSize * 2 + threadIdx.x;
                    #pragma unroll
                    for (int i = 0; i < kNItems * 2; ++i) {
                        if (i * kNThreads < seqlen_remaining) {
                            if constexpr (kIsVariableB) { gpuAtomicAdd(dB_cur + i * kNThreads, dB_vals_f[i]); }
                            if constexpr (kIsVariableC) { gpuAtomicAdd(dC_cur + i * kNThreads, dC_vals_f[i]); }
                        }
                    }
                }
                if constexpr (!kIsVariableB || !kIsVariableC) {
                    float4 dA_dBC_val = make_float4(dA_val.real_, dA_val.imag_, dBC_val.real_, dBC_val.imag_);
                    dA_dBC_val = typename Ktraits::BlockReduceT(smem_reduce).Sum(dA_dBC_val);
                    dA_val = complex_t(dA_dBC_val.x, dA_dBC_val.y);
                    dBC_val = complex_t(dA_dBC_val.z, dA_dBC_val.w);
                    if (threadIdx.x == 0) {
                        smem_dbc[state_idx] = chunk == params.n_chunks - 1 ? dBC_val : dBC_val + smem_dbc[state_idx];
                    }
                } else {
                    dA_val = typename Ktraits::BlockReduceComplexT(smem_reduce_complex).Sum(dA_val);
                }
                if (threadIdx.x == 0) {
                    smem_da[state_idx] = chunk == params.n_chunks - 1 ? dA_val : dA_val + smem_da[state_idx];
                }
            }
        }

        if constexpr (kDeltaSoftplus) {
            __syncthreads();
            input_t delta_vals_load[kNItems];
            load_input<Ktraits>(delta, delta_vals_load, smem_load, params.seqlen - chunk * kChunkSize);
            delta -= kChunkSize;
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                float delta_val = float(delta_vals_load[i]) + delta_bias;
                float delta_val_neg_exp = expf(-delta_val);
                ddelta_vals[i] = delta_val <= 20.f
                    ? ddelta_vals[i] / (1.f + delta_val_neg_exp)
                    : ddelta_vals[i];
            }
        }
        for (int i = 0; i < kNItems; ++i) { ddelta_bias_val += ddelta_vals[i]; }

        input_t *du = reinterpret_cast<input_t *>(params.du_ptr) + batch_id * params.du_batch_stride
            + dim_id * params.du_d_stride + chunk * kChunkSize;
        input_t *ddelta = reinterpret_cast<input_t *>(params.ddelta_ptr) + batch_id * params.ddelta_batch_stride
            + dim_id * params.ddelta_d_stride + chunk * kChunkSize;
        __syncthreads();
        store_output<Ktraits>(du, du_vals, smem_store, params.seqlen - chunk * kChunkSize);
        __syncthreads();
        store_output<Ktraits>(ddelta, ddelta_vals, smem_store, params.seqlen - chunk * kChunkSize);
        // store ddeltaA if it's provided
        if (params.ddeltaA_ptr != nullptr) {
            input_t *ddeltaA = reinterpret_cast<input_t *>(params.ddeltaA_ptr) + batch_id * params.ddeltaA_batch_stride
                + dim_id * params.ddeltaA_d_stride + chunk * kChunkSize;
            __syncthreads();
            store_output<Ktraits>(ddeltaA, ddeltaA_vals, smem_store, params.seqlen - chunk * kChunkSize);
        }

        Bvar -= kChunkSize * (!kIsComplex ? 1 : 2);
        Cvar -= kChunkSize * (!kIsComplex ? 1 : 2);
    }
    if (params.dD_ptr != nullptr) {
        dD_val = typename Ktraits::BlockReduceFloatT(smem_reduce_float).Sum(dD_val);
        if (threadIdx.x == 0) { gpuAtomicAdd(dD, dD_val); }
    }
    if (params.ddelta_bias_ptr != nullptr) {
        __syncthreads();
        ddelta_bias_val = typename Ktraits::BlockReduceFloatT(smem_reduce_float).Sum(ddelta_bias_val);
        if (threadIdx.x == 0) { gpuAtomicAdd(ddelta_bias, ddelta_bias_val); }
    }
    for (int state_idx = threadIdx.x; state_idx < params.dstate; state_idx += blockDim.x) {
        gpuAtomicAdd(&(dA[state_idx * params.dA_dstate_stride]), smem_da[state_idx]);
        weight_t dBC_val;
        if (!kIsVariableB || !kIsVariableC) { dBC_val = smem_dbc[state_idx]; }
        if constexpr (!kIsVariableB) {
            gpuAtomicAdd(&(dB[state_idx * params.dB_dstate_stride]),
                         !kIsVariableC ? dBC_val * conj(C[state_idx * params.C_dstate_stride]) : dBC_val);
        }
        if constexpr (!kIsVariableC) {
            gpuAtomicAdd(&(dC[state_idx * params.dC_dstate_stride]),
                        !kIsVariableB ? dBC_val * conj(B[state_idx * params.B_dstate_stride]) : dBC_val);
        }
    }
}

////////////////////////////////////////////////////////////////////////////

template<int kNThreads, int kNItems, typename input_t, typename weight_t, DiscretizationMethod DMethod>
void selective_scan_bwd_launch(SSMParamsBwd &params, cudaStream_t stream) {
    BOOL_SWITCH(params.seqlen % (kNThreads * kNItems) == 0, kIsEvenLen, [&] {
        BOOL_SWITCH(params.is_variable_B, kIsVariableB, [&] {
            BOOL_SWITCH(params.is_variable_C, kIsVariableC, [&] {
                BOOL_SWITCH(params.delta_softplus, kDeltaSoftplus, [&] {
                    BOOL_SWITCH(params.z_ptr != nullptr , kHasZ, [&] {
                        using Ktraits = Selective_Scan_bwd_kernel_traits<
                            kNThreads, kNItems, kIsEvenLen, kIsVariableB, kIsVariableC,
                            kDeltaSoftplus, kHasZ, input_t, weight_t, DMethod
                        >;

                        constexpr int kSmemSize = Ktraits::kSmemSize
                            + MAX_DSTATE * sizeof(typename Ktraits::scan_t)
                            + (kNThreads + 4 * MAX_DSTATE) * sizeof(typename Ktraits::weight_t);

                        dim3 grid(params.batch, params.dim);

                        auto kernel = &selective_scan_bwd_kernel<Ktraits>;

                        if (kSmemSize >= 48 * 1024) {
                            #ifndef USE_ROCM
                            C10_CUDA_CHECK(cudaFuncSetAttribute(
                                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
                            #else
                            C10_CUDA_CHECK(cudaFuncSetAttribute(
                                (void *) kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
                            std::cerr << "Warning (selective_scan_bwd_kernel): attempting to set maxDynamicSharedMemorySize on an AMD GPU which is currently a non-op (in ROCm versions <= 6.1). This might lead to undefined behavior. \n" << std::endl;
                            #endif
                        }

                        kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
                        C10_CUDA_KERNEL_LAUNCH_CHECK();
                    });
                });
            });
        });
    });
}

template<typename input_t, typename weight_t, DiscretizationMethod DMethod>
void selective_scan_bwd_cuda(SSMParamsBwd &params, cudaStream_t stream) {

    #ifndef USE_ROCM
        if (params.seqlen <= 128) {

            selective_scan_bwd_launch<32, 4, input_t, weight_t, DMethod>(params, stream);
        } else if (params.seqlen <= 256) {

            selective_scan_bwd_launch<32, 8, input_t, weight_t, DMethod>(params, stream);
        } else if (params.seqlen <= 512) {

            selective_scan_bwd_launch<32, 16, input_t, weight_t, DMethod>(params, stream);
        } else if (params.seqlen <= 1024) {

            selective_scan_bwd_launch<64, 16, input_t, weight_t, DMethod>(params, stream);
        } else {

            selective_scan_bwd_launch<128, 16, input_t, weight_t, DMethod>(params, stream);
        }
    #else
        if (params.seqlen <= 256) {

            selective_scan_bwd_launch<64, 4, input_t, weight_t, DMethod>(params, stream);
        } else if (params.seqlen <= 512) {

            selective_scan_bwd_launch<64, 8, input_t, weight_t, DMethod>(params, stream);
        } else if (params.seqlen <= 1024) {

            selective_scan_bwd_launch<64, 16, input_t, weight_t, DMethod>(params, stream);
        } else {

            selective_scan_bwd_launch<128, 16, input_t, weight_t, DMethod>(params, stream);
        }
    #endif
}
