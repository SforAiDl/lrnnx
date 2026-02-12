/******************************************************************************
 * Copyright (c) 2026, LRNNx Team.
 * Simplified Scan Backward Pass
 *
 * This file provides backward pass support for complex-valued inputs (u).
 ******************************************************************************/

#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/Atomic.cuh>

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
#include "simplified_scan.h"
#include "../reverse_scan.cuh"

template<typename scalar_t> __device__ __forceinline__ scalar_t conj(scalar_t x);
template<> __device__ __forceinline__ float conj<float>(float x) { return x; }
template<> __device__ __forceinline__ complex_t conj<complex_t>(complex_t x) { return std::conj(x); }


template<int kNThreads_, int kNItems_, bool kIsEvenLen_, DiscretizationMethod DMethod_>
struct Simplified_Scan_bwd_kernel_traits {
    static_assert(kNItems_ % 4 == 0);
    using input_t = float;
    using weight_t = complex_t;
    static constexpr int kNThreads = kNThreads_;
    static constexpr int kNItems = kNItems_;
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 4);
    static constexpr int kNElts = 4;
    static_assert(kNItems % kNElts == 0);
    static constexpr int kNLoads = kNItems / kNElts;
    static constexpr bool kIsEvenLen = kIsEvenLen_;
    static constexpr DiscretizationMethod DMethod = DMethod_;
    static constexpr int kMinBlocks = kNThreads == 128 && 2;
    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    using scan_t = float4;
    using BlockLoadT = cub::BlockLoad<float, kNThreads, kNItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadVecT = cub::BlockLoad<vec_t, kNThreads, kNLoads, cub::BLOCK_LOAD_WARP_TRANSPOSE>;

    // For loading complex values (u, dout)
    using BlockLoadComplexT = cub::BlockLoad<float, kNThreads, kNItems * 2, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadComplexVecT = cub::BlockLoad<vec_t, kNThreads, kNLoads * 2, cub::BLOCK_LOAD_WARP_TRANSPOSE>;

    // For loading weights
    using BlockLoadWeightT = cub::BlockLoad<float, kNThreads, kNItems * 2, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadWeightVecT = cub::BlockLoad<vec_t, kNThreads, kNLoads * 2, cub::BLOCK_LOAD_WARP_TRANSPOSE>;

    // For storing
    using BlockStoreT = cub::BlockStore<float, kNThreads, kNItems, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockStoreVecT = cub::BlockStore<vec_t, kNThreads, kNLoads, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockStoreComplexT = cub::BlockStore<float, kNThreads, kNItems * 2, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockStoreComplexVecT = cub::BlockStore<vec_t, kNThreads, kNLoads * 2, cub::BLOCK_STORE_WARP_TRANSPOSE>;

    using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_RAKING>;
    using BlockReverseScanT = BlockReverseScan<scan_t, kNThreads>;
    using BlockReduceT = cub::BlockReduce<scan_t, kNThreads>;
    using BlockReduceFloatT = cub::BlockReduce<float, kNThreads>;
    using BlockReduceComplexT = cub::BlockReduce<complex_t, kNThreads>;
    using BlockExchangeT = cub::BlockExchange<float, kNThreads, kNItems * 2>;

    static constexpr int kSmemIOSize = custom_max({sizeof(typename BlockLoadT::TempStorage),
                                                    sizeof(typename BlockLoadVecT::TempStorage),
                                                    sizeof(typename BlockLoadComplexT::TempStorage),
                                                    sizeof(typename BlockLoadComplexVecT::TempStorage),
                                                    sizeof(typename BlockLoadWeightT::TempStorage),
                                                    sizeof(typename BlockLoadWeightVecT::TempStorage),
                                                    sizeof(typename BlockStoreT::TempStorage),
                                                    sizeof(typename BlockStoreVecT::TempStorage),
                                                    sizeof(typename BlockStoreComplexT::TempStorage),
                                                    sizeof(typename BlockStoreComplexVecT::TempStorage)});
    static constexpr int kSmemReduceSize = sizeof(typename BlockReduceT::TempStorage);
    static constexpr int kSmemSize = kSmemIOSize + kSmemReduceSize + sizeof(typename BlockScanT::TempStorage) + sizeof(typename BlockReverseScanT::TempStorage);
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads, Ktraits::kMinBlocks)
void simplified_scan_bwd_kernel(SSMParamsBwd params) {
    constexpr DiscretizationMethod DMethod = Ktraits::DMethod;
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kNItems = Ktraits::kNItems;
    using scan_t = typename Ktraits::scan_t;

    // Shared memory.
    extern __shared__ char smem_[];
    auto& smem_load = reinterpret_cast<typename Ktraits::BlockLoadT::TempStorage&>(smem_);
    auto& smem_load_complex = reinterpret_cast<typename Ktraits::BlockLoadComplexT::TempStorage&>(smem_);
    auto& smem_load_weight = reinterpret_cast<typename Ktraits::BlockLoadWeightT::TempStorage&>(smem_);
    auto& smem_load_weight1 = *reinterpret_cast<typename Ktraits::BlockLoadWeightT::TempStorage*>(
        smem_ + sizeof(typename Ktraits::BlockLoadWeightT::TempStorage));
    auto& smem_store = reinterpret_cast<typename Ktraits::BlockStoreT::TempStorage&>(smem_);
    auto& smem_store_complex = reinterpret_cast<typename Ktraits::BlockStoreComplexT::TempStorage&>(smem_);
    auto& smem_exchange = *reinterpret_cast<typename Ktraits::BlockExchangeT::TempStorage*>(smem_ + Ktraits::kSmemIOSize);
    auto& smem_exchange1 = *reinterpret_cast<typename Ktraits::BlockExchangeT::TempStorage*>(
        smem_ + Ktraits::kSmemIOSize + sizeof(typename Ktraits::BlockExchangeT::TempStorage));
    auto& smem_reduce = *reinterpret_cast<typename Ktraits::BlockReduceT::TempStorage*>(
        reinterpret_cast<char*>(&smem_exchange));
    auto& smem_reduce_float = *reinterpret_cast<typename Ktraits::BlockReduceFloatT::TempStorage*>(&smem_reduce);
    auto& smem_reduce_complex = *reinterpret_cast<typename Ktraits::BlockReduceComplexT::TempStorage*>(&smem_reduce);
    auto& smem_scan = *reinterpret_cast<typename Ktraits::BlockScanT::TempStorage*>(
        reinterpret_cast<char*>(&smem_reduce) + Ktraits::kSmemReduceSize);
    auto& smem_reverse_scan = *reinterpret_cast<typename Ktraits::BlockReverseScanT::TempStorage*>(
        reinterpret_cast<char*>(&smem_scan) + sizeof(typename Ktraits::BlockScanT::TempStorage));

    complex_t *smem_delta_a = reinterpret_cast<complex_t*>(smem_ + Ktraits::kSmemSize);
    scan_t *smem_running_postfix = reinterpret_cast<scan_t*>(smem_delta_a + 2 * MAX_DSTATE + kNThreads);
    complex_t *smem_da = reinterpret_cast<complex_t*>(smem_running_postfix + MAX_DSTATE);
    complex_t *smem_db = reinterpret_cast<complex_t*>(smem_da + MAX_DSTATE);
    complex_t *smem_dc = reinterpret_cast<complex_t*>(smem_db + MAX_DSTATE);

    const int batch_id = blockIdx.x;
    const int dim_id = blockIdx.y;
    const int group_id = dim_id / (params.dim_ngroups_ratio);
    float *u = reinterpret_cast<float*>(params.u_ptr) + batch_id * params.u_batch_stride * 2
        + dim_id * params.u_d_stride * 2;
    float *dout = reinterpret_cast<float*>(params.dout_ptr) + batch_id * params.dout_batch_stride * 2
        + dim_id * params.dout_d_stride * 2;
    float *du = reinterpret_cast<float*>(params.du_ptr) + batch_id * params.du_batch_stride * 2
        + dim_id * params.du_d_stride * 2;
    float *delta = reinterpret_cast<float*>(params.delta_ptr) + batch_id * params.delta_batch_stride
        + dim_id * params.delta_d_stride;
    float *deltaA = params.deltaA_ptr != nullptr 
        ? reinterpret_cast<float*>(params.deltaA_ptr) + batch_id * params.deltaA_batch_stride
            + dim_id * params.deltaA_d_stride 
        : nullptr;
    float *ddelta = reinterpret_cast<float*>(params.ddelta_ptr) + batch_id * params.ddelta_batch_stride
        + dim_id * params.ddelta_d_stride;
    float *ddeltaA = params.ddeltaA_ptr != nullptr
        ? reinterpret_cast<float*>(params.ddeltaA_ptr) + batch_id * params.ddeltaA_batch_stride
            + dim_id * params.ddeltaA_d_stride
        : nullptr;
    complex_t *A = reinterpret_cast<complex_t*>(params.A_ptr) + dim_id * params.A_d_stride;
    complex_t *B = reinterpret_cast<complex_t*>(params.B_ptr) + dim_id * params.B_d_stride;
    complex_t *C = reinterpret_cast<complex_t*>(params.C_ptr) + dim_id * params.C_d_stride;
    complex_t *dA = reinterpret_cast<complex_t*>(params.dA_ptr) + dim_id * params.dA_d_stride;
    complex_t *dB = reinterpret_cast<complex_t*>(params.dB_ptr) + dim_id * params.dB_d_stride + group_id * params.dB_group_stride;
    complex_t *dC = reinterpret_cast<complex_t*>(params.dC_ptr) + dim_id * params.dC_d_stride + group_id * params.dC_group_stride;

    scan_t *x = params.x_ptr == nullptr
        ? nullptr
        : reinterpret_cast<scan_t*>(params.x_ptr) + (batch_id * params.dim + dim_id) * params.n_chunks * params.dstate;

    constexpr int kChunkSize = kNThreads * kNItems;
    u += (params.n_chunks - 1) * kChunkSize * 2;
    delta += (params.n_chunks - 1) * kChunkSize;
    if (deltaA != nullptr) {
        deltaA += (params.n_chunks - 1) * kChunkSize;
    }
    dout += (params.n_chunks - 1) * kChunkSize * 2;
    for (int chunk = params.n_chunks - 1; chunk >= 0; --chunk) {
        complex_t u_vals[kNItems];
        float delta_vals_load[kNItems], deltaA_vals_load[kNItems];
        complex_t dout_vals[kNItems];
        __syncthreads();
        load_complex_input<Ktraits>(u, u_vals, smem_load_complex, params.seqlen - chunk * kChunkSize);
        u -= kChunkSize * 2;
        __syncthreads();
        load_input<Ktraits>(delta, delta_vals_load, smem_load, params.seqlen - chunk * kChunkSize);
        delta -= kChunkSize;
        // Load deltaA if provided
        if (deltaA != nullptr) {
            __syncthreads();
            load_input<Ktraits>(deltaA, deltaA_vals_load, smem_load, params.seqlen - chunk * kChunkSize);
            deltaA -= kChunkSize;
        }
        __syncthreads();
        load_complex_input<Ktraits>(dout, dout_vals, smem_load_complex, params.seqlen - chunk * kChunkSize);
        dout -= kChunkSize * 2;

        float delta_vals[kNItems], deltaA_vals[kNItems];
        #pragma unroll
        for (int i = 0; i < kNItems; ++i) {
            delta_vals[i] = float(delta_vals_load[i]);
            // Load deltaA values if provided
            if (deltaA != nullptr) {
                deltaA_vals[i] = float(deltaA_vals_load[i]);
            }
        }
        complex_t du_vals[kNItems];
        #pragma unroll
        for (int i = 0; i < kNItems; ++i) {
            du_vals[i] = complex_t(0.f, 0.f);
        }
        float ddelta_vals[kNItems] = {0};
        float ddeltaA_vals[kNItems] = {0};
        __syncthreads();
        for (int state_idx = 0; state_idx < params.dstate; ++state_idx) {
            const complex_t A_val = A[state_idx * params.A_dstate_stride];
            complex_t B_val, C_val;
            B_val = B[state_idx * params.B_dstate_stride];
            C_val = C[state_idx * params.C_dstate_stride];
            scan_t thread_data[kNItems], thread_reverse_data[kNItems];
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                complex_t A_bar;
                complex_t B_u_term;
                const complex_t one = complex_t(1.0f, 0.0f);
                const complex_t B_term = one;
                // Use deltaA for A discretization if provided, otherwise use delta
                const float delta_for_A = (deltaA != nullptr) ? deltaA_vals[i] : delta_vals[i];
                if constexpr (DMethod == DiscretizationMethod::ZOH) {
                    const complex_t A_del_for_A = A_val * delta_for_A;
                    A_bar = cexpf(A_del_for_A);
                    const complex_t A_del_for_B = A_val * delta_vals[i];
                    const complex_t A_disc_for_B = cexpf(A_del_for_B);
                    const complex_t B_tilde = (A_disc_for_B - one) / A_val;
                    B_u_term = B_term * B_tilde * u_vals[i];
                } else if constexpr (DMethod == DiscretizationMethod::BILINEAR) {
                    const complex_t v_for_A = 0.5f * delta_for_A * A_val;
                    const complex_t den_inv_for_A = one / (one - v_for_A);
                    A_bar = (one + v_for_A) * den_inv_for_A;
                    const complex_t v_for_B = 0.5f * delta_vals[i] * A_val;
                    const complex_t den_inv_for_B = one / (one - v_for_B);
                    B_u_term = B_term * den_inv_for_B * delta_vals[i] * u_vals[i];
                } else if constexpr (DMethod == DiscretizationMethod::DIRAC) {
                    const complex_t A_del = A_val * delta_for_A;
                    A_bar = cexpf(A_del);
                    B_u_term = B_term * u_vals[i];
                }
                thread_data[i] = make_float4(A_bar.real_, A_bar.imag_, B_u_term.real_, B_u_term.imag_);
                const complex_t C_val_use = C_val;
                const complex_t dout_C = dout_vals[i] * conj(C_val_use);
                if (i == 0) {
                    smem_delta_a[threadIdx.x == 0 ? state_idx + (chunk % 2) * MAX_DSTATE : threadIdx.x + 2 * MAX_DSTATE] = A_bar;
                } else {
                    thread_reverse_data[i - 1].x = conj(A_bar).real_;
                    thread_reverse_data[i - 1].y = conj(A_bar).imag_;
                }
                thread_reverse_data[i].z = dout_C.real_;
                thread_reverse_data[i].w = dout_C.imag_;
            }
            __syncthreads();
            complex_t delta_a_exp_next = threadIdx.x == kNThreads - 1
                ? (chunk == params.n_chunks - 1 ? complex_t(1.f, 0.f) : smem_delta_a[state_idx + ((chunk + 1) % 2) * MAX_DSTATE])
                : smem_delta_a[threadIdx.x + 1 + 2 * MAX_DSTATE];
            thread_reverse_data[kNItems - 1].x = conj(delta_a_exp_next).real_;
            thread_reverse_data[kNItems - 1].y = conj(delta_a_exp_next).imag_;

            scan_t running_prefix = chunk > 0 && threadIdx.x % 32 == 0
                ? x[(chunk - 1) * params.dstate + state_idx] : make_float4(1.f, 0.f, 0.f, 0.f);
            SSMScanPrefixCallbackOp<complex_t> prefix_op(running_prefix);
            typename Ktraits::BlockScanT(smem_scan).InclusiveScan(
                thread_data, thread_data, SSMScanOp<complex_t>(), prefix_op
            );

            scan_t running_postfix = chunk < params.n_chunks - 1 && threadIdx.x % 32 == 0
                ? smem_running_postfix[state_idx] : make_float4(1.f, 0.f, 0.f, 0.f);
            SSMScanPrefixCallbackOp<complex_t> postfix_op(running_postfix);
            typename Ktraits::BlockReverseScanT(smem_reverse_scan).InclusiveReverseScan(
                thread_reverse_data, thread_reverse_data, SSMScanOp<complex_t>(), postfix_op
            );
            if (threadIdx.x == 0) { smem_running_postfix[state_idx] = postfix_op.running_prefix; }

            // Compute gradients
            complex_t dA_val = complex_t(0.f, 0.f);
            complex_t dB_val = complex_t(0.f, 0.f);
            complex_t dC_val = complex_t(0.f, 0.f);

            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                const complex_t x_state = complex_t(thread_data[i].z, thread_data[i].w);
                const complex_t dx = complex_t(thread_reverse_data[i].z, thread_reverse_data[i].w);
                const complex_t one = complex_t(1.0f, 0.0f);
                const complex_t B_term = one;
                // Use deltaA for A discretization if provided, otherwise use delta
                const float delta_for_A = (deltaA != nullptr) ? deltaA_vals[i] : delta_vals[i];

                if constexpr (DMethod == DiscretizationMethod::ZOH) {
                    // ZOH gradient computation
                    // A_bar = exp(A * delta_for_A)
                    // B_tilde = (exp(A * delta) - 1) / A
                    const complex_t A_del_for_A = A_val * delta_for_A;
                    const complex_t A_bar = cexpf(A_del_for_A);
                    const complex_t A_del_for_B = A_val * delta_vals[i];
                    const complex_t A_disc_for_B = cexpf(A_del_for_B);
                    const complex_t B_tilde = (A_disc_for_B - one) / A_val;
                    const complex_t B_u_term = B_term * B_tilde * u_vals[i];

                    // x_prev = (x_state - B_u_term) / A_bar
                    const complex_t x_prev = (x_state - B_u_term) / A_bar;
                    du_vals[i] = du_vals[i] + conj(B_term * B_tilde) * dx;
                    
                    if (deltaA != nullptr) {
                        // deltaA affects A_bar: d/d(deltaA) = A * A_bar * x_prev
                        const complex_t ddeltaA_term = A_val * A_bar * x_prev;
                        ddeltaA_vals[i] += (dx * conj(ddeltaA_term)).real_;
                        // delta affects B_bar through A_disc_for_B: d/d(delta) = B * u * A_disc_for_B
                        const complex_t ddelta_term = B_term * u_vals[i] * A_disc_for_B;
                        ddelta_vals[i] += (dx * conj(ddelta_term)).real_;
                    } else {
                        // Both A_bar and B_bar use delta
                        const complex_t ddelta_term = A_bar * (A_val * x_prev + B_term * u_vals[i]);
                        ddelta_vals[i] += (dx * conj(ddelta_term)).real_;
                    }
                    
                    const complex_t dA_bar_dA = delta_for_A * A_bar;
                    const complex_t dB_tilde_dA = (delta_vals[i] * A_disc_for_B - B_tilde) / A_val;
                    const complex_t dA_term = dA_bar_dA * x_prev + B_term * dB_tilde_dA * u_vals[i];
                    dA_val += dx * conj(dA_term);
                    dB_val = dB_val + dx * conj(B_tilde * u_vals[i]);

                } else if constexpr (DMethod == DiscretizationMethod::BILINEAR) {
                    const complex_t a_half_for_A = 0.5f * delta_for_A * A_val;
                    const complex_t inv_term_for_A = one / (one - a_half_for_A);
                    const complex_t A_bar = (one + a_half_for_A) * inv_term_for_A;
                    const complex_t a_half_for_B = 0.5f * delta_vals[i] * A_val;
                    const complex_t inv_term_for_B = one / (one - a_half_for_B);
                    const complex_t B_u = u_vals[i];
                    const complex_t input_term = delta_vals[i] * inv_term_for_B * B_u;
                    const complex_t x_prev = (x_state - input_term) / A_bar;
                    const complex_t inv_term_sq_for_A = inv_term_for_A * inv_term_for_A;
                    const complex_t inv_term_sq_for_B = inv_term_for_B * inv_term_for_B;

                    const complex_t A_conj = conj(A_val);
                    const complex_t B_u_conj = conj(B_u);
                    const complex_t x_prev_conj = conj(x_prev);

                    // dA has two contributions: dynamics (A_bar) and input (B_bar)
                    const complex_t dA_from_dynamics = x_prev_conj * conj(inv_term_sq_for_A) * delta_for_A;
                    const complex_t dA_from_input = B_u_conj * conj(inv_term_sq_for_B)
                        * (0.5f * delta_vals[i] * delta_vals[i]);
                    dA_val += dx * (dA_from_dynamics + dA_from_input);

                    // deltaA affects A_bar only; delta affects B_bar (and A_bar if no deltaA)
                    const complex_t term1 = inv_term_for_B * B_u;
                    const complex_t term2 = delta_vals[i] * (0.5f * A_val * inv_term_sq_for_B) * B_u;
                    const complex_t dDelta_input = dx * conj(term1 + term2);

                    if (deltaA != nullptr) {
                        const complex_t dDeltaA_dyn = dx * x_prev_conj * A_conj * conj(inv_term_sq_for_A);
                        ddeltaA_vals[i] += dDeltaA_dyn.real_;
                        ddelta_vals[i] += dDelta_input.real_;
                    } else {
                        const complex_t dDelta_dyn = dx * x_prev_conj * A_conj * conj(inv_term_sq_for_A);
                        ddelta_vals[i] += (dDelta_dyn + dDelta_input).real_;
                    }

                    du_vals[i] = du_vals[i] + conj(B_term * inv_term_for_B * delta_vals[i]) * dx;
                    dB_val = dB_val + dx * conj(inv_term_for_B * delta_vals[i] * u_vals[i]);

                } else if constexpr (DMethod == DiscretizationMethod::DIRAC) {
                    // A_bar = exp(A * delta_for_A), uses deltaA if provided
                    // B_bar = B (doesn't use delta)
                    const complex_t A_del = A_val * delta_for_A;
                    const complex_t A_bar = cexpf(A_del);
                    const complex_t B_u_term = B_term * u_vals[i];
                    const complex_t x_prev = (x_state - B_u_term) / A_bar;
                    du_vals[i] = du_vals[i] + conj(B_term) * dx;
                    
                    // For DIRAC, only A_bar depends on delta (or deltaA)
                    const complex_t ddelta_term = A_val * A_bar * x_prev;
                    if (deltaA != nullptr) {
                        // deltaA affects A_bar
                        ddeltaA_vals[i] += (dx * conj(ddelta_term)).real_;
                        // delta doesn't affect B_bar in DIRAC, so ddelta_vals[i] += 0
                    } else {
                        // delta affects A_bar
                        ddelta_vals[i] += (dx * conj(ddelta_term)).real_;
                    }
                    
                    const complex_t dA_term = delta_for_A * A_bar * x_prev;
                    dA_val += dx * conj(dA_term);
                    dB_val = dB_val + dx * conj(u_vals[i]);
                }
                dC_val = dC_val + dout_vals[i] * conj(x_state);
            }

            dA_val = typename Ktraits::BlockReduceComplexT(smem_reduce_complex).Sum(dA_val);
            if (threadIdx.x == 0) {
                smem_da[state_idx] = chunk == params.n_chunks - 1 ? dA_val : dA_val + smem_da[state_idx];
            }
            __syncthreads();

            dB_val = typename Ktraits::BlockReduceComplexT(smem_reduce_complex).Sum(dB_val);
            if (threadIdx.x == 0) {
                smem_db[state_idx] = chunk == params.n_chunks - 1 ? dB_val : dB_val + smem_db[state_idx];
            }
            __syncthreads();

            dC_val = typename Ktraits::BlockReduceComplexT(smem_reduce_complex).Sum(dC_val);
            if (threadIdx.x == 0) {
                smem_dc[state_idx] = chunk == params.n_chunks - 1 ? dC_val : dC_val + smem_dc[state_idx];
            }
            __syncthreads();

        }
        float *du_chunk = du + chunk * kChunkSize * 2;
        __syncthreads();
        store_complex_output<Ktraits>(du_chunk, du_vals, smem_store_complex, params.seqlen - chunk * kChunkSize);
        float *ddelta_chunk = ddelta + chunk * kChunkSize;
        __syncthreads();
        store_output<Ktraits>(ddelta_chunk, ddelta_vals, smem_store, params.seqlen - chunk * kChunkSize);
        if (ddeltaA != nullptr) {
            float *ddeltaA_chunk = ddeltaA + chunk * kChunkSize;
            __syncthreads();
            store_output<Ktraits>(ddeltaA_chunk, ddeltaA_vals, smem_store, params.seqlen - chunk * kChunkSize);
        }
    }
    __syncthreads();
    for (int state_idx = threadIdx.x; state_idx < params.dstate; state_idx += kNThreads) {
        gpuAtomicAdd(dA + state_idx * params.dA_dstate_stride, smem_da[state_idx]);
        gpuAtomicAdd(dB + state_idx * params.dB_dstate_stride, smem_db[state_idx]);
        gpuAtomicAdd(dC + state_idx * params.dC_dstate_stride, smem_dc[state_idx]);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<int kNThreads, int kNItems, DiscretizationMethod DMethod>
void simplified_scan_bwd_launch(SSMParamsBwd &params, cudaStream_t stream) {
    BOOL_SWITCH(params.seqlen % (kNThreads * kNItems) == 0, kIsEvenLen, [&] {
        using Ktraits = Simplified_Scan_bwd_kernel_traits<kNThreads, kNItems, kIsEvenLen, DMethod>;

        // smem_delta_a: (2*MAX_DSTATE + kNThreads) * sizeof(complex_t)
        // smem_running_postfix: MAX_DSTATE * sizeof(scan_t)
        // smem_da, smem_db, smem_dc: 3 * MAX_DSTATE * sizeof(complex_t)
        constexpr int kSmemSize = Ktraits::kSmemSize
                                + (2 * MAX_DSTATE + kNThreads) * sizeof(complex_t)
                                + MAX_DSTATE * sizeof(typename Ktraits::scan_t)
                                + 3 * MAX_DSTATE * sizeof(complex_t);
        dim3 grid(params.batch, params.dim);

        auto kernel = &simplified_scan_bwd_kernel<Ktraits>;

        if (kSmemSize >= 48 * 1024) {
            #ifndef USE_ROCM
            C10_CUDA_CHECK(cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
            #else
            C10_CUDA_CHECK(cudaFuncSetAttribute(
                (void *) kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
            #endif
        }

        kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<DiscretizationMethod DMethod>
void simplified_scan_bwd_cuda(SSMParamsBwd &params, cudaStream_t stream) {
    #ifndef USE_ROCM
        if (params.seqlen <= 128) {
            simplified_scan_bwd_launch<32, 4, DMethod>(params, stream);
        } else if (params.seqlen <= 256) {
            simplified_scan_bwd_launch<32, 8, DMethod>(params, stream);
        } else if (params.seqlen <= 512) {
            simplified_scan_bwd_launch<32, 16, DMethod>(params, stream);
        } else if (params.seqlen <= 1024) {
            simplified_scan_bwd_launch<64, 16, DMethod>(params, stream);
        } else {
            simplified_scan_bwd_launch<128, 16, DMethod>(params, stream);
        }
    #else
        if (params.seqlen <= 256) {
            simplified_scan_bwd_launch<64, 4, DMethod>(params, stream);
        } else if (params.seqlen <= 512) {
            simplified_scan_bwd_launch<64, 8, DMethod>(params, stream);
        } else if (params.seqlen <= 1024) {
            simplified_scan_bwd_launch<64, 16, DMethod>(params, stream);
        } else {
            simplified_scan_bwd_launch<128, 16, DMethod>(params, stream);
        }
    #endif
}
