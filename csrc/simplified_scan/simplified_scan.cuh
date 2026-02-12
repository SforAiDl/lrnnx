/******************************************************************************
 * Copyright (c) 2026, LRNNx Team.
 * Simplified Scan
 *
 * This file provides forward pass support for complex-valued inputs (u).
 *
 * This kernel is an adaption of the selective scan kernel for S5, the main change which allows
 * the SISO kernel be to be used in this way is that we perform the input projection before the scan
 * and then the output projection after the scan. The projected input makes this into a SISO system.
 * The other difference is that we allow complex inputs and outputs.
 ******************************************************************************/

#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/cuda/CUDAException.h>

#ifndef USE_ROCM
    #include <cub/block/block_load.cuh>
    #include <cub/block/block_store.cuh>
    #include <cub/block/block_scan.cuh>
#else
    #include <hipcub/hipcub.hpp>
    namespace cub = hipcub;
#endif

#include "simplified_scan.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
// Kernel traits for complex input
// When input is complex, state and output are always complex (float4 scan_t)
////////////////////////////////////////////////////////////////////////////////////////////////////

template<int kNThreads_, int kNItems_, int kNRows_, bool kIsEvenLen_, DiscretizationMethod DMethod_>
struct Simplified_Scan_fwd_kernel_traits {
    static_assert(kNItems_ % 4 == 0);
    using input_t = float;
    using weight_t = complex_t;
    static constexpr int kNThreads = kNThreads_;
    static constexpr int kMinBlocks = kNThreads < 128 ? 5 : 3;
    static constexpr int kNItems = kNItems_;
    static constexpr int kNRows = kNRows_;
    static constexpr int kNBytes = sizeof(float);
    static_assert(kNBytes == 4);
    static constexpr int kNElts = 4;
    static_assert(kNItems % kNElts == 0);
    static constexpr int kNLoads = kNItems / kNElts;
    static constexpr bool kIsEvenLen = kIsEvenLen_;
    static constexpr DiscretizationMethod DMethod = DMethod_;

    static constexpr bool kDirectIO = kIsEvenLen && kNLoads == 1;

    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    using scan_t = float4;
    using BlockLoadT = cub::BlockLoad<float, kNThreads, kNItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadVecT = cub::BlockLoad<vec_t, kNThreads, kNLoads,
        !kDirectIO ? cub::BLOCK_LOAD_WARP_TRANSPOSE : cub::BLOCK_LOAD_DIRECT>;
    using BlockLoadComplexT = cub::BlockLoad<float, kNThreads, kNItems * 2, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadComplexVecT = cub::BlockLoad<vec_t, kNThreads, kNLoads * 2,
        !kDirectIO ? cub::BLOCK_LOAD_WARP_TRANSPOSE : cub::BLOCK_LOAD_DIRECT>;
    using BlockLoadWeightT = cub::BlockLoad<float, kNThreads, kNItems * 2, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadWeightVecT = cub::BlockLoad<vec_t, kNThreads, kNLoads * 2,
        !kDirectIO ? cub::BLOCK_LOAD_WARP_TRANSPOSE : cub::BLOCK_LOAD_DIRECT>;
    using BlockStoreT = cub::BlockStore<float, kNThreads, kNItems, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockStoreVecT = cub::BlockStore<vec_t, kNThreads, kNLoads,
        !kDirectIO ? cub::BLOCK_STORE_WARP_TRANSPOSE : cub::BLOCK_STORE_DIRECT>;
    using BlockStoreComplexT = cub::BlockStore<float, kNThreads, kNItems * 2, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockStoreComplexVecT = cub::BlockStore<vec_t, kNThreads, kNLoads * 2,
        !kDirectIO ? cub::BLOCK_STORE_WARP_TRANSPOSE : cub::BLOCK_STORE_DIRECT>;
    using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_WARP_SCANS>;
    static constexpr int kSmemIOSize = custom_max({
        sizeof(typename BlockLoadT::TempStorage),
        sizeof(typename BlockLoadVecT::TempStorage),
        sizeof(typename BlockLoadComplexT::TempStorage),
        sizeof(typename BlockLoadComplexVecT::TempStorage),
        sizeof(typename BlockStoreT::TempStorage),
        sizeof(typename BlockStoreVecT::TempStorage),
        sizeof(typename BlockStoreComplexT::TempStorage),
        sizeof(typename BlockStoreComplexVecT::TempStorage)
    });
    static constexpr int kSmemSize = kSmemIOSize + sizeof(typename BlockScanT::TempStorage);
};

template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads, Ktraits::kMinBlocks)
void simplified_scan_fwd_kernel(SSMParamsBase params) {
    constexpr DiscretizationMethod DMethod = Ktraits::DMethod;
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kNItems = Ktraits::kNItems;
    constexpr int kNRows = Ktraits::kNRows;
    constexpr bool kDirectIO = Ktraits::kDirectIO;
    using scan_t = float4;

    // Shared memory.
    extern __shared__ char smem_[];
    auto& smem_load = reinterpret_cast<typename Ktraits::BlockLoadT::TempStorage&>(smem_);
    auto& smem_load_complex = reinterpret_cast<typename Ktraits::BlockLoadComplexT::TempStorage&>(smem_);
    auto& smem_load_weight = reinterpret_cast<typename Ktraits::BlockLoadWeightT::TempStorage&>(smem_);
    auto& smem_load_weight1 = *reinterpret_cast<typename Ktraits::BlockLoadWeightT::TempStorage*>(
        smem_ + sizeof(typename Ktraits::BlockLoadWeightT::TempStorage));
    auto& smem_store = reinterpret_cast<typename Ktraits::BlockStoreT::TempStorage&>(smem_);
    auto& smem_store_complex = reinterpret_cast<typename Ktraits::BlockStoreComplexT::TempStorage&>(smem_);
    auto& smem_scan = *reinterpret_cast<typename Ktraits::BlockScanT::TempStorage*>(smem_ + Ktraits::kSmemIOSize);
    scan_t *smem_running_prefix = reinterpret_cast<scan_t *>(smem_ + Ktraits::kSmemSize);

    const int batch_id = blockIdx.x;
    const int dim_id = blockIdx.y;
    float *u = reinterpret_cast<float *>(params.u_ptr) + batch_id * params.u_batch_stride * 2
        + dim_id * kNRows * params.u_d_stride * 2;
    float *delta = reinterpret_cast<float *>(params.delta_ptr) + batch_id * params.delta_batch_stride
        + dim_id * kNRows * params.delta_d_stride;
    float *deltaA = params.deltaA_ptr != nullptr 
        ? reinterpret_cast<float *>(params.deltaA_ptr) + batch_id * params.deltaA_batch_stride
            + dim_id * kNRows * params.deltaA_d_stride 
        : nullptr;
    complex_t *A = reinterpret_cast<complex_t *>(params.A_ptr) + dim_id * kNRows * params.A_d_stride;
    complex_t *B = reinterpret_cast<complex_t *>(params.B_ptr) + dim_id * kNRows * params.B_d_stride;
    complex_t *C = reinterpret_cast<complex_t *>(params.C_ptr) + dim_id * kNRows * params.C_d_stride;
    scan_t *x = reinterpret_cast<scan_t *>(params.x_ptr) + (batch_id * params.dim + dim_id * kNRows) * params.n_chunks * params.dstate;

    constexpr int kChunkSize = kNThreads * kNItems;
    for (int chunk = 0; chunk < params.n_chunks; ++chunk) {
        complex_t u_vals[kNRows][kNItems];
        float delta_vals_load[kNRows][kNItems], deltaA_vals_load[kNRows][kNItems];
        __syncthreads();
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            if constexpr (!kDirectIO) {
                if (r > 0) { __syncthreads(); }
            }
            load_complex_input<Ktraits>(u + r * params.u_d_stride * 2, u_vals[r], smem_load_complex, params.seqlen - chunk * kChunkSize);
            if constexpr (!kDirectIO) { __syncthreads(); }
            load_input<Ktraits>(delta + r * params.delta_d_stride, delta_vals_load[r], smem_load, params.seqlen - chunk * kChunkSize);
            // Load deltaA if provided
            if (deltaA != nullptr) {
                if constexpr (!kDirectIO) { __syncthreads(); }
                load_input<Ktraits>(deltaA + r * params.deltaA_d_stride, deltaA_vals_load[r], smem_load, params.seqlen - chunk * kChunkSize);
            }
        }
        u += kChunkSize * 2;  // Complex: advance by 2 floats per element
        delta += kChunkSize;
        if (deltaA != nullptr) {
            deltaA += kChunkSize;
        }

        float delta_vals[kNRows][kNItems], deltaA_vals[kNRows][kNItems];
        complex_t delta_u_vals[kNRows][kNItems];
        complex_t out_vals[kNRows][kNItems];
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                complex_t u_val = u_vals[r][i];
                delta_vals[r][i] = float(delta_vals_load[r][i]);
                delta_u_vals[r][i] = delta_vals[r][i] * u_val;
                out_vals[r][i] = complex_t(0.0f, 0.0f);
                // Load deltaA values if provided
                if (deltaA != nullptr) {
                    deltaA_vals[r][i] = float(deltaA_vals_load[r][i]);
                }
            }
        }

        __syncthreads();
        for (int state_idx = 0; state_idx < params.dstate; ++state_idx) {
            complex_t A_val[kNRows];
            #pragma unroll
            for (int r = 0; r < kNRows; ++r) {
                A_val[r] = A[state_idx * params.A_dstate_stride + r * params.A_d_stride];
            }

            complex_t BC_val[kNRows];
            #pragma unroll
            for (int r = 0; r < kNRows; ++r) {
                BC_val[r] = B[state_idx * params.B_dstate_stride + r * params.B_d_stride] *
                            C[state_idx * params.C_dstate_stride + r * params.C_d_stride];
            }

            #pragma unroll
            for (int r = 0; r < kNRows; ++r) {
                if (r > 0) { __syncthreads(); }
                scan_t thread_data[kNItems];
                #pragma unroll
                for (int i = 0; i < kNItems; ++i) {
                    // Use deltaA for A discretization if provided, otherwise use delta
                    const float delta_for_A = (deltaA != nullptr) ? deltaA_vals[r][i] : delta_vals[r][i];

                    if constexpr (DMethod == DiscretizationMethod::ZOH) {
                        // A_bar = exp(A * delta_for_A)
                        // B_bar = A^-1 * (A_bar - I) * B
                        const complex_t A_del_for_A = A_val[r] * delta_for_A;
                        const complex_t A_bar = cexpf(A_del_for_A);
                        const complex_t one = complex_t(1.0f, 0.0f);
                        const complex_t A_del_for_B = A_val[r] * delta_vals[r][i];
                        const complex_t A_disc_for_B = cexpf(A_del_for_B);
                        const complex_t B_tilde = (A_disc_for_B - one) / A_val[r];
                        const complex_t B_u_term = one * B_tilde * u_vals[r][i];
                        thread_data[i] = make_float4(A_bar.real_, A_bar.imag_, B_u_term.real_, B_u_term.imag_);
                    } else if constexpr (DMethod == DiscretizationMethod::BILINEAR) {
                        // A_bar = (I + 0.5 * A * delta_for_A) / (I - 0.5 * A * delta_for_A)
                        // B_bar = B * delta / (I - 0.5 * A * delta)
                        const complex_t v_for_A = 0.5f * delta_for_A * A_val[r];
                        const complex_t one = complex_t(1.0f, 0.0f);
                        const complex_t den_inv_for_A = one / (one - v_for_A);
                        const complex_t A_bar = (one + v_for_A) * den_inv_for_A;
                        const complex_t v_for_B = 0.5f * delta_vals[r][i] * A_val[r];
                        const complex_t den_inv_for_B = one / (one - v_for_B);
                        const complex_t B_u_term = one * den_inv_for_B * delta_u_vals[r][i];
                        thread_data[i] = make_float4(A_bar.real_, A_bar.imag_, B_u_term.real_, B_u_term.imag_);
                    } else if constexpr (DMethod == DiscretizationMethod::DIRAC) {
                        // A_bar = exp(A * delta_for_A)
                        // B_bar = B
                        const complex_t A_del = A_val[r] * delta_for_A;
                        const complex_t A_bar = cexpf(A_del);
                        const complex_t one = complex_t(1.0f, 0.0f);
                        const complex_t B_u_term = one * u_vals[r][i];
                        thread_data[i] = make_float4(A_bar.real_, A_bar.imag_, B_u_term.real_, B_u_term.imag_);
                    }
                    if constexpr (!Ktraits::kIsEvenLen) {
                        if (threadIdx.x * kNItems + i >= params.seqlen - chunk * kChunkSize) {
                            thread_data[i] = make_float4(1.f, 0.f, 0.f, 0.f);
                        }
                    }
                }

                // Initialize running prefix
                scan_t running_prefix;
                running_prefix = chunk > 0 && threadIdx.x % 32 == 0 ?
                    smem_running_prefix[state_idx + r * MAX_DSTATE] : make_float4(1.f, 0.f, 0.f, 0.f);

                SSMScanPrefixCallbackOp<complex_t> prefix_op(running_prefix);
                typename Ktraits::BlockScanT(smem_scan).InclusiveScan(
                    thread_data, thread_data, SSMScanOp<complex_t>(), prefix_op
                );

                if (threadIdx.x == 0) {
                    smem_running_prefix[state_idx] = prefix_op.running_prefix;
                    x[(r * params.n_chunks + chunk) * params.dstate + state_idx] = prefix_op.running_prefix;
                }

                #pragma unroll
                for (int i = 0; i < kNItems; ++i) {
                    const complex_t x_state = complex_t(thread_data[i].z, thread_data[i].w);
                    out_vals[r][i] = out_vals[r][i] + x_state * BC_val[r];
                }
            }
        }

        float *out = reinterpret_cast<float *>(params.out_ptr) + batch_id * params.out_batch_stride * 2
            + dim_id * kNRows * params.out_d_stride * 2 + chunk * kChunkSize * 2;
        __syncthreads();
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            if constexpr (!kDirectIO) {
                if (r > 0) { __syncthreads(); }
            }
            store_complex_output<Ktraits>(out + r * params.out_d_stride * 2, out_vals[r], smem_store_complex, params.seqlen - chunk * kChunkSize);
        }

    }
}

template<int kNThreads, int kNItems, DiscretizationMethod DMethod>
void simplified_scan_fwd_launch(SSMParamsBase &params, cudaStream_t stream) {
    constexpr int kNRows = 1;
    BOOL_SWITCH(params.seqlen % (kNThreads * kNItems) == 0, kIsEvenLen, [&] {
        using Ktraits = Simplified_Scan_fwd_kernel_traits<kNThreads, kNItems, kNRows, kIsEvenLen, DMethod>;
        constexpr int kSmemSize = Ktraits::kSmemSize + kNRows * MAX_DSTATE * sizeof(typename Ktraits::scan_t);
        dim3 grid(params.batch, params.dim / kNRows);

        auto kernel = &simplified_scan_fwd_kernel<Ktraits>;

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


template<DiscretizationMethod DMethod>
void simplified_scan_fwd_cuda(SSMParamsBase &params, cudaStream_t stream) {
    #ifndef USE_ROCM
        if (params.seqlen <= 128) {
            simplified_scan_fwd_launch<32, 4, DMethod>(params, stream);
        } else if (params.seqlen <= 256) {
            simplified_scan_fwd_launch<32, 8, DMethod>(params, stream);
        } else if (params.seqlen <= 512) {
            simplified_scan_fwd_launch<32, 16, DMethod>(params, stream);
        } else if (params.seqlen <= 1024) {
            simplified_scan_fwd_launch<64, 16, DMethod>(params, stream);
        } else {
            simplified_scan_fwd_launch<128, 16, DMethod>(params, stream);
        }
    #else
        if (params.seqlen <= 256) {
            simplified_scan_fwd_launch<64, 4, DMethod>(params, stream);
        } else if (params.seqlen <= 512) {
            simplified_scan_fwd_launch<64, 8, DMethod>(params, stream);
        } else if (params.seqlen <= 1024) {
            simplified_scan_fwd_launch<64, 16, DMethod>(params, stream);
        } else {
            simplified_scan_fwd_launch<128, 16, DMethod>(params, stream);
        }
    #endif
}
