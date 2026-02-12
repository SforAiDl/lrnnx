#pragma once
// This file helps for making copies in a safer, more efficient way
#ifndef USE_ROCM
    #include <cub/config.cuh>

    #include <cuda/std/type_traits>
#else
    #include <hipcub/hipcub.hpp>
    // Map ::cuda::std to the standard std namespace
    namespace cuda {
        namespace std = ::std;
    }
#endif


namespace detail
{

#if defined(_NVHPC_CUDA)
template <typename T, typename U>
__host__ __device__ void uninitialized_copy(T *ptr, U &&val)
{
  // NVBug 3384810
  new (ptr) T(::cuda::std::forward<U>(val));
}
#else
template <typename T,
          typename U,
          typename ::cuda::std::enable_if<
            ::cuda::std::is_trivially_copyable<T>::value,
            int
          >::type = 0>
__host__ __device__ void uninitialized_copy(T *ptr, U &&val)
{
  *ptr = ::cuda::std::forward<U>(val);
}

template <typename T,
         typename U,
         typename ::cuda::std::enable_if<
           !::cuda::std::is_trivially_copyable<T>::value,
           int
         >::type = 0>
__host__ __device__ void uninitialized_copy(T *ptr, U &&val)
{
  new (ptr) T(::cuda::std::forward<U>(val));
}
#endif

} 
