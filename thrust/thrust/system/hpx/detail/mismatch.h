// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*! \file mismatch.h
 *   \brief HPX implementation of mismatch algorithms.
 */

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/mismatch.h>
#include <thrust/system/hpx/detail/contiguous_iterator.h>
#include <thrust/system/hpx/detail/execution_policy.h>
#include <thrust/system/hpx/detail/function.h>

#include <cuda/std/__functional/operations.h>
#include <cuda/std/__utility/pair.h>

#include <hpx/parallel/algorithms/mismatch.hpp>

THRUST_NAMESPACE_BEGIN
namespace system::hpx::detail
{
template <typename ExecutionPolicy, typename InputIterator1, typename InputIterator2, typename BinaryPredicate>
::cuda::std::pair<InputIterator1, InputIterator2> mismatch(
  execution_policy<ExecutionPolicy>& exec [[maybe_unused]],
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  BinaryPredicate pred)
{
  hpx_wrapped_function<BinaryPredicate> wrapped_pred{pred};

  if constexpr (::hpx::traits::is_forward_iterator_v<InputIterator1>
                && ::hpx::traits::is_forward_iterator_v<InputIterator2>)
  {
    auto [res1, res2] = ::hpx::mismatch(
      hpx::detail::to_hpx_execution_policy(exec),
      ::thrust::try_unwrap_contiguous_iterator(first1),
      ::thrust::try_unwrap_contiguous_iterator(last1),
      ::thrust::try_unwrap_contiguous_iterator(first2),
      wrapped_pred);
    return ::cuda::std::make_pair(
      detail::rewrap_contiguous_iterator(res1, first1), detail::rewrap_contiguous_iterator(res2, first2));
  }
  else
  {
    auto [res1, res2] = ::hpx::mismatch(first1, last1, first2, wrapped_pred);
    return ::cuda::std::make_pair(res1, res2);
  }
}

template <typename ExecutionPolicy, typename InputIterator1, typename InputIterator2>
::cuda::std::pair<InputIterator1, InputIterator2>
mismatch(execution_policy<ExecutionPolicy>& exec, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2)
{
  using value_type = thrust::detail::it_value_t<InputIterator1>;
  return mismatch(exec, first1, last1, first2, ::cuda::std::equal_to<value_type>());
}
} // end namespace system::hpx::detail
THRUST_NAMESPACE_END
