// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*! \file swap_ranges.h
 *   \brief HPX implementation of swap_ranges.
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

#include <thrust/system/detail/generic/swap_ranges.h>
#include <thrust/system/hpx/detail/contiguous_iterator.h>
#include <thrust/system/hpx/detail/execution_policy.h>

#include <hpx/parallel/algorithms/swap_ranges.hpp>

THRUST_NAMESPACE_BEGIN
namespace system::hpx::detail
{
template <typename ExecutionPolicy, typename ForwardIterator1, typename ForwardIterator2>
ForwardIterator2 swap_ranges(
  execution_policy<ExecutionPolicy>& exec [[maybe_unused]],
  ForwardIterator1 first1,
  ForwardIterator1 last1,
  ForwardIterator2 first2)
{
  if constexpr (::hpx::traits::is_forward_iterator_v<ForwardIterator1>
                && ::hpx::traits::is_forward_iterator_v<ForwardIterator2>)
  {
    auto res = ::hpx::swap_ranges(
      hpx::detail::to_hpx_execution_policy(exec),
      ::thrust::try_unwrap_contiguous_iterator(first1),
      ::thrust::try_unwrap_contiguous_iterator(last1),
      ::thrust::try_unwrap_contiguous_iterator(first2));
    return detail::rewrap_contiguous_iterator(res, first2);
  }
  else
  {
    return ::hpx::swap_ranges(first1, last1, first2);
  }
}
} // end namespace system::hpx::detail
THRUST_NAMESPACE_END
