// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*! \file reverse.h
 *   \brief HPX implementation of reverse algorithms.
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

#include <thrust/system/detail/generic/reverse.h>
#include <thrust/system/hpx/detail/contiguous_iterator.h>
#include <thrust/system/hpx/detail/execution_policy.h>

#include <hpx/parallel/algorithms/reverse.hpp>

THRUST_NAMESPACE_BEGIN
namespace system::hpx::detail
{
template <typename ExecutionPolicy, typename BidirectionalIterator>
void reverse(execution_policy<ExecutionPolicy>& exec [[maybe_unused]],
             BidirectionalIterator first,
             BidirectionalIterator last)
{
  if constexpr (::hpx::traits::is_forward_iterator_v<BidirectionalIterator>)
  {
    ::hpx::reverse(hpx::detail::to_hpx_execution_policy(exec),
                   ::thrust::try_unwrap_contiguous_iterator(first),
                   ::thrust::try_unwrap_contiguous_iterator(last));
  }
  else
  {
    ::hpx::reverse(first, last);
  }
}

template <typename ExecutionPolicy, typename BidirectionalIterator, typename OutputIterator>
OutputIterator reverse_copy(execution_policy<ExecutionPolicy>& exec [[maybe_unused]],
                            BidirectionalIterator first,
                            BidirectionalIterator last,
                            OutputIterator result)
{
  if constexpr (::hpx::traits::is_forward_iterator_v<BidirectionalIterator>
                && ::hpx::traits::is_forward_iterator_v<OutputIterator>)
  {
    auto res = ::hpx::reverse_copy(
      hpx::detail::to_hpx_execution_policy(exec),
      ::thrust::try_unwrap_contiguous_iterator(first),
      ::thrust::try_unwrap_contiguous_iterator(last),
      ::thrust::try_unwrap_contiguous_iterator(result));
    return detail::rewrap_contiguous_iterator(res, result);
  }
  else
  {
    return ::hpx::reverse_copy(first, last, result);
  }
}
} // end namespace system::hpx::detail
THRUST_NAMESPACE_END
