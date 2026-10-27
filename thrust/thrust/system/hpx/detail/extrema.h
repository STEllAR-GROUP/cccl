// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*! \file extrema.h
 *   \brief HPX implementation of extrema algorithms.
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
#include <thrust/system/detail/generic/extrema.h>
#include <thrust/system/hpx/detail/contiguous_iterator.h>
#include <thrust/system/hpx/detail/execution_policy.h>
#include <thrust/system/hpx/detail/function.h>

#include <cuda/std/__functional/operations.h>
#include <cuda/std/__utility/pair.h>

#include <hpx/parallel/algorithms/minmax.hpp>

THRUST_NAMESPACE_BEGIN
namespace system::hpx::detail
{
template <typename ExecutionPolicy, typename ForwardIterator, typename BinaryPredicate>
ForwardIterator min_element(execution_policy<ExecutionPolicy>& exec [[maybe_unused]],
                            ForwardIterator first,
                            ForwardIterator last,
                            BinaryPredicate comp)
{
  if (first == last)
  {
    return last;
  }

  hpx_wrapped_function<BinaryPredicate> wrapped_comp{comp};

  if constexpr (::hpx::traits::is_forward_iterator_v<ForwardIterator>)
  {
    auto res = ::hpx::min_element(
      hpx::detail::to_hpx_execution_policy(exec),
      ::thrust::try_unwrap_contiguous_iterator(first),
      ::thrust::try_unwrap_contiguous_iterator(last),
      wrapped_comp);
    return detail::rewrap_contiguous_iterator(res, first);
  }
  else
  {
    return ::hpx::min_element(first, last, wrapped_comp);
  }
}

template <typename ExecutionPolicy, typename ForwardIterator>
ForwardIterator min_element(execution_policy<ExecutionPolicy>& exec, ForwardIterator first, ForwardIterator last)
{
  using value_type = thrust::detail::it_value_t<ForwardIterator>;
  return min_element(exec, first, last, ::cuda::std::less<value_type>());
}

template <typename ExecutionPolicy, typename ForwardIterator, typename BinaryPredicate>
ForwardIterator max_element(execution_policy<ExecutionPolicy>& exec [[maybe_unused]],
                            ForwardIterator first,
                            ForwardIterator last,
                            BinaryPredicate comp)
{
  if (first == last)
  {
    return last;
  }

  hpx_wrapped_function<BinaryPredicate> wrapped_comp{comp};

  if constexpr (::hpx::traits::is_forward_iterator_v<ForwardIterator>)
  {
    auto res = ::hpx::max_element(
      hpx::detail::to_hpx_execution_policy(exec),
      ::thrust::try_unwrap_contiguous_iterator(first),
      ::thrust::try_unwrap_contiguous_iterator(last),
      wrapped_comp);
    return detail::rewrap_contiguous_iterator(res, first);
  }
  else
  {
    return ::hpx::max_element(first, last, wrapped_comp);
  }
}

template <typename ExecutionPolicy, typename ForwardIterator>
ForwardIterator max_element(execution_policy<ExecutionPolicy>& exec, ForwardIterator first, ForwardIterator last)
{
  using value_type = thrust::detail::it_value_t<ForwardIterator>;
  return max_element(exec, first, last, ::cuda::std::less<value_type>());
}

template <typename ExecutionPolicy, typename ForwardIterator, typename BinaryPredicate>
::cuda::std::pair<ForwardIterator, ForwardIterator> minmax_element(
  execution_policy<ExecutionPolicy>& exec [[maybe_unused]],
  ForwardIterator first,
  ForwardIterator last,
  BinaryPredicate comp)
{
  if (first == last)
  {
    return ::cuda::std::make_pair(last, last);
  }

  hpx_wrapped_function<BinaryPredicate> wrapped_comp{comp};

  if constexpr (::hpx::traits::is_forward_iterator_v<ForwardIterator>)
  {
    auto [res_min, res_max] = ::hpx::minmax_element(
      hpx::detail::to_hpx_execution_policy(exec),
      ::thrust::try_unwrap_contiguous_iterator(first),
      ::thrust::try_unwrap_contiguous_iterator(last),
      wrapped_comp);
    return ::cuda::std::make_pair(
      detail::rewrap_contiguous_iterator(res_min, first), detail::rewrap_contiguous_iterator(res_max, first));
  }
  else
  {
    auto [res_min, res_max] = ::hpx::minmax_element(first, last, wrapped_comp);
    return ::cuda::std::make_pair(res_min, res_max);
  }
}

template <typename ExecutionPolicy, typename ForwardIterator>
::cuda::std::pair<ForwardIterator, ForwardIterator>
minmax_element(execution_policy<ExecutionPolicy>& exec, ForwardIterator first, ForwardIterator last)
{
  using value_type = thrust::detail::it_value_t<ForwardIterator>;
  return minmax_element(exec, first, last, ::cuda::std::less<value_type>());
}
} // end namespace system::hpx::detail
THRUST_NAMESPACE_END
