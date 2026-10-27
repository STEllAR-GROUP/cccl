// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*! \file unique.h
 *   \brief HPX implementation of unique algorithms.
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
#include <thrust/system/detail/generic/unique.h>
#include <thrust/system/hpx/detail/contiguous_iterator.h>
#include <thrust/system/hpx/detail/execution_policy.h>
#include <thrust/system/hpx/detail/function.h>

#include <cuda/std/__functional/operations.h>

#include <hpx/parallel/algorithms/unique.hpp>

THRUST_NAMESPACE_BEGIN
namespace system::hpx::detail
{
template <typename ExecutionPolicy, typename ForwardIterator, typename BinaryPredicate>
ForwardIterator unique(execution_policy<ExecutionPolicy>& exec [[maybe_unused]],
                       ForwardIterator first,
                       ForwardIterator last,
                       BinaryPredicate binary_pred)
{
  hpx_wrapped_function<BinaryPredicate> wrapped_pred{binary_pred};

  if constexpr (::hpx::traits::is_forward_iterator_v<ForwardIterator>)
  {
    auto res = ::hpx::unique(
      hpx::detail::to_hpx_execution_policy(exec),
      ::thrust::try_unwrap_contiguous_iterator(first),
      ::thrust::try_unwrap_contiguous_iterator(last),
      wrapped_pred);
    return detail::rewrap_contiguous_iterator(res, first);
  }
  else
  {
    return ::hpx::unique(first, last, wrapped_pred);
  }
}

template <typename ExecutionPolicy, typename ForwardIterator>
ForwardIterator unique(execution_policy<ExecutionPolicy>& exec, ForwardIterator first, ForwardIterator last)
{
  using InputType = thrust::detail::it_value_t<ForwardIterator>;
  return unique(exec, first, last, ::cuda::std::equal_to<InputType>());
}

template <typename ExecutionPolicy, typename InputIterator, typename OutputIterator, typename BinaryPredicate>
OutputIterator unique_copy(
  execution_policy<ExecutionPolicy>& exec [[maybe_unused]],
  InputIterator first,
  InputIterator last,
  OutputIterator output,
  BinaryPredicate binary_pred)
{
  hpx_wrapped_function<BinaryPredicate> wrapped_pred{binary_pred};

  if constexpr (::hpx::traits::is_forward_iterator_v<InputIterator>
                && ::hpx::traits::is_forward_iterator_v<OutputIterator>)
  {
    auto res = ::hpx::unique_copy(
      hpx::detail::to_hpx_execution_policy(exec),
      ::thrust::try_unwrap_contiguous_iterator(first),
      ::thrust::try_unwrap_contiguous_iterator(last),
      ::thrust::try_unwrap_contiguous_iterator(output),
      wrapped_pred);
    return detail::rewrap_contiguous_iterator(res, output);
  }
  else
  {
    return ::hpx::unique_copy(first, last, output, wrapped_pred);
  }
}

template <typename ExecutionPolicy, typename InputIterator, typename OutputIterator>
OutputIterator
unique_copy(execution_policy<ExecutionPolicy>& exec, InputIterator first, InputIterator last, OutputIterator output)
{
  using value_type = thrust::detail::it_value_t<InputIterator>;
  return unique_copy(exec, first, last, output, ::cuda::std::equal_to<value_type>());
}

template <typename ExecutionPolicy, typename ForwardIterator, typename BinaryPredicate>
thrust::detail::it_difference_t<ForwardIterator> unique_count(
  execution_policy<ExecutionPolicy>& exec, ForwardIterator first, ForwardIterator last, BinaryPredicate binary_pred)
{
  return thrust::system::detail::generic::unique_count(exec, first, last, binary_pred);
}

template <typename ExecutionPolicy, typename ForwardIterator>
thrust::detail::it_difference_t<ForwardIterator>
unique_count(execution_policy<ExecutionPolicy>& exec, ForwardIterator first, ForwardIterator last)
{
  return thrust::system::detail::generic::unique_count(exec, first, last);
}
} // end namespace system::hpx::detail
THRUST_NAMESPACE_END
