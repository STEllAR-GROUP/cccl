// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*! \file partition.h
 *   \brief HPX implementation of partition.
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

#include <thrust/system/detail/generic/partition.h>
#include <thrust/system/hpx/detail/contiguous_iterator.h>
#include <thrust/system/hpx/detail/execution_policy.h>
#include <thrust/system/hpx/detail/function.h>

#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/pair.h>

#include <hpx/parallel/algorithms/partition.hpp>

THRUST_NAMESPACE_BEGIN
namespace system::hpx::detail
{
template <typename ExecutionPolicy, typename ForwardIterator, typename Predicate>
ForwardIterator partition(
  execution_policy<ExecutionPolicy>& exec [[maybe_unused]], ForwardIterator first, ForwardIterator last, Predicate pred)
{
  hpx_wrapped_function<Predicate> wrapped_pred{pred};

  if constexpr (::hpx::traits::is_forward_iterator_v<ForwardIterator>)
  {
    auto res = ::hpx::partition(
      hpx::detail::to_hpx_execution_policy(exec),
      ::thrust::try_unwrap_contiguous_iterator(first),
      ::thrust::try_unwrap_contiguous_iterator(last),
      wrapped_pred);
    return detail::rewrap_contiguous_iterator(res, first);
  }
  else
  {
    return ::hpx::partition(first, last, wrapped_pred);
  }
}

template <typename ExecutionPolicy, typename ForwardIterator, typename InputIterator, typename Predicate>
ForwardIterator partition(
  execution_policy<ExecutionPolicy>& exec,
  ForwardIterator first,
  ForwardIterator last,
  InputIterator stencil,
  Predicate pred)
{
  if constexpr (::cuda::std::is_same_v<ForwardIterator, InputIterator>)
  {
    if (first == stencil)
    {
      return partition(exec, first, last, pred);
    }
  }

  return thrust::system::detail::generic::partition(exec, first, last, stencil, pred);
}

template <typename ExecutionPolicy, typename ForwardIterator, typename Predicate>
ForwardIterator stable_partition(
  execution_policy<ExecutionPolicy>& exec [[maybe_unused]], ForwardIterator first, ForwardIterator last, Predicate pred)
{
  hpx_wrapped_function<Predicate> wrapped_pred{pred};

  if constexpr (::hpx::traits::is_forward_iterator_v<ForwardIterator>)
  {
    auto res = ::hpx::stable_partition(
      hpx::detail::to_hpx_execution_policy(exec),
      ::thrust::try_unwrap_contiguous_iterator(first),
      ::thrust::try_unwrap_contiguous_iterator(last),
      wrapped_pred);
    return detail::rewrap_contiguous_iterator(res, first);
  }
  else
  {
    return ::hpx::stable_partition(first, last, wrapped_pred);
  }
}

template <typename ExecutionPolicy, typename ForwardIterator, typename InputIterator, typename Predicate>
ForwardIterator stable_partition(
  execution_policy<ExecutionPolicy>& exec,
  ForwardIterator first,
  ForwardIterator last,
  InputIterator stencil,
  Predicate pred)
{
  if constexpr (::cuda::std::is_same_v<ForwardIterator, InputIterator>)
  {
    if (first == stencil)
    {
      return stable_partition(exec, first, last, pred);
    }
  }

  return thrust::system::detail::generic::stable_partition(exec, first, last, stencil, pred);
}

template <typename ExecutionPolicy,
          typename InputIterator,
          typename OutputIterator1,
          typename OutputIterator2,
          typename Predicate>
::cuda::std::pair<OutputIterator1, OutputIterator2> partition_copy(
  execution_policy<ExecutionPolicy>& exec [[maybe_unused]],
  InputIterator first,
  InputIterator last,
  OutputIterator1 out_true,
  OutputIterator2 out_false,
  Predicate pred)
{
  hpx_wrapped_function<Predicate> wrapped_pred{pred};

  if constexpr (::hpx::traits::is_forward_iterator_v<InputIterator>
                && ::hpx::traits::is_forward_iterator_v<OutputIterator1>
                && ::hpx::traits::is_forward_iterator_v<OutputIterator2>)
  {
    auto [res_true, res_false] = ::hpx::partition_copy(
      hpx::detail::to_hpx_execution_policy(exec),
      ::thrust::try_unwrap_contiguous_iterator(first),
      ::thrust::try_unwrap_contiguous_iterator(last),
      ::thrust::try_unwrap_contiguous_iterator(out_true),
      ::thrust::try_unwrap_contiguous_iterator(out_false),
      wrapped_pred);
    return ::cuda::std::make_pair(
      detail::rewrap_contiguous_iterator(res_true, out_true), detail::rewrap_contiguous_iterator(res_false, out_false));
  }
  else
  {
    auto [res_true, res_false] = ::hpx::partition_copy(first, last, out_true, out_false, wrapped_pred);
    return ::cuda::std::make_pair(res_true, res_false);
  }
}

template <typename ExecutionPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2,
          typename Predicate>
::cuda::std::pair<OutputIterator1, OutputIterator2> partition_copy(
  execution_policy<ExecutionPolicy>& exec,
  InputIterator1 first,
  InputIterator1 last,
  InputIterator2 stencil,
  OutputIterator1 out_true,
  OutputIterator2 out_false,
  Predicate pred)
{
  if constexpr (::cuda::std::is_same_v<InputIterator1, InputIterator2>)
  {
    if (first == stencil)
    {
      return partition_copy(exec, first, last, out_true, out_false, pred);
    }
  }

  return thrust::system::detail::generic::partition_copy(exec, first, last, stencil, out_true, out_false, pred);
}

template <typename ExecutionPolicy,
          typename InputIterator,
          typename OutputIterator1,
          typename OutputIterator2,
          typename Predicate>
::cuda::std::pair<OutputIterator1, OutputIterator2> stable_partition_copy(
  execution_policy<ExecutionPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator1 out_true,
  OutputIterator2 out_false,
  Predicate pred)
{
  return partition_copy(exec, first, last, out_true, out_false, pred);
}

template <typename ExecutionPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2,
          typename Predicate>
::cuda::std::pair<OutputIterator1, OutputIterator2> stable_partition_copy(
  execution_policy<ExecutionPolicy>& exec,
  InputIterator1 first,
  InputIterator1 last,
  InputIterator2 stencil,
  OutputIterator1 out_true,
  OutputIterator2 out_false,
  Predicate pred)
{
  return partition_copy(exec, first, last, stencil, out_true, out_false, pred);
}

template <typename ExecutionPolicy, typename ForwardIterator, typename Predicate>
ForwardIterator
partition_point(execution_policy<ExecutionPolicy>& exec, ForwardIterator first, ForwardIterator last, Predicate pred)
{
  return thrust::system::detail::generic::partition_point(exec, first, last, pred);
}

template <typename ExecutionPolicy, typename InputIterator, typename Predicate>
bool is_partitioned(execution_policy<ExecutionPolicy>& exec, InputIterator first, InputIterator last, Predicate pred)
{
  return thrust::system::detail::generic::is_partitioned(exec, first, last, pred);
}
} // end namespace system::hpx::detail
THRUST_NAMESPACE_END
