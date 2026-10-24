// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*! \file remove.h
 *   \brief HPX implementation of remove algorithms.
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

#include <thrust/system/detail/generic/remove.h>
#include <thrust/system/hpx/detail/contiguous_iterator.h>
#include <thrust/system/hpx/detail/execution_policy.h>
#include <thrust/system/hpx/detail/function.h>

#include <cuda/std/__type_traits/is_same.h>

#include <hpx/parallel/algorithms/remove.hpp>

THRUST_NAMESPACE_BEGIN
namespace system::hpx::detail
{
template <typename ExecutionPolicy, typename ForwardIterator, typename T>
ForwardIterator remove(
  execution_policy<ExecutionPolicy>& exec [[maybe_unused]], ForwardIterator first, ForwardIterator last, const T& value)
{
  if constexpr (::hpx::traits::is_forward_iterator_v<ForwardIterator>)
  {
    auto res = ::hpx::remove(
      hpx::detail::to_hpx_execution_policy(exec),
      ::thrust::try_unwrap_contiguous_iterator(first),
      ::thrust::try_unwrap_contiguous_iterator(last),
      value);
    return detail::rewrap_contiguous_iterator(res, first);
  }
  else
  {
    return ::hpx::remove(first, last, value);
  }
}

template <typename ExecutionPolicy, typename ForwardIterator, typename Predicate>
ForwardIterator remove_if(
  execution_policy<ExecutionPolicy>& exec [[maybe_unused]], ForwardIterator first, ForwardIterator last, Predicate pred)
{
  hpx_wrapped_function<Predicate> wrapped_pred{pred};

  if constexpr (::hpx::traits::is_forward_iterator_v<ForwardIterator>)
  {
    auto res = ::hpx::remove_if(
      hpx::detail::to_hpx_execution_policy(exec),
      ::thrust::try_unwrap_contiguous_iterator(first),
      ::thrust::try_unwrap_contiguous_iterator(last),
      wrapped_pred);
    return detail::rewrap_contiguous_iterator(res, first);
  }
  else
  {
    return ::hpx::remove_if(first, last, wrapped_pred);
  }
}

template <typename ExecutionPolicy, typename ForwardIterator, typename InputIterator, typename Predicate>
ForwardIterator remove_if(
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
      return remove_if(exec, first, last, pred);
    }
  }

  return thrust::system::detail::generic::remove_if(exec, first, last, stencil, pred);
}

template <typename ExecutionPolicy, typename InputIterator, typename OutputIterator, typename T>
OutputIterator remove_copy(
  execution_policy<ExecutionPolicy>& exec [[maybe_unused]],
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  const T& value)
{
  if constexpr (::hpx::traits::is_forward_iterator_v<InputIterator>
                && ::hpx::traits::is_forward_iterator_v<OutputIterator>)
  {
    auto res = ::hpx::remove_copy(
      hpx::detail::to_hpx_execution_policy(exec),
      ::thrust::try_unwrap_contiguous_iterator(first),
      ::thrust::try_unwrap_contiguous_iterator(last),
      ::thrust::try_unwrap_contiguous_iterator(result),
      value);
    return detail::rewrap_contiguous_iterator(res, result);
  }
  else
  {
    return ::hpx::remove_copy(first, last, result, value);
  }
}

template <typename ExecutionPolicy, typename InputIterator, typename OutputIterator, typename Predicate>
OutputIterator remove_copy_if(
  execution_policy<ExecutionPolicy>& exec [[maybe_unused]],
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  Predicate pred)
{
  hpx_wrapped_function<Predicate> wrapped_pred{pred};

  if constexpr (::hpx::traits::is_forward_iterator_v<InputIterator>
                && ::hpx::traits::is_forward_iterator_v<OutputIterator>)
  {
    auto res = ::hpx::remove_copy_if(
      hpx::detail::to_hpx_execution_policy(exec),
      ::thrust::try_unwrap_contiguous_iterator(first),
      ::thrust::try_unwrap_contiguous_iterator(last),
      ::thrust::try_unwrap_contiguous_iterator(result),
      wrapped_pred);
    return detail::rewrap_contiguous_iterator(res, result);
  }
  else
  {
    return ::hpx::remove_copy_if(first, last, result, wrapped_pred);
  }
}

template <typename ExecutionPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename Predicate>
OutputIterator remove_copy_if(
  execution_policy<ExecutionPolicy>& exec,
  InputIterator1 first,
  InputIterator1 last,
  InputIterator2 stencil,
  OutputIterator result,
  Predicate pred)
{
  if constexpr (::cuda::std::is_same_v<InputIterator1, InputIterator2>)
  {
    if (first == stencil)
    {
      return remove_copy_if(exec, first, last, result, pred);
    }
  }

  return thrust::system::detail::generic::remove_copy_if(exec, first, last, stencil, result, pred);
}
} // end namespace system::hpx::detail
THRUST_NAMESPACE_END
