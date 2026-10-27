// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*! \file replace.h
 *   \brief HPX implementation of replace algorithms.
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

#include <thrust/system/detail/generic/replace.h>
#include <thrust/system/hpx/detail/contiguous_iterator.h>
#include <thrust/system/hpx/detail/execution_policy.h>
#include <thrust/system/hpx/detail/function.h>

#include <cuda/std/__type_traits/is_same.h>

#include <hpx/parallel/algorithms/replace.hpp>

THRUST_NAMESPACE_BEGIN
namespace system::hpx::detail
{
template <typename ExecutionPolicy, typename ForwardIterator, typename T>
void replace(execution_policy<ExecutionPolicy>& exec [[maybe_unused]],
             ForwardIterator first,
             ForwardIterator last,
             const T& old_value,
             const T& new_value)
{
  if constexpr (::hpx::traits::is_forward_iterator_v<ForwardIterator>)
  {
    ::hpx::replace(hpx::detail::to_hpx_execution_policy(exec),
                   ::thrust::try_unwrap_contiguous_iterator(first),
                   ::thrust::try_unwrap_contiguous_iterator(last),
                   old_value,
                   new_value);
  }
  else
  {
    ::hpx::replace(first, last, old_value, new_value);
  }
}

template <typename ExecutionPolicy, typename ForwardIterator, typename Predicate, typename T>
void replace_if(execution_policy<ExecutionPolicy>& exec [[maybe_unused]],
                ForwardIterator first,
                ForwardIterator last,
                Predicate pred,
                const T& new_value)
{
  hpx_wrapped_function<Predicate> wrapped_pred{pred};

  if constexpr (::hpx::traits::is_forward_iterator_v<ForwardIterator>)
  {
    ::hpx::replace_if(
      hpx::detail::to_hpx_execution_policy(exec),
      ::thrust::try_unwrap_contiguous_iterator(first),
      ::thrust::try_unwrap_contiguous_iterator(last),
      wrapped_pred,
      new_value);
  }
  else
  {
    ::hpx::replace_if(first, last, wrapped_pred, new_value);
  }
}

template <typename ExecutionPolicy, typename ForwardIterator, typename InputIterator, typename Predicate, typename T>
void replace_if(execution_policy<ExecutionPolicy>& exec,
                ForwardIterator first,
                ForwardIterator last,
                InputIterator stencil,
                Predicate pred,
                const T& new_value)
{
  if constexpr (::cuda::std::is_same_v<ForwardIterator, InputIterator>)
  {
    if (first == stencil)
    {
      return replace_if(exec, first, last, pred, new_value);
    }
  }

  return thrust::system::detail::generic::replace_if(exec, first, last, stencil, pred, new_value);
}

template <typename ExecutionPolicy, typename InputIterator, typename OutputIterator, typename T>
OutputIterator replace_copy(
  execution_policy<ExecutionPolicy>& exec [[maybe_unused]],
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  const T& old_value,
  const T& new_value)
{
  if constexpr (::hpx::traits::is_forward_iterator_v<InputIterator>
                && ::hpx::traits::is_forward_iterator_v<OutputIterator>)
  {
    auto res = ::hpx::replace_copy(
      hpx::detail::to_hpx_execution_policy(exec),
      ::thrust::try_unwrap_contiguous_iterator(first),
      ::thrust::try_unwrap_contiguous_iterator(last),
      ::thrust::try_unwrap_contiguous_iterator(result),
      old_value,
      new_value);
    return detail::rewrap_contiguous_iterator(res, result);
  }
  else
  {
    return ::hpx::replace_copy(first, last, result, old_value, new_value);
  }
}

template <typename ExecutionPolicy, typename InputIterator, typename OutputIterator, typename Predicate, typename T>
OutputIterator replace_copy_if(
  execution_policy<ExecutionPolicy>& exec [[maybe_unused]],
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  Predicate pred,
  const T& new_value)
{
  hpx_wrapped_function<Predicate> wrapped_pred{pred};

  if constexpr (::hpx::traits::is_forward_iterator_v<InputIterator>
                && ::hpx::traits::is_forward_iterator_v<OutputIterator>)
  {
    auto res = ::hpx::replace_copy_if(
      hpx::detail::to_hpx_execution_policy(exec),
      ::thrust::try_unwrap_contiguous_iterator(first),
      ::thrust::try_unwrap_contiguous_iterator(last),
      ::thrust::try_unwrap_contiguous_iterator(result),
      wrapped_pred,
      new_value);
    return detail::rewrap_contiguous_iterator(res, result);
  }
  else
  {
    return ::hpx::replace_copy_if(first, last, result, wrapped_pred, new_value);
  }
}

template <typename ExecutionPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename Predicate,
          typename T>
OutputIterator replace_copy_if(
  execution_policy<ExecutionPolicy>& exec,
  InputIterator1 first,
  InputIterator1 last,
  InputIterator2 stencil,
  OutputIterator result,
  Predicate pred,
  const T& new_value)
{
  if constexpr (::cuda::std::is_same_v<InputIterator1, InputIterator2>)
  {
    if (first == stencil)
    {
      return replace_copy_if(exec, first, last, result, pred, new_value);
    }
  }

  return thrust::system::detail::generic::replace_copy_if(exec, first, last, stencil, result, pred, new_value);
}
} // end namespace system::hpx::detail
THRUST_NAMESPACE_END
