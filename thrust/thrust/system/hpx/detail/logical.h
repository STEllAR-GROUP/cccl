// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*! \file logical.h
 *   \brief HPX implementation of logical algorithms.
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

#include <thrust/system/detail/generic/logical.h>
#include <thrust/system/hpx/detail/contiguous_iterator.h>
#include <thrust/system/hpx/detail/execution_policy.h>
#include <thrust/system/hpx/detail/function.h>

#include <hpx/parallel/algorithms/all_any_none.hpp>

THRUST_NAMESPACE_BEGIN
namespace system::hpx::detail
{
template <typename ExecutionPolicy, typename InputIterator, typename Predicate>
bool all_of(
  execution_policy<ExecutionPolicy>& exec [[maybe_unused]], InputIterator first, InputIterator last, Predicate pred)
{
  hpx_wrapped_function<Predicate> wrapped_pred{pred};

  if constexpr (::hpx::traits::is_forward_iterator_v<InputIterator>)
  {
    return ::hpx::all_of(
      hpx::detail::to_hpx_execution_policy(exec),
      ::thrust::try_unwrap_contiguous_iterator(first),
      ::thrust::try_unwrap_contiguous_iterator(last),
      wrapped_pred);
  }
  else
  {
    return ::hpx::all_of(first, last, wrapped_pred);
  }
}

template <typename ExecutionPolicy, typename InputIterator, typename Predicate>
bool any_of(
  execution_policy<ExecutionPolicy>& exec [[maybe_unused]], InputIterator first, InputIterator last, Predicate pred)
{
  hpx_wrapped_function<Predicate> wrapped_pred{pred};

  if constexpr (::hpx::traits::is_forward_iterator_v<InputIterator>)
  {
    return ::hpx::any_of(
      hpx::detail::to_hpx_execution_policy(exec),
      ::thrust::try_unwrap_contiguous_iterator(first),
      ::thrust::try_unwrap_contiguous_iterator(last),
      wrapped_pred);
  }
  else
  {
    return ::hpx::any_of(first, last, wrapped_pred);
  }
}

template <typename ExecutionPolicy, typename InputIterator, typename Predicate>
bool none_of(
  execution_policy<ExecutionPolicy>& exec [[maybe_unused]], InputIterator first, InputIterator last, Predicate pred)
{
  hpx_wrapped_function<Predicate> wrapped_pred{pred};

  if constexpr (::hpx::traits::is_forward_iterator_v<InputIterator>)
  {
    return ::hpx::none_of(
      hpx::detail::to_hpx_execution_policy(exec),
      ::thrust::try_unwrap_contiguous_iterator(first),
      ::thrust::try_unwrap_contiguous_iterator(last),
      wrapped_pred);
  }
  else
  {
    return ::hpx::none_of(first, last, wrapped_pred);
  }
}
} // end namespace system::hpx::detail
THRUST_NAMESPACE_END
