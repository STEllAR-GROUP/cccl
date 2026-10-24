// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*! \file scan.h
 *   \brief HPX implementation of prefix scan functions.
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

#include <thrust/system/detail/generic/scan.h>
#include <thrust/system/hpx/detail/contiguous_iterator.h>
#include <thrust/system/hpx/detail/execution_policy.h>
#include <thrust/system/hpx/detail/function.h>

#include <cuda/std/__iterator/distance.h>

#include <hpx/parallel/algorithms/exclusive_scan.hpp>
#include <hpx/parallel/algorithms/inclusive_scan.hpp>

THRUST_NAMESPACE_BEGIN
namespace system::hpx::detail
{
template <typename ExecutionPolicy, typename InputIterator, typename OutputIterator, typename BinaryFunction>
OutputIterator inclusive_scan(
  execution_policy<ExecutionPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  BinaryFunction binary_op)
{
  if (first == last)
  {
    return result;
  }

  hpx_wrapped_function<BinaryFunction> wrapped_binary_op{binary_op};

  if constexpr (::hpx::traits::is_forward_iterator_v<InputIterator>
                && ::hpx::traits::is_forward_iterator_v<OutputIterator>)
  {
    auto res = ::hpx::inclusive_scan(
      hpx::detail::to_hpx_execution_policy(exec),
      ::thrust::try_unwrap_contiguous_iterator(first),
      ::thrust::try_unwrap_contiguous_iterator(last),
      ::thrust::try_unwrap_contiguous_iterator(result),
      wrapped_binary_op);
    return detail::rewrap_contiguous_iterator(res, result);
  }
  else
  {
    auto res = ::hpx::inclusive_scan(first, last, result, wrapped_binary_op);
    return detail::rewrap_contiguous_iterator(res, result);
  }
}

template <typename ExecutionPolicy,
          typename InputIterator,
          typename OutputIterator,
          typename InitialValueType,
          typename BinaryFunction>
OutputIterator inclusive_scan(
  execution_policy<ExecutionPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  InitialValueType init,
  BinaryFunction binary_op)
{
  if (first == last)
  {
    return result;
  }

  hpx_wrapped_function<BinaryFunction> wrapped_binary_op{binary_op};

  if constexpr (::hpx::traits::is_forward_iterator_v<InputIterator>
                && ::hpx::traits::is_forward_iterator_v<OutputIterator>)
  {
    auto res = ::hpx::inclusive_scan(
      hpx::detail::to_hpx_execution_policy(exec),
      ::thrust::try_unwrap_contiguous_iterator(first),
      ::thrust::try_unwrap_contiguous_iterator(last),
      ::thrust::try_unwrap_contiguous_iterator(result),
      wrapped_binary_op,
      init);
    return detail::rewrap_contiguous_iterator(res, result);
  }
  else
  {
    auto res = ::hpx::inclusive_scan(first, last, result, wrapped_binary_op, init);
    return detail::rewrap_contiguous_iterator(res, result);
  }
}

template <typename ExecutionPolicy,
          typename InputIterator,
          typename OutputIterator,
          typename InitialValueType,
          typename BinaryFunction>
OutputIterator exclusive_scan(
  execution_policy<ExecutionPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  InitialValueType init,
  BinaryFunction binary_op)
{
  if (first == last)
  {
    return result;
  }

  hpx_wrapped_function<BinaryFunction> wrapped_binary_op{binary_op};

  if constexpr (::hpx::traits::is_forward_iterator_v<InputIterator>
                && ::hpx::traits::is_forward_iterator_v<OutputIterator>)
  {
    auto res = ::hpx::exclusive_scan(
      hpx::detail::to_hpx_execution_policy(exec),
      ::thrust::try_unwrap_contiguous_iterator(first),
      ::thrust::try_unwrap_contiguous_iterator(last),
      ::thrust::try_unwrap_contiguous_iterator(result),
      init,
      wrapped_binary_op);
    return detail::rewrap_contiguous_iterator(res, result);
  }
  else
  {
    auto res = ::hpx::exclusive_scan(first, last, result, init, wrapped_binary_op);
    return detail::rewrap_contiguous_iterator(res, result);
  }
}
} // end namespace system::hpx::detail
THRUST_NAMESPACE_END
