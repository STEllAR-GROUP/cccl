// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*! \file gather.h
 *   \brief HPX implementation of gather algorithms.
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

#include <thrust/system/detail/generic/gather.h>
#include <thrust/system/hpx/detail/execution_policy.h>

THRUST_NAMESPACE_BEGIN
namespace system::hpx::detail
{
template <typename ExecutionPolicy, typename InputIterator, typename RandomAccessIterator, typename OutputIterator>
OutputIterator
gather(execution_policy<ExecutionPolicy>& exec,
       InputIterator map_first,
       InputIterator map_last,
       RandomAccessIterator input_first,
       OutputIterator result)
{
  return thrust::system::detail::generic::gather(exec, map_first, map_last, input_first, result);
}

template <typename ExecutionPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename RandomAccessIterator,
          typename OutputIterator>
OutputIterator gather_if(
  execution_policy<ExecutionPolicy>& exec,
  InputIterator1 map_first,
  InputIterator1 map_last,
  InputIterator2 stencil,
  RandomAccessIterator input_first,
  OutputIterator result)
{
  return thrust::system::detail::generic::gather_if(exec, map_first, map_last, stencil, input_first, result);
}

template <typename ExecutionPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename RandomAccessIterator,
          typename OutputIterator,
          typename Predicate>
OutputIterator gather_if(
  execution_policy<ExecutionPolicy>& exec,
  InputIterator1 map_first,
  InputIterator1 map_last,
  InputIterator2 stencil,
  RandomAccessIterator input_first,
  OutputIterator result,
  Predicate pred)
{
  return thrust::system::detail::generic::gather_if(exec, map_first, map_last, stencil, input_first, result, pred);
}
} // namespace system::hpx::detail
THRUST_NAMESPACE_END
