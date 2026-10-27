// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*! \file scatter.h
 *   \brief HPX implementation of scatter algorithms.
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

#include <thrust/system/detail/generic/scatter.h>
#include <thrust/system/hpx/detail/execution_policy.h>

THRUST_NAMESPACE_BEGIN
namespace system::hpx::detail
{
template <typename ExecutionPolicy, typename InputIterator1, typename InputIterator2, typename RandomAccessIterator>
void scatter(execution_policy<ExecutionPolicy>& exec,
             InputIterator1 first,
             InputIterator1 last,
             InputIterator2 map,
             RandomAccessIterator output)
{
  thrust::system::detail::generic::scatter(exec, first, last, map, output);
}

template <typename ExecutionPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename RandomAccessIterator>
void scatter_if(execution_policy<ExecutionPolicy>& exec,
                InputIterator1 first,
                InputIterator1 last,
                InputIterator2 map,
                InputIterator3 stencil,
                RandomAccessIterator output)
{
  thrust::system::detail::generic::scatter_if(exec, first, last, map, stencil, output);
}

template <typename ExecutionPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename RandomAccessIterator,
          typename Predicate>
void scatter_if(
  execution_policy<ExecutionPolicy>& exec,
  InputIterator1 first,
  InputIterator1 last,
  InputIterator2 map,
  InputIterator3 stencil,
  RandomAccessIterator output,
  Predicate pred)
{
  thrust::system::detail::generic::scatter_if(exec, first, last, map, stencil, output, pred);
}
} // namespace system::hpx::detail
THRUST_NAMESPACE_END
