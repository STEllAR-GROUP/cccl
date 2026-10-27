// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*! \file uninitialized_fill.h
 *   \brief HPX implementation of uninitialized_fill algorithms.
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

#include <thrust/system/detail/generic/uninitialized_fill.h>
#include <thrust/system/hpx/detail/execution_policy.h>

THRUST_NAMESPACE_BEGIN
namespace system::hpx::detail
{
template <typename ExecutionPolicy, typename ForwardIterator, typename T>
void uninitialized_fill(execution_policy<ExecutionPolicy>& exec, ForwardIterator first, ForwardIterator last, const T& x)
{
  thrust::system::detail::generic::uninitialized_fill(exec, first, last, x);
}

template <typename ExecutionPolicy, typename ForwardIterator, typename Size, typename T>
ForwardIterator uninitialized_fill_n(execution_policy<ExecutionPolicy>& exec, ForwardIterator first, Size n, const T& x)
{
  return thrust::system::detail::generic::uninitialized_fill_n(exec, first, n, x);
}
} // namespace system::hpx::detail
THRUST_NAMESPACE_END
