// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*! \file uninitialized_copy.h
 *   \brief HPX implementation of uninitialized_copy algorithms.
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

#include <thrust/system/detail/generic/uninitialized_copy.h>
#include <thrust/system/hpx/detail/execution_policy.h>

THRUST_NAMESPACE_BEGIN
namespace system::hpx::detail
{
template <typename ExecutionPolicy, typename InputIterator, typename ForwardIterator>
ForwardIterator uninitialized_copy(
  execution_policy<ExecutionPolicy>& exec, InputIterator first, InputIterator last, ForwardIterator result)
{
  return thrust::system::detail::generic::uninitialized_copy(exec, first, last, result);
}

template <typename ExecutionPolicy, typename InputIterator, typename Size, typename ForwardIterator>
ForwardIterator
uninitialized_copy_n(execution_policy<ExecutionPolicy>& exec, InputIterator first, Size n, ForwardIterator result)
{
  return thrust::system::detail::generic::uninitialized_copy_n(exec, first, n, result);
}
} // namespace system::hpx::detail
THRUST_NAMESPACE_END
