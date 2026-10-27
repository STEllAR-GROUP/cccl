// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*! \file tabulate.h
 *   \brief HPX implementation of tabulate algorithms.
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

#include <thrust/system/detail/generic/tabulate.h>
#include <thrust/system/hpx/detail/execution_policy.h>

THRUST_NAMESPACE_BEGIN
namespace system::hpx::detail
{
template <typename ExecutionPolicy, typename ForwardIterator, typename UnaryOperation>
void tabulate(
  execution_policy<ExecutionPolicy>& exec, ForwardIterator first, ForwardIterator last, UnaryOperation unary_op)
{
  thrust::system::detail::generic::tabulate(exec, first, last, unary_op);
}
} // end namespace system::hpx::detail
THRUST_NAMESPACE_END
