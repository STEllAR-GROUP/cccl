// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*! \file sequence.h
 *   \brief HPX implementation of sequence algorithms.
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

#include <thrust/system/detail/generic/sequence.h>
#include <thrust/system/hpx/detail/execution_policy.h>

THRUST_NAMESPACE_BEGIN
namespace system::hpx::detail
{
template <typename ExecutionPolicy, typename ForwardIterator, typename T = thrust::detail::it_value_t<ForwardIterator>>
void sequence(
  execution_policy<ExecutionPolicy>& exec, ForwardIterator first, ForwardIterator last, T init = T{}, T step = T{1})
{
  thrust::system::detail::generic::sequence(exec, first, last, init, step);
}
} // end namespace system::hpx::detail
THRUST_NAMESPACE_END
