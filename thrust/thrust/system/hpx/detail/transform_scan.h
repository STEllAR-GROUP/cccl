// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*! \file transform_scan.h
 *   \brief HPX implementation of transform_scan functions.
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

#include <thrust/system/detail/generic/transform_scan.h>
#include <thrust/system/hpx/detail/execution_policy.h>

THRUST_NAMESPACE_BEGIN
namespace system::hpx::detail
{
template <typename ExecutionPolicy,
          typename InputIterator,
          typename OutputIterator,
          typename UnaryFunction,
          typename BinaryFunction>
OutputIterator transform_inclusive_scan(
  execution_policy<ExecutionPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  UnaryFunction unary_op,
  BinaryFunction binary_op)
{
  return thrust::system::detail::generic::transform_inclusive_scan(exec, first, last, result, unary_op, binary_op);
}

template <typename ExecutionPolicy,
          typename InputIterator,
          typename OutputIterator,
          typename UnaryFunction,
          typename T,
          typename BinaryFunction>
OutputIterator transform_inclusive_scan(
  execution_policy<ExecutionPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  UnaryFunction unary_op,
  T init,
  BinaryFunction binary_op)
{
  return thrust::system::detail::generic::transform_inclusive_scan(exec, first, last, result, unary_op, init, binary_op);
}

template <typename ExecutionPolicy,
          typename InputIterator,
          typename OutputIterator,
          typename UnaryFunction,
          typename T,
          typename AssociativeOperator>
OutputIterator transform_exclusive_scan(
  execution_policy<ExecutionPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  UnaryFunction unary_op,
  T init,
  AssociativeOperator binary_op)
{
  return thrust::system::detail::generic::transform_exclusive_scan(exec, first, last, result, unary_op, init, binary_op);
}
} // end namespace system::hpx::detail
THRUST_NAMESPACE_END
