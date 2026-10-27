// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*! \file scan_by_key.h
 *   \brief HPX implementation of scan_by_key.
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

#include <thrust/system/detail/generic/scan_by_key.h>
#include <thrust/system/hpx/detail/execution_policy.h>

THRUST_NAMESPACE_BEGIN
namespace system::hpx::detail
{
template <typename ExecutionPolicy, typename InputIterator1, typename InputIterator2, typename OutputIterator>
OutputIterator inclusive_scan_by_key(
  execution_policy<ExecutionPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  OutputIterator result)
{
  return thrust::system::detail::generic::inclusive_scan_by_key(exec, first1, last1, first2, result);
}

template <typename ExecutionPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename BinaryPredicate>
OutputIterator inclusive_scan_by_key(
  execution_policy<ExecutionPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  OutputIterator result,
  BinaryPredicate binary_pred)
{
  return thrust::system::detail::generic::inclusive_scan_by_key(exec, first1, last1, first2, result, binary_pred);
}

template <typename ExecutionPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename BinaryPredicate,
          typename AssociativeOperator>
OutputIterator inclusive_scan_by_key(
  execution_policy<ExecutionPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  OutputIterator result,
  BinaryPredicate binary_pred,
  AssociativeOperator binary_op)
{
  return thrust::system::detail::generic::inclusive_scan_by_key(
    exec, first1, last1, first2, result, binary_pred, binary_op);
}

template <typename ExecutionPolicy, typename InputIterator1, typename InputIterator2, typename OutputIterator>
OutputIterator exclusive_scan_by_key(
  execution_policy<ExecutionPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  OutputIterator result)
{
  return thrust::system::detail::generic::exclusive_scan_by_key(exec, first1, last1, first2, result);
}

template <typename ExecutionPolicy, typename InputIterator1, typename InputIterator2, typename OutputIterator, typename T>
OutputIterator exclusive_scan_by_key(
  execution_policy<ExecutionPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  OutputIterator result,
  T init)
{
  return thrust::system::detail::generic::exclusive_scan_by_key(exec, first1, last1, first2, result, init);
}

template <typename ExecutionPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename T,
          typename BinaryPredicate>
OutputIterator exclusive_scan_by_key(
  execution_policy<ExecutionPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  OutputIterator result,
  T init,
  BinaryPredicate binary_pred)
{
  return thrust::system::detail::generic::exclusive_scan_by_key(exec, first1, last1, first2, result, init, binary_pred);
}

template <typename ExecutionPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename T,
          typename BinaryPredicate,
          typename AssociativeOperator>
OutputIterator exclusive_scan_by_key(
  execution_policy<ExecutionPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  OutputIterator result,
  T init,
  BinaryPredicate binary_pred,
  AssociativeOperator binary_op)
{
  return thrust::system::detail::generic::exclusive_scan_by_key(
    exec, first1, last1, first2, result, init, binary_pred, binary_op);
}
} // namespace system::hpx::detail
THRUST_NAMESPACE_END
