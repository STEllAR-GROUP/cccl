// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*! \file unique_by_key.h
 *   \brief HPX implementation of unique_by_key.
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

#include <thrust/system/detail/generic/unique_by_key.h>
#include <thrust/system/hpx/detail/execution_policy.h>

#include <cuda/std/__utility/pair.h>

THRUST_NAMESPACE_BEGIN
namespace system::hpx::detail
{
template <typename ExecutionPolicy, typename ForwardIterator1, typename ForwardIterator2>
::cuda::std::pair<ForwardIterator1, ForwardIterator2> unique_by_key(
  execution_policy<ExecutionPolicy>& exec,
  ForwardIterator1 keys_first,
  ForwardIterator1 keys_last,
  ForwardIterator2 values_first)
{
  return thrust::system::detail::generic::unique_by_key(exec, keys_first, keys_last, values_first);
}

template <typename ExecutionPolicy, typename ForwardIterator1, typename ForwardIterator2, typename BinaryPredicate>
::cuda::std::pair<ForwardIterator1, ForwardIterator2> unique_by_key(
  execution_policy<ExecutionPolicy>& exec,
  ForwardIterator1 keys_first,
  ForwardIterator1 keys_last,
  ForwardIterator2 values_first,
  BinaryPredicate binary_pred)
{
  return thrust::system::detail::generic::unique_by_key(exec, keys_first, keys_last, values_first, binary_pred);
}

template <typename ExecutionPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2>
::cuda::std::pair<OutputIterator1, OutputIterator2> unique_by_key_copy(
  execution_policy<ExecutionPolicy>& exec,
  InputIterator1 keys_first,
  InputIterator1 keys_last,
  InputIterator2 values_first,
  OutputIterator1 keys_output,
  OutputIterator2 values_output)
{
  return thrust::system::detail::generic::unique_by_key_copy(
    exec, keys_first, keys_last, values_first, keys_output, values_output);
}

template <typename ExecutionPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2,
          typename BinaryPredicate>
::cuda::std::pair<OutputIterator1, OutputIterator2> unique_by_key_copy(
  execution_policy<ExecutionPolicy>& exec,
  InputIterator1 keys_first,
  InputIterator1 keys_last,
  InputIterator2 values_first,
  OutputIterator1 keys_output,
  OutputIterator2 values_output,
  BinaryPredicate binary_pred)
{
  return thrust::system::detail::generic::unique_by_key_copy(
    exec, keys_first, keys_last, values_first, keys_output, values_output, binary_pred);
}
} // namespace system::hpx::detail
THRUST_NAMESPACE_END
