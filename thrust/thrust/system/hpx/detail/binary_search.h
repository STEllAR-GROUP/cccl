// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*! \file binary_search.h
 *   \brief HPX implementation of binary search algorithms.
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

#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/binary_search.h>
#include <thrust/system/detail/generic/scalar/binary_search.h>
#include <thrust/system/hpx/detail/execution_policy.h>

#include <cuda/std/__functional/operations.h>
#include <cuda/std/__utility/pair.h>

THRUST_NAMESPACE_BEGIN
namespace system::hpx::detail
{
template <typename ExecutionPolicy, typename ForwardIterator, typename T, typename StrictWeakOrdering>
ForwardIterator lower_bound(
  execution_policy<ExecutionPolicy>& exec [[maybe_unused]],
  ForwardIterator begin,
  ForwardIterator end,
  const T& value,
  StrictWeakOrdering comp)
{
  return thrust::system::detail::generic::scalar::lower_bound(begin, end, value, comp);
}

template <typename ExecutionPolicy, typename ForwardIterator, typename T>
ForwardIterator
lower_bound(execution_policy<ExecutionPolicy>& exec, ForwardIterator begin, ForwardIterator end, const T& value)
{
  using value_type = thrust::detail::it_value_t<ForwardIterator>;
  return lower_bound(exec, begin, end, value, ::cuda::std::less<value_type>());
}

template <typename ExecutionPolicy,
          typename ForwardIterator,
          typename InputIterator,
          typename OutputIterator,
          typename StrictWeakOrdering>
OutputIterator lower_bound(
  execution_policy<ExecutionPolicy>& exec,
  ForwardIterator begin,
  ForwardIterator end,
  InputIterator values_begin,
  InputIterator values_end,
  OutputIterator output,
  StrictWeakOrdering comp)
{
  return thrust::system::detail::generic::lower_bound(exec, begin, end, values_begin, values_end, output, comp);
}

template <typename ExecutionPolicy, typename ForwardIterator, typename InputIterator, typename OutputIterator>
OutputIterator lower_bound(
  execution_policy<ExecutionPolicy>& exec,
  ForwardIterator begin,
  ForwardIterator end,
  InputIterator values_begin,
  InputIterator values_end,
  OutputIterator output)
{
  return thrust::system::detail::generic::lower_bound(exec, begin, end, values_begin, values_end, output);
}

template <typename ExecutionPolicy, typename ForwardIterator, typename T, typename StrictWeakOrdering>
ForwardIterator upper_bound(
  execution_policy<ExecutionPolicy>& exec [[maybe_unused]],
  ForwardIterator begin,
  ForwardIterator end,
  const T& value,
  StrictWeakOrdering comp)
{
  return thrust::system::detail::generic::scalar::upper_bound(begin, end, value, comp);
}

template <typename ExecutionPolicy, typename ForwardIterator, typename T>
ForwardIterator
upper_bound(execution_policy<ExecutionPolicy>& exec, ForwardIterator begin, ForwardIterator end, const T& value)
{
  using value_type = thrust::detail::it_value_t<ForwardIterator>;
  return upper_bound(exec, begin, end, value, ::cuda::std::less<value_type>());
}

template <typename ExecutionPolicy,
          typename ForwardIterator,
          typename InputIterator,
          typename OutputIterator,
          typename StrictWeakOrdering>
OutputIterator upper_bound(
  execution_policy<ExecutionPolicy>& exec,
  ForwardIterator begin,
  ForwardIterator end,
  InputIterator values_begin,
  InputIterator values_end,
  OutputIterator output,
  StrictWeakOrdering comp)
{
  return thrust::system::detail::generic::upper_bound(exec, begin, end, values_begin, values_end, output, comp);
}

template <typename ExecutionPolicy, typename ForwardIterator, typename InputIterator, typename OutputIterator>
OutputIterator upper_bound(
  execution_policy<ExecutionPolicy>& exec,
  ForwardIterator begin,
  ForwardIterator end,
  InputIterator values_begin,
  InputIterator values_end,
  OutputIterator output)
{
  return thrust::system::detail::generic::upper_bound(exec, begin, end, values_begin, values_end, output);
}

template <typename ExecutionPolicy, typename ForwardIterator, typename T, typename StrictWeakOrdering>
bool binary_search(execution_policy<ExecutionPolicy>& exec [[maybe_unused]],
                   ForwardIterator begin,
                   ForwardIterator end,
                   const T& value,
                   StrictWeakOrdering comp)
{
  return thrust::system::detail::generic::scalar::binary_search(begin, end, value, comp);
}

template <typename ExecutionPolicy, typename ForwardIterator, typename T>
bool binary_search(execution_policy<ExecutionPolicy>& exec, ForwardIterator begin, ForwardIterator end, const T& value)
{
  using value_type = thrust::detail::it_value_t<ForwardIterator>;
  return binary_search(exec, begin, end, value, ::cuda::std::less<value_type>());
}

template <typename ExecutionPolicy,
          typename ForwardIterator,
          typename InputIterator,
          typename OutputIterator,
          typename StrictWeakOrdering>
OutputIterator binary_search(
  execution_policy<ExecutionPolicy>& exec,
  ForwardIterator begin,
  ForwardIterator end,
  InputIterator values_begin,
  InputIterator values_end,
  OutputIterator output,
  StrictWeakOrdering comp)
{
  return thrust::system::detail::generic::binary_search(exec, begin, end, values_begin, values_end, output, comp);
}

template <typename ExecutionPolicy, typename ForwardIterator, typename InputIterator, typename OutputIterator>
OutputIterator binary_search(
  execution_policy<ExecutionPolicy>& exec,
  ForwardIterator begin,
  ForwardIterator end,
  InputIterator values_begin,
  InputIterator values_end,
  OutputIterator output)
{
  return thrust::system::detail::generic::binary_search(exec, begin, end, values_begin, values_end, output);
}

template <typename ExecutionPolicy, typename ForwardIterator, typename LessThanComparable, typename StrictWeakOrdering>
::cuda::std::pair<ForwardIterator, ForwardIterator> equal_range(
  execution_policy<ExecutionPolicy>& exec [[maybe_unused]],
  ForwardIterator first,
  ForwardIterator last,
  const LessThanComparable& value,
  StrictWeakOrdering comp)
{
  return thrust::system::detail::generic::scalar::equal_range(first, last, value, comp);
}

template <typename ExecutionPolicy, typename ForwardIterator, typename LessThanComparable>
::cuda::std::pair<ForwardIterator, ForwardIterator> equal_range(
  execution_policy<ExecutionPolicy>& exec, ForwardIterator first, ForwardIterator last, const LessThanComparable& value)
{
  using value_type = thrust::detail::it_value_t<ForwardIterator>;
  return equal_range(exec, first, last, value, ::cuda::std::less<value_type>());
}
} // end namespace system::hpx::detail
THRUST_NAMESPACE_END
