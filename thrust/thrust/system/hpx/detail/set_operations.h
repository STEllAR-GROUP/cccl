// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*! \file set_operations.h
 *   \brief HPX implementation of set operations.
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
#include <thrust/system/detail/generic/set_operations.h>
#include <thrust/system/hpx/detail/contiguous_iterator.h>
#include <thrust/system/hpx/detail/execution_policy.h>
#include <thrust/system/hpx/detail/function.h>

#include <cuda/std/__functional/operations.h>
#include <cuda/std/__utility/pair.h>

#include <hpx/parallel/algorithms/set_difference.hpp>
#include <hpx/parallel/algorithms/set_intersection.hpp>
#include <hpx/parallel/algorithms/set_symmetric_difference.hpp>
#include <hpx/parallel/algorithms/set_union.hpp>

THRUST_NAMESPACE_BEGIN
namespace system::hpx::detail
{
// set_union
template <typename ExecutionPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename StrictWeakOrdering>
OutputIterator set_union(
  execution_policy<ExecutionPolicy>& exec [[maybe_unused]],
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  InputIterator2 last2,
  OutputIterator result,
  StrictWeakOrdering comp)
{
  hpx_wrapped_function<StrictWeakOrdering> wrapped_comp{comp};

  if constexpr (::hpx::traits::is_forward_iterator_v<InputIterator1>
                && ::hpx::traits::is_forward_iterator_v<InputIterator2>
                && ::hpx::traits::is_forward_iterator_v<OutputIterator>)
  {
    auto res = ::hpx::set_union(
      hpx::detail::to_hpx_execution_policy(exec),
      ::thrust::try_unwrap_contiguous_iterator(first1),
      ::thrust::try_unwrap_contiguous_iterator(last1),
      ::thrust::try_unwrap_contiguous_iterator(first2),
      ::thrust::try_unwrap_contiguous_iterator(last2),
      ::thrust::try_unwrap_contiguous_iterator(result),
      wrapped_comp);
    return detail::rewrap_contiguous_iterator(res, result);
  }
  else
  {
    return ::hpx::set_union(first1, last1, first2, last2, result, wrapped_comp);
  }
}

template <typename ExecutionPolicy, typename InputIterator1, typename InputIterator2, typename OutputIterator>
OutputIterator set_union(
  execution_policy<ExecutionPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  InputIterator2 last2,
  OutputIterator result)
{
  using value_type = thrust::detail::it_value_t<InputIterator1>;
  return set_union(exec, first1, last1, first2, last2, result, ::cuda::std::less<value_type>());
}

// set_intersection
template <typename ExecutionPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename StrictWeakOrdering>
OutputIterator set_intersection(
  execution_policy<ExecutionPolicy>& exec [[maybe_unused]],
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  InputIterator2 last2,
  OutputIterator result,
  StrictWeakOrdering comp)
{
  hpx_wrapped_function<StrictWeakOrdering> wrapped_comp{comp};

  if constexpr (::hpx::traits::is_forward_iterator_v<InputIterator1>
                && ::hpx::traits::is_forward_iterator_v<InputIterator2>
                && ::hpx::traits::is_forward_iterator_v<OutputIterator>)
  {
    auto res = ::hpx::set_intersection(
      hpx::detail::to_hpx_execution_policy(exec),
      ::thrust::try_unwrap_contiguous_iterator(first1),
      ::thrust::try_unwrap_contiguous_iterator(last1),
      ::thrust::try_unwrap_contiguous_iterator(first2),
      ::thrust::try_unwrap_contiguous_iterator(last2),
      ::thrust::try_unwrap_contiguous_iterator(result),
      wrapped_comp);
    return detail::rewrap_contiguous_iterator(res, result);
  }
  else
  {
    return ::hpx::set_intersection(first1, last1, first2, last2, result, wrapped_comp);
  }
}

template <typename ExecutionPolicy, typename InputIterator1, typename InputIterator2, typename OutputIterator>
OutputIterator set_intersection(
  execution_policy<ExecutionPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  InputIterator2 last2,
  OutputIterator result)
{
  using value_type = thrust::detail::it_value_t<InputIterator1>;
  return set_intersection(exec, first1, last1, first2, last2, result, ::cuda::std::less<value_type>());
}

// set_difference
template <typename ExecutionPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename StrictWeakOrdering>
OutputIterator set_difference(
  execution_policy<ExecutionPolicy>& exec [[maybe_unused]],
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  InputIterator2 last2,
  OutputIterator result,
  StrictWeakOrdering comp)
{
  hpx_wrapped_function<StrictWeakOrdering> wrapped_comp{comp};

  if constexpr (::hpx::traits::is_forward_iterator_v<InputIterator1>
                && ::hpx::traits::is_forward_iterator_v<InputIterator2>
                && ::hpx::traits::is_forward_iterator_v<OutputIterator>)
  {
    auto res = ::hpx::set_difference(
      hpx::detail::to_hpx_execution_policy(exec),
      ::thrust::try_unwrap_contiguous_iterator(first1),
      ::thrust::try_unwrap_contiguous_iterator(last1),
      ::thrust::try_unwrap_contiguous_iterator(first2),
      ::thrust::try_unwrap_contiguous_iterator(last2),
      ::thrust::try_unwrap_contiguous_iterator(result),
      wrapped_comp);
    return detail::rewrap_contiguous_iterator(res, result);
  }
  else
  {
    return ::hpx::set_difference(first1, last1, first2, last2, result, wrapped_comp);
  }
}

template <typename ExecutionPolicy, typename InputIterator1, typename InputIterator2, typename OutputIterator>
OutputIterator set_difference(
  execution_policy<ExecutionPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  InputIterator2 last2,
  OutputIterator result)
{
  using value_type = thrust::detail::it_value_t<InputIterator1>;
  return set_difference(exec, first1, last1, first2, last2, result, ::cuda::std::less<value_type>());
}

// set_symmetric_difference
template <typename ExecutionPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename StrictWeakOrdering>
OutputIterator set_symmetric_difference(
  execution_policy<ExecutionPolicy>& exec [[maybe_unused]],
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  InputIterator2 last2,
  OutputIterator result,
  StrictWeakOrdering comp)
{
  hpx_wrapped_function<StrictWeakOrdering> wrapped_comp{comp};

  if constexpr (::hpx::traits::is_forward_iterator_v<InputIterator1>
                && ::hpx::traits::is_forward_iterator_v<InputIterator2>
                && ::hpx::traits::is_forward_iterator_v<OutputIterator>)
  {
    auto res = ::hpx::set_symmetric_difference(
      hpx::detail::to_hpx_execution_policy(exec),
      ::thrust::try_unwrap_contiguous_iterator(first1),
      ::thrust::try_unwrap_contiguous_iterator(last1),
      ::thrust::try_unwrap_contiguous_iterator(first2),
      ::thrust::try_unwrap_contiguous_iterator(last2),
      ::thrust::try_unwrap_contiguous_iterator(result),
      wrapped_comp);
    return detail::rewrap_contiguous_iterator(res, result);
  }
  else
  {
    return ::hpx::set_symmetric_difference(first1, last1, first2, last2, result, wrapped_comp);
  }
}

template <typename ExecutionPolicy, typename InputIterator1, typename InputIterator2, typename OutputIterator>
OutputIterator set_symmetric_difference(
  execution_policy<ExecutionPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  InputIterator2 last2,
  OutputIterator result)
{
  using value_type = thrust::detail::it_value_t<InputIterator1>;
  return set_symmetric_difference(exec, first1, last1, first2, last2, result, ::cuda::std::less<value_type>());
}

// by_key operations
template <typename ExecutionPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename InputIterator4,
          typename OutputIterator1,
          typename OutputIterator2,
          typename StrictWeakOrdering>
::cuda::std::pair<OutputIterator1, OutputIterator2> set_difference_by_key(
  execution_policy<ExecutionPolicy>& exec,
  InputIterator1 keys_first1,
  InputIterator1 keys_last1,
  InputIterator2 keys_first2,
  InputIterator2 keys_last2,
  InputIterator3 values_first1,
  InputIterator4 values_first2,
  OutputIterator1 keys_result,
  OutputIterator2 values_result,
  StrictWeakOrdering comp)
{
  return thrust::system::detail::generic::set_difference_by_key(
    exec,
    keys_first1,
    keys_last1,
    keys_first2,
    keys_last2,
    values_first1,
    values_first2,
    keys_result,
    values_result,
    comp);
}

template <typename ExecutionPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename InputIterator4,
          typename OutputIterator1,
          typename OutputIterator2>
::cuda::std::pair<OutputIterator1, OutputIterator2> set_difference_by_key(
  execution_policy<ExecutionPolicy>& exec,
  InputIterator1 keys_first1,
  InputIterator1 keys_last1,
  InputIterator2 keys_first2,
  InputIterator2 keys_last2,
  InputIterator3 values_first1,
  InputIterator4 values_first2,
  OutputIterator1 keys_result,
  OutputIterator2 values_result)
{
  return thrust::system::detail::generic::set_difference_by_key(
    exec, keys_first1, keys_last1, keys_first2, keys_last2, values_first1, values_first2, keys_result, values_result);
}

template <typename ExecutionPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename OutputIterator1,
          typename OutputIterator2,
          typename StrictWeakOrdering>
::cuda::std::pair<OutputIterator1, OutputIterator2> set_intersection_by_key(
  execution_policy<ExecutionPolicy>& exec,
  InputIterator1 keys_first1,
  InputIterator1 keys_last1,
  InputIterator2 keys_first2,
  InputIterator2 keys_last2,
  InputIterator3 values_first1,
  OutputIterator1 keys_result,
  OutputIterator2 values_result,
  StrictWeakOrdering comp)
{
  return thrust::system::detail::generic::set_intersection_by_key(
    exec, keys_first1, keys_last1, keys_first2, keys_last2, values_first1, keys_result, values_result, comp);
}

template <typename ExecutionPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename OutputIterator1,
          typename OutputIterator2>
::cuda::std::pair<OutputIterator1, OutputIterator2> set_intersection_by_key(
  execution_policy<ExecutionPolicy>& exec,
  InputIterator1 keys_first1,
  InputIterator1 keys_last1,
  InputIterator2 keys_first2,
  InputIterator2 keys_last2,
  InputIterator3 values_first1,
  OutputIterator1 keys_result,
  OutputIterator2 values_result)
{
  return thrust::system::detail::generic::set_intersection_by_key(
    exec, keys_first1, keys_last1, keys_first2, keys_last2, values_first1, keys_result, values_result);
}

template <typename ExecutionPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename InputIterator4,
          typename OutputIterator1,
          typename OutputIterator2,
          typename StrictWeakOrdering>
::cuda::std::pair<OutputIterator1, OutputIterator2> set_symmetric_difference_by_key(
  execution_policy<ExecutionPolicy>& exec,
  InputIterator1 keys_first1,
  InputIterator1 keys_last1,
  InputIterator2 keys_first2,
  InputIterator2 keys_last2,
  InputIterator3 values_first1,
  InputIterator4 values_first2,
  OutputIterator1 keys_result,
  OutputIterator2 values_result,
  StrictWeakOrdering comp)
{
  return thrust::system::detail::generic::set_symmetric_difference_by_key(
    exec,
    keys_first1,
    keys_last1,
    keys_first2,
    keys_last2,
    values_first1,
    values_first2,
    keys_result,
    values_result,
    comp);
}

template <typename ExecutionPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename InputIterator4,
          typename OutputIterator1,
          typename OutputIterator2>
::cuda::std::pair<OutputIterator1, OutputIterator2> set_symmetric_difference_by_key(
  execution_policy<ExecutionPolicy>& exec,
  InputIterator1 keys_first1,
  InputIterator1 keys_last1,
  InputIterator2 keys_first2,
  InputIterator2 keys_last2,
  InputIterator3 values_first1,
  InputIterator4 values_first2,
  OutputIterator1 keys_result,
  OutputIterator2 values_result)
{
  return thrust::system::detail::generic::set_symmetric_difference_by_key(
    exec, keys_first1, keys_last1, keys_first2, keys_last2, values_first1, values_first2, keys_result, values_result);
}

template <typename ExecutionPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename InputIterator4,
          typename OutputIterator1,
          typename OutputIterator2,
          typename StrictWeakOrdering>
::cuda::std::pair<OutputIterator1, OutputIterator2> set_union_by_key(
  execution_policy<ExecutionPolicy>& exec,
  InputIterator1 keys_first1,
  InputIterator1 keys_last1,
  InputIterator2 keys_first2,
  InputIterator2 keys_last2,
  InputIterator3 values_first1,
  InputIterator4 values_first2,
  OutputIterator1 keys_result,
  OutputIterator2 values_result,
  StrictWeakOrdering comp)
{
  return thrust::system::detail::generic::set_union_by_key(
    exec,
    keys_first1,
    keys_last1,
    keys_first2,
    keys_last2,
    values_first1,
    values_first2,
    keys_result,
    values_result,
    comp);
}

template <typename ExecutionPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename InputIterator4,
          typename OutputIterator1,
          typename OutputIterator2>
::cuda::std::pair<OutputIterator1, OutputIterator2> set_union_by_key(
  execution_policy<ExecutionPolicy>& exec,
  InputIterator1 keys_first1,
  InputIterator1 keys_last1,
  InputIterator2 keys_first2,
  InputIterator2 keys_last2,
  InputIterator3 values_first1,
  InputIterator4 values_first2,
  OutputIterator1 keys_result,
  OutputIterator2 values_result)
{
  return thrust::system::detail::generic::set_union_by_key(
    exec, keys_first1, keys_last1, keys_first2, keys_last2, values_first1, values_first2, keys_result, values_result);
}
} // end namespace system::hpx::detail
THRUST_NAMESPACE_END
