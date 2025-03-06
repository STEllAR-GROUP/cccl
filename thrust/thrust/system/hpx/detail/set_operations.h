/*
 *  Copyright 2008-2025 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*! \file set_operations.h
 *  \brief HPX implementation of set_difference/set_intersection/set_symmetric_difference/set_union.
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
#include <thrust/system/hpx/detail/execution_policy.h>

#include <hpx/parallel/algorithms/set_difference.hpp>
#include <hpx/parallel/algorithms/set_intersection.hpp>
#include <hpx/parallel/algorithms/set_symmetric_difference.hpp>
#include <hpx/parallel/algorithms/set_union.hpp>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace hpx
{
namespace detail
{

template <typename ExecutionPolicy, typename InputIterator1, typename InputIterator2, typename OutputIterator>
OutputIterator set_difference(
  execution_policy<ExecutionPolicy>&,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  InputIterator2 last2,
  OutputIterator result)
{
  return ::hpx::set_difference(first1, last1, first2, last2, result);
}

template <typename ExecutionPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename StrictWeakOrdering>
OutputIterator set_difference(
  execution_policy<ExecutionPolicy>&,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  InputIterator2 last2,
  OutputIterator result,
  StrictWeakOrdering comp)
{
  return ::hpx::set_difference(first1, last1, first2, last2, result, comp);
}

template <typename ExecutionPolicy, typename InputIterator1, typename InputIterator2, typename OutputIterator>
OutputIterator set_intersection(
  execution_policy<ExecutionPolicy>&,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  InputIterator2 last2,
  OutputIterator result)
{
  return ::hpx::set_intersection(first1, last1, first2, last2, result);
}

template <typename ExecutionPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename StrictWeakOrdering>
OutputIterator set_intersection(
  execution_policy<ExecutionPolicy>&,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  InputIterator2 last2,
  OutputIterator result,
  StrictWeakOrdering comp)
{
  return ::hpx::set_intersection(first1, last1, first2, last2, result, comp);
}

template <typename ExecutionPolicy, typename InputIterator1, typename InputIterator2, typename OutputIterator>
OutputIterator set_symmetric_difference(
  execution_policy<ExecutionPolicy>&,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  InputIterator2 last2,
  OutputIterator result)
{
  return ::hpx::set_symmetric_difference(first1, last1, first2, last2, result);
}

template <typename ExecutionPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename StrictWeakOrdering>
OutputIterator set_symmetric_difference(
  execution_policy<ExecutionPolicy>&,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  InputIterator2 last2,
  OutputIterator result,
  StrictWeakOrdering comp)
{
  return ::hpx::set_symmetric_difference(first1, last1, first2, last2, result, comp);
}

template <typename ExecutionPolicy, typename InputIterator1, typename InputIterator2, typename OutputIterator>
OutputIterator set_union(
  execution_policy<ExecutionPolicy>&,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  InputIterator2 last2,
  OutputIterator result)
{
  return ::hpx::set_union(first1, last1, first2, last2, result);
}

template <typename ExecutionPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename StrictWeakOrdering>
OutputIterator set_union(
  execution_policy<ExecutionPolicy>&,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  InputIterator2 last2,
  OutputIterator result,
  StrictWeakOrdering comp)
{
  return ::hpx::set_union(first1, last1, first2, last2, result, comp);
}

} // end namespace detail
} // end namespace hpx
} // end namespace system
THRUST_NAMESPACE_END

// this system inherits set_operations
#include <thrust/system/cpp/detail/set_operations.h>
