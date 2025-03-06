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

/*! \file sort.h
 *  \brief HPX implementation of sort.
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

#include <hpx/parallel/algorithms/is_sorted.hpp>
#include <hpx/parallel/algorithms/sort.hpp>
#include <hpx/parallel/algorithms/stable_sort.hpp>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace hpx
{
namespace detail
{

template <typename ExecutionPolicy, typename RandomAccessIterator>
void sort(execution_policy<ExecutionPolicy>&, RandomAccessIterator first, RandomAccessIterator last)
{
  return ::hpx::sort(first, last);
}

template <typename ExecutionPolicy, typename RandomAccessIterator, typename StrictWeakOrdering>
void sort(
  execution_policy<ExecutionPolicy>&, RandomAccessIterator first, RandomAccessIterator last, StrictWeakOrdering comp)
{
  return ::hpx::sort(first, last, comp);
}

template <typename ExecutionPolicy, typename RandomAccessIterator>
void stable_sort(execution_policy<ExecutionPolicy>&, RandomAccessIterator first, RandomAccessIterator last)
{
  return ::hpx::stable_sort(first, last);
}

// XXX it is an error to call this function; it has no implementation
template <typename ExecutionPolicy, typename RandomAccessIterator, typename StrictWeakOrdering>
void stable_sort(
  execution_policy<ExecutionPolicy>&, RandomAccessIterator first, RandomAccessIterator last, StrictWeakOrdering comp)
{
  return ::hpx::stable_sort(first, last, comp);
}

template <typename ExecutionPolicy, typename ForwardIterator>
bool is_sorted(execution_policy<ExecutionPolicy>&, ForwardIterator first, ForwardIterator last)
{
  return ::hpx::is_sorted(first, last);
}

template <typename ExecutionPolicy, typename ForwardIterator, typename Compare>
bool is_sorted(execution_policy<ExecutionPolicy>&, ForwardIterator first, ForwardIterator last, Compare comp)
{
  return ::hpx::is_sorted(first, last, comp);
}

template <typename ExecutionPolicy, typename ForwardIterator>
ForwardIterator is_sorted_until(execution_policy<ExecutionPolicy>&, ForwardIterator first, ForwardIterator last)
{
  return ::hpx::is_sorted_until(first, last);
}

template <typename ExecutionPolicy, typename ForwardIterator, typename Compare>
ForwardIterator
is_sorted_until(execution_policy<ExecutionPolicy>&, ForwardIterator first, ForwardIterator last, Compare comp)
{
  return ::hpx::is_sorted_until(first, last, comp);
}

} // end namespace detail
} // end namespace hpx
} // end namespace system
THRUST_NAMESPACE_END
