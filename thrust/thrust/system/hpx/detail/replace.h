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

/*! \file replace.h
 *  \brief HPX implementation of replace/replace_copy/replace_if/replace_copy_if.
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

#include <hpx/parallel/algorithms/replace.hpp>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace hpx
{
namespace detail
{

template <typename DerivedPolicy, typename ForwardIterator, typename T>
void replace(
  execution_policy<DerivedPolicy>&, ForwardIterator first, ForwardIterator last, const T& old_value, const T& new_value)
{
  return ::hpx::replace(first, last, old_value, new_value);
}

template <typename DerivedPolicy, typename ForwardIterator, typename Predicate, typename T>
void replace_if(
  execution_policy<DerivedPolicy>& exec, ForwardIterator first, ForwardIterator last, Predicate pred, const T& new_value)
{
  return ::hpx::replace_if(first, last, pred, new_value);
}

template <typename DerivedPolicy, typename InputIterator, typename OutputIterator, typename T>
OutputIterator replace_copy(
  execution_policy<DerivedPolicy>&,
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  const T& old_value,
  const T& new_value)
{
  return ::hpx::replace_copy(first, last, result, old_value, new_value);
}

template <typename DerivedPolicy, typename InputIterator, typename OutputIterator, typename Predicate, typename T>
OutputIterator replace_copy_if(
  execution_policy<DerivedPolicy>&,
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  Predicate pred,
  const T& new_value)
{
  return ::hpx::replace_copy_if(first, last, result, pred, new_value);
}

} // end namespace detail
} // end namespace hpx
} // end namespace system
THRUST_NAMESPACE_END
