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

/*! \file extrema.h
 *  \brief HPX implementation of min_element/max_element/minmax_element.
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

#include <hpx/parallel/algorithms/minmax.hpp>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace hpx
{
namespace detail
{

template <typename DerivedPolicy, typename ForwardIterator>
ForwardIterator max_element(execution_policy<DerivedPolicy>&, ForwardIterator first, ForwardIterator last)
{
  return ::hpx::max_element(first, last);
}

template <typename DerivedPolicy, typename ForwardIterator, typename BinaryPredicate>
ForwardIterator
max_element(execution_policy<DerivedPolicy>&, ForwardIterator first, ForwardIterator last, BinaryPredicate comp)
{
  return ::hpx::max_element(first, last, comp);
}

template <typename DerivedPolicy, typename ForwardIterator>
ForwardIterator min_element(execution_policy<DerivedPolicy>&, ForwardIterator first, ForwardIterator last)
{
  return ::hpx::min_element(first, last);
}

template <typename DerivedPolicy, typename ForwardIterator, typename BinaryPredicate>
ForwardIterator
min_element(execution_policy<DerivedPolicy>&, ForwardIterator first, ForwardIterator last, BinaryPredicate comp)
{
  return ::hpx::min_element(first, last, comp);
}

template <typename DerivedPolicy, typename ForwardIterator>
pair<ForwardIterator, ForwardIterator>
minmax_element(execution_policy<DerivedPolicy>&, ForwardIterator first, ForwardIterator last)
{
  auto r = ::hpx::minmax_element(first, last);
  return pair<ForwardIterator, ForwardIterator>(r.min, r.max);
}

template <typename DerivedPolicy, typename ForwardIterator, typename BinaryPredicate>
pair<ForwardIterator, ForwardIterator>
minmax_element(execution_policy<DerivedPolicy>&, ForwardIterator first, ForwardIterator last, BinaryPredicate comp)
{
  auto r = ::hpx::minmax_element(first, last, comp);
  return pair<ForwardIterator, ForwardIterator>(r.min, r.max);
}

} // end namespace detail
} // end namespace hpx
} // end namespace system
THRUST_NAMESPACE_END
