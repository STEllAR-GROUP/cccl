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

/*! \file reverse.h
 *  \brief HPX implementation of reverse/reverse_copy.
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

#include <hpx/parallel/algorithms/reverse.hpp>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace hpx
{
namespace detail
{

template <typename DerivedPolicy, typename BidirectionalIterator>
void reverse(execution_policy<DerivedPolicy>&, BidirectionalIterator first, BidirectionalIterator last)
{
  return ::hpx::reverse(first, last);
}

template <typename DerivedPolicy, typename BidirectionalIterator, typename OutputIterator>
OutputIterator reverse_copy(
  execution_policy<DerivedPolicy>&, BidirectionalIterator first, BidirectionalIterator last, OutputIterator result)
{
  return ::hpx::reverse_copy(first, last, result);
}

} // end namespace detail
} // end namespace hpx
} // end namespace system
THRUST_NAMESPACE_END
