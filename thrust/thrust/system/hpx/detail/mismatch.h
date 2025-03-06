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

/*! \file mismatch.h
 *  \brief HPX implementation of mismatch.
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
#include <thrust/pair.h>
#include <thrust/system/hpx/detail/execution_policy.h>

#include <hpx/parallel/algorithms/mismatch.hpp>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace hpx
{
namespace detail
{

template <typename DerivedPolicy, typename InputIterator1, typename InputIterator2>
_CCCL_HOST_DEVICE pair<InputIterator1, InputIterator2>
mismatch(execution_policy<DerivedPolicy>&, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2)
{
  return ::hpx::mismatch(first1, last1, first2);
}

template <typename DerivedPolicy, typename InputIterator1, typename InputIterator2, typename BinaryPredicate>
_CCCL_HOST_DEVICE pair<InputIterator1, InputIterator2> mismatch(
  execution_policy<DerivedPolicy>&,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  BinaryPredicate pred)
{
  return ::hpx::mismatch(first1, last1, first2, pred);
}

} // end namespace detail
} // end namespace hpx
} // end namespace system
THRUST_NAMESPACE_END
