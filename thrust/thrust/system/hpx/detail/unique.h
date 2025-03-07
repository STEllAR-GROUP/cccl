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

/*! \file unique.h
 *  \brief HPX implementation of unique.
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

#include <hpx/parallel/algorithms/unique.hpp>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace hpx
{
namespace detail
{

template <typename DerivedPolicy, typename ForwardIterator>
ForwardIterator unique(execution_policy<DerivedPolicy>&, ForwardIterator first, ForwardIterator last)
{
  return ::hpx::unique(first, last);
}

template <typename DerivedPolicy, typename ForwardIterator, typename BinaryPredicate>
ForwardIterator
unique(execution_policy<DerivedPolicy>&, ForwardIterator first, ForwardIterator last, BinaryPredicate binary_pred)
{
  return ::hpx::unique(first, last, binary_pred);
}

template <typename DerivedPolicy, typename InputIterator, typename OutputIterator>
OutputIterator
unique_copy(execution_policy<DerivedPolicy>&, InputIterator first, InputIterator last, OutputIterator output)
{
  return ::hpx::unique_copy(first, last, output);
}

template <typename DerivedPolicy, typename InputIterator, typename OutputIterator, typename BinaryPredicate>
OutputIterator unique_copy(
  execution_policy<DerivedPolicy>&,
  InputIterator first,
  InputIterator last,
  OutputIterator output,
  BinaryPredicate binary_pred)
{
  return ::hpx::unique_copy(first, last, output, binary_pred);
}


} // end namespace detail
} // end namespace hpx
} // end namespace system
THRUST_NAMESPACE_END
