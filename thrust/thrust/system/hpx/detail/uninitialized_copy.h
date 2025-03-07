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

/*! \file uninitialized_copy.h
 *  \brief HPX implementation of uninitialized_copy/uninitialized_copy_n.
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

#include <hpx/parallel/algorithms/uninitialized_copy.hpp>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace hpx
{
namespace detail
{

template <typename ExecutionPolicy, typename InputIterator, typename ForwardIterator>
ForwardIterator
uninitialized_copy(execution_policy<ExecutionPolicy>&, InputIterator first, InputIterator last, ForwardIterator result)
{
  return ::hpx::uninitialized_copy(first, last, result);
}

template <typename ExecutionPolicy, typename InputIterator, typename Size, typename ForwardIterator>
ForwardIterator
uninitialized_copy_n(execution_policy<ExecutionPolicy>&, InputIterator first, Size n, ForwardIterator result)
{
  return ::hpx::uninitialized_copy_n(first, n, result);
}

} // end namespace detail
} // end namespace hpx
} // end namespace system
THRUST_NAMESPACE_END
