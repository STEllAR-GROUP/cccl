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

/*! \file fill.h
 *  \brief HPX implementation of fill/fill_n.
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
#include <thrust/system/hpx/detail/runtime.h>

#include <hpx/parallel/algorithms/fill.hpp>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace hpx
{
namespace detail
{

template <typename DerivedPolicy, typename ForwardIterator, typename T>
void fill(execution_policy<DerivedPolicy>& exec, ForwardIterator first, ForwardIterator last, const T& value)
{
  if constexpr (::hpx::traits::is_forward_iterator_v<ForwardIterator>)
  {
    return hpx::detail::run_as_hpx_thread([&] {
      return ::hpx::fill(hpx::detail::to_hpx_execution_policy(exec), first, last, value);
    });
  }
  else
  {
    (void) exec;
    return ::hpx::fill(first, last, value);
  }
}

template <typename DerivedPolicy, typename OutputIterator, typename Size, typename T>
OutputIterator fill_n(execution_policy<DerivedPolicy>& exec, OutputIterator first, Size n, const T& value)
{
  if constexpr (::hpx::traits::is_forward_iterator_v<OutputIterator>)
  {
    return hpx::detail::run_as_hpx_thread([&] {
      return ::hpx::fill_n(hpx::detail::to_hpx_execution_policy(exec), first, n, value);
    });
  }
  else
  {
    (void) exec;
    return ::hpx::fill_n(first, n, value);
  }
}

} // end namespace detail
} // end namespace hpx
} // end namespace system
THRUST_NAMESPACE_END
