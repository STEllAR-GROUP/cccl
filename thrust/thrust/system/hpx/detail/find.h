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

/*! \file find.h
 *  \brief HPX implementation of find, find_if, and find_if_not.
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
#include <thrust/system/hpx/detail/execution_policy.h>
#include <thrust/system/hpx/detail/runtime.h>

#include <hpx/parallel/algorithms/find.hpp>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace hpx
{
namespace detail
{

// Helper trait to detect if an iterator's reference type can be bound to a non-const lvalue reference
// HPX's parallel algorithms expect this property, but many Thrust iterators return rvalues
template <typename Iterator>
struct iterator_reference_is_lvalue_reference
{
  using reference_type        = typename thrust::iterator_traits<Iterator>::reference;
  static constexpr bool value = std::is_lvalue_reference_v<reference_type>;
};

template <typename DerivedPolicy, typename InputIterator, typename T>
InputIterator find(execution_policy<DerivedPolicy>& exec, InputIterator first, InputIterator last, const T& value)
{
  if constexpr (::hpx::traits::is_forward_iterator_v<InputIterator>
                && iterator_reference_is_lvalue_reference<InputIterator>::value)
  {
    return hpx::detail::run_as_hpx_thread([&] {
      return ::hpx::find(hpx::detail::to_hpx_execution_policy(exec), first, last, value);
    });
  }
  else
  {
    (void) exec;
    return ::hpx::find(first, last, value);
  }
}

template <typename DerivedPolicy, typename InputIterator, typename Predicate>
InputIterator find_if(execution_policy<DerivedPolicy>& exec, InputIterator first, InputIterator last, Predicate pred)
{
  if constexpr (::hpx::traits::is_forward_iterator_v<InputIterator>
                && iterator_reference_is_lvalue_reference<InputIterator>::value)
  {
    return hpx::detail::run_as_hpx_thread([&] {
      return ::hpx::find_if(hpx::detail::to_hpx_execution_policy(exec), first, last, pred);
    });
  }
  else
  {
    (void) exec;
    return ::hpx::find_if(first, last, pred);
  }
}

template <typename DerivedPolicy, typename InputIterator, typename Predicate>
InputIterator find_if_not(execution_policy<DerivedPolicy>& exec, InputIterator first, InputIterator last, Predicate pred)
{
  if constexpr (::hpx::traits::is_forward_iterator_v<InputIterator>
                && iterator_reference_is_lvalue_reference<InputIterator>::value)
  {
    return hpx::detail::run_as_hpx_thread([&] {
      return ::hpx::find_if_not(hpx::detail::to_hpx_execution_policy(exec), first, last, pred);
    });
  }
  else
  {
    (void) exec;
    return ::hpx::find_if_not(first, last, pred);
  }
}

} // end namespace detail
} // end namespace hpx
} // end namespace system
THRUST_NAMESPACE_END
