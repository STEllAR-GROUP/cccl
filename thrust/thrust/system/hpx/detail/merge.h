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

/*! \file merge.h
 *  \brief HPX implementation of merge.
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
#include <thrust/system/hpx/detail/function.h>
#include <thrust/system/hpx/detail/runtime.h>
#include <thrust/type_traits/is_contiguous_iterator.h>
#include <thrust/type_traits/unwrap_contiguous_iterator.h>

#include <hpx/parallel/algorithms/merge.hpp>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace hpx
{
namespace detail
{

template <typename ExecutionPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename StrictWeakOrdering>
OutputIterator
merge(execution_policy<ExecutionPolicy>& exec,
      InputIterator1 first1,
      InputIterator1 last1,
      InputIterator2 first2,
      InputIterator2 last2,
      OutputIterator result,
      StrictWeakOrdering comp)
{
  // wrap comp
  wrapped_function<StrictWeakOrdering> wrapped_comp{comp};

  if constexpr (::hpx::traits::is_forward_iterator_v<InputIterator1>
                && ::hpx::traits::is_forward_iterator_v<InputIterator2>
                && ::hpx::traits::is_forward_iterator_v<OutputIterator>)
  {
    return hpx::detail::run_as_hpx_thread([&] {
      auto res = ::hpx::merge(
        hpx::detail::to_hpx_execution_policy(exec),
        thrust::try_unwrap_contiguous_iterator(first1),
        thrust::try_unwrap_contiguous_iterator(last1),
        thrust::try_unwrap_contiguous_iterator(first2),
        thrust::try_unwrap_contiguous_iterator(last2),
        thrust::try_unwrap_contiguous_iterator(result),
        wrapped_comp);
      if constexpr (thrust::is_contiguous_iterator_v<OutputIterator>)
      { // rewrap
        return result + (res - thrust::try_unwrap_contiguous_iterator(result));
      }
      else
      {
        return res;
      }
    });
  }
  else
  {
    (void) exec;
    return ::hpx::merge(first1, last1, first2, last2, result, wrapped_comp);
  }
}

} // end namespace detail
} // end namespace hpx
} // end namespace system
THRUST_NAMESPACE_END

// this system inherits merge_by_key
#include <thrust/system/cpp/detail/merge.h>
