/*
 *  Copyright 2008-2018 NVIDIA Corporation
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

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header
#include <thrust/detail/allocator_aware_execution_policy.h>
#include <thrust/system/detail/sequential/execution_policy.h>

THRUST_NAMESPACE_BEGIN
namespace detail
{

struct seq_t
    : thrust::system::detail::sequential::execution_policy<seq_t>
    , thrust::detail::allocator_aware_execution_policy<thrust::system::detail::sequential::execution_policy>
{
  _CCCL_HOST_DEVICE constexpr seq_t()
      : thrust::system::detail::sequential::execution_policy<seq_t>()
  {}

  // allow any execution_policy to convert to seq_t
  template <typename DerivedPolicy>
  _CCCL_HOST_DEVICE seq_t(const thrust::execution_policy<DerivedPolicy>&)
      : thrust::system::detail::sequential::execution_policy<seq_t>()
  {}
};

} // namespace detail

_CCCL_GLOBAL_CONSTANT detail::seq_t seq;

THRUST_NAMESPACE_END
