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
#include <thrust/system/hpx/detail/execution_policy.h>

#include <hpx/executors/execution_policy.hpp>
#include <hpx/executors/parallel_executor.hpp>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace hpx
{
namespace detail
{

template <typename Executor, typename Parameters>
struct parallel_policy_shim
    : ::hpx::execution::detail::execution_policy<parallel_policy_shim, Executor, Parameters>
    , execution_policy<parallel_policy_shim<Executor, Parameters>>
{
  using base_type = ::hpx::execution::detail::execution_policy<parallel_policy_shim, Executor, Parameters>;

  using base_type::base_type;
};

using par_t = parallel_policy_shim<::hpx::execution::parallel_executor,
                                   ::hpx::traits::executor_parameters_type_t<::hpx::execution::parallel_executor>>;

} // namespace detail

_CCCL_GLOBAL_CONSTANT detail::par_t par;

} // namespace hpx
} // namespace system

// alias par here
namespace hpx
{

using thrust::system::hpx::par;

} // namespace hpx
THRUST_NAMESPACE_END

namespace hpx::detail
{
template <typename Executor, typename Parameters>
struct is_execution_policy<thrust::system::hpx::detail::parallel_policy_shim<Executor, Parameters>> : std::true_type
{};
} // namespace hpx::detail
