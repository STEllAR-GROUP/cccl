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

/*! \file generate.h
 *  \brief HPX implementation of generate/generate_n.
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

#include <hpx/parallel/algorithms/generate.hpp>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace hpx
{
namespace detail
{

template <typename ExecutionPolicy, typename ForwardIterator, typename Generator>
void generate(execution_policy<ExecutionPolicy>&, ForwardIterator first, ForwardIterator last, Generator gen)
{
  return (void)::hpx::generate(first, last, gen);
}

template <typename ExecutionPolicy, typename OutputIterator, typename Size, typename Generator>
OutputIterator generate_n(execution_policy<ExecutionPolicy>&, OutputIterator first, Size n, Generator gen)
{
  return ::hpx::generate_n(first, n, gen);
}

} // end namespace detail
} // end namespace hpx
} // end namespace system
THRUST_NAMESPACE_END
