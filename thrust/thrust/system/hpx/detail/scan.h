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

/*! \file scan.h
 *  \brief HPX implementation of inclusive_scan/exclusive_scan.
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

#include <hpx/parallel/algorithms/exclusive_scan.hpp>
#include <hpx/parallel/algorithms/inclusive_scan.hpp>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace hpx
{
namespace detail
{

template <typename ExecutionPolicy, typename InputIterator, typename OutputIterator>
OutputIterator
inclusive_scan(execution_policy<ExecutionPolicy>&, InputIterator first, InputIterator last, OutputIterator result)
{
  return ::hpx::inclusive_scan(first, last, result);
}

template <typename ExecutionPolicy, typename InputIterator, typename OutputIterator, typename BinaryFunction>
OutputIterator inclusive_scan(
  execution_policy<ExecutionPolicy>&,
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  BinaryFunction binary_op)
{
  return ::hpx::inclusive_scan(first, last, result, binary_op);
}

template <typename ExecutionPolicy, typename InputIterator, typename OutputIterator>
OutputIterator
exclusive_scan(execution_policy<ExecutionPolicy>&, InputIterator first, InputIterator last, OutputIterator result)
{
  return ::hpx::exclusive_scan(first, last, result);
}

template <typename ExecutionPolicy, typename InputIterator, typename OutputIterator, typename T>
OutputIterator exclusive_scan(
  execution_policy<ExecutionPolicy>&, InputIterator first, InputIterator last, OutputIterator result, T init)
{
  return ::hpx::exclusive_scan(first, last, result, init);
}

template <typename ExecutionPolicy, typename InputIterator, typename OutputIterator, typename T, typename BinaryFunction>
OutputIterator exclusive_scan(
  execution_policy<ExecutionPolicy>&,
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  T init,
  BinaryFunction binary_op)
{
  return ::hpx::exclusive_scan(first, last, result, init, binary_op);
}

} // end namespace detail
} // end namespace hpx
} // end namespace system
THRUST_NAMESPACE_END
