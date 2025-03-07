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

/*! \file transform_scan.h
 *  \brief HPX implementation of transform_inclusive_scan/transform_exclusive_scan.
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

#include <hpx/parallel/algorithms/transform_exclusive_scan.hpp>
#include <hpx/parallel/algorithms/transform_inclusive_scan.hpp>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace hpx
{
namespace detail
{

template <typename ExecutionPolicy,
          typename InputIterator,
          typename OutputIterator,
          typename UnaryFunction,
          typename BinaryFunction>
OutputIterator transform_inclusive_scan(
  execution_policy<ExecutionPolicy>&,
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  UnaryFunction unary_op,
  BinaryFunction binary_op)
{
  return ::hpx::transform_inclusive_scan(first, last, result, binary_op, unary_op);
}

template <typename ExecutionPolicy,
          typename InputIterator,
          typename OutputIterator,
          typename UnaryFunction,
          typename T,
          typename BinaryFunction>
OutputIterator transform_inclusive_scan(
  execution_policy<ExecutionPolicy>&,
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  UnaryFunction unary_op,
  T init,
  BinaryFunction binary_op)
{
  return ::hpx::transform_inclusive_scan(first, last, result, binary_op, unary_op, init);
}

template <typename ExecutionPolicy,
          typename InputIterator,
          typename OutputIterator,
          typename UnaryFunction,
          typename T,
          typename AssociativeOperator>
OutputIterator transform_exclusive_scan(
  execution_policy<ExecutionPolicy>&,
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  UnaryFunction unary_op,
  T init,
  AssociativeOperator binary_op)
{
  return ::hpx::transform_exclusive_scan(first, last, result, init, binary_op, unary_op);
}

} // end namespace detail
} // end namespace hpx
} // end namespace system
THRUST_NAMESPACE_END
