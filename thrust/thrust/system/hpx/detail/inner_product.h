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

/*! \file inner_product.h
 *  \brief HPX implementation of inner_product.
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

#include <hpx/parallel/algorithms/transform_reduce.hpp>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace hpx
{
namespace detail
{

template <typename DerivedPolicy, typename InputIterator1, typename InputIterator2, typename OutputType>
OutputType inner_product(
  execution_policy<DerivedPolicy>&, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, OutputType init)
{
  return ::hpx::transform_reduce(first1, last1, first2, init);
}

template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputType,
          typename BinaryFunction1,
          typename BinaryFunction2>
OutputType inner_product(
  execution_policy<DerivedPolicy>&,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  OutputType init,
  BinaryFunction1 binary_op1,
  BinaryFunction2 binary_op2)
{
  return ::hpx::transform_reduce(first1, last1, first2, init, binary_op1, binary_op2);
}

} // end namespace detail
} // end namespace hpx
} // end namespace system
THRUST_NAMESPACE_END
