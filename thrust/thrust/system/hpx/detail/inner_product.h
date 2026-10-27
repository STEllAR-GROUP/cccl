// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*! \file inner_product.h
 *   \brief HPX implementation of inner_product.
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

#include <thrust/system/detail/generic/inner_product.h>
#include <thrust/system/hpx/detail/contiguous_iterator.h>
#include <thrust/system/hpx/detail/execution_policy.h>
#include <thrust/system/hpx/detail/function.h>

#include <cuda/std/__functional/operations.h>

#include <hpx/parallel/algorithms/inner_product.hpp>

THRUST_NAMESPACE_BEGIN
namespace system::hpx::detail
{
template <typename ExecutionPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputType,
          typename BinaryFunction1,
          typename BinaryFunction2>
OutputType inner_product(
  execution_policy<ExecutionPolicy>& exec [[maybe_unused]],
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  OutputType init,
  BinaryFunction1 binary_op1,
  BinaryFunction2 binary_op2)
{
  if (first1 == last1)
  {
    return init;
  }

  hpx_wrapped_function<BinaryFunction1> wrapped_op1{binary_op1};
  hpx_wrapped_function<BinaryFunction2> wrapped_op2{binary_op2};

  if constexpr (::hpx::traits::is_forward_iterator_v<InputIterator1>
                && ::hpx::traits::is_forward_iterator_v<InputIterator2>)
  {
    return ::hpx::inner_product(
      hpx::detail::to_hpx_execution_policy(exec),
      ::thrust::try_unwrap_contiguous_iterator(first1),
      ::thrust::try_unwrap_contiguous_iterator(last1),
      ::thrust::try_unwrap_contiguous_iterator(first2),
      init,
      wrapped_op1,
      wrapped_op2);
  }
  else
  {
    return ::hpx::inner_product(first1, last1, first2, init, wrapped_op1, wrapped_op2);
  }
}

template <typename ExecutionPolicy, typename InputIterator1, typename InputIterator2, typename OutputType>
OutputType inner_product(
  execution_policy<ExecutionPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  OutputType init)
{
  return inner_product(
    exec, first1, last1, first2, init, ::cuda::std::plus<OutputType>(), ::cuda::std::multiplies<OutputType>());
}
} // end namespace system::hpx::detail
THRUST_NAMESPACE_END
