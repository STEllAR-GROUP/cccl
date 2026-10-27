// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*! \file reduce_by_key.h
 *   \brief HPX implementation of reduce_by_key.
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
#include <thrust/system/detail/generic/reduce_by_key.h>
#include <thrust/system/hpx/detail/contiguous_iterator.h>
#include <thrust/system/hpx/detail/execution_policy.h>
#include <thrust/system/hpx/detail/function.h>

#include <cuda/std/__functional/operations.h>
#include <cuda/std/__utility/pair.h>

#include <hpx/parallel/algorithms/reduce_by_key.hpp>

THRUST_NAMESPACE_BEGIN
namespace system::hpx::detail
{
template <typename ExecutionPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2,
          typename BinaryPredicate,
          typename BinaryFunction>
::cuda::std::pair<OutputIterator1, OutputIterator2> reduce_by_key(
  execution_policy<ExecutionPolicy>& exec,
  InputIterator1 keys_first,
  InputIterator1 keys_last,
  InputIterator2 values_first,
  OutputIterator1 keys_output,
  OutputIterator2 values_output,
  BinaryPredicate binary_pred,
  BinaryFunction binary_op)
{
  auto unwrapped_keys_first   = ::thrust::try_unwrap_contiguous_iterator(keys_first);
  auto unwrapped_keys_last    = ::thrust::try_unwrap_contiguous_iterator(keys_last);
  auto unwrapped_values_first = ::thrust::try_unwrap_contiguous_iterator(values_first);
  auto unwrapped_keys_out     = ::thrust::try_unwrap_contiguous_iterator(keys_output);
  auto unwrapped_values_out   = ::thrust::try_unwrap_contiguous_iterator(values_output);

  using unwrapped_key_it_t  = decltype(unwrapped_keys_first);
  using unwrapped_val_it_t  = decltype(unwrapped_values_first);
  using unwrapped_kout_it_t = decltype(unwrapped_keys_out);
  using unwrapped_vout_it_t = decltype(unwrapped_values_out);

  hpx_wrapped_function<BinaryPredicate> wrapped_pred{binary_pred};
  hpx_wrapped_function<BinaryFunction> wrapped_op{binary_op};

  if constexpr (::hpx::traits::is_random_access_iterator_v<unwrapped_key_it_t>
                && ::hpx::traits::is_random_access_iterator_v<unwrapped_val_it_t>
                && ::hpx::traits::is_forward_iterator_v<unwrapped_kout_it_t>
                && ::hpx::traits::is_forward_iterator_v<unwrapped_vout_it_t>)
  {
    auto res = ::hpx::experimental::reduce_by_key(
      hpx::detail::to_hpx_execution_policy(exec),
      unwrapped_keys_first,
      unwrapped_keys_last,
      unwrapped_values_first,
      unwrapped_keys_out,
      unwrapped_values_out,
      wrapped_pred,
      wrapped_op);

    return ::cuda::std::make_pair(detail::rewrap_contiguous_iterator(res.in, keys_output),
                                  detail::rewrap_contiguous_iterator(res.out, values_output));
  }
  else
  {
    return thrust::system::detail::generic::reduce_by_key(
      exec, keys_first, keys_last, values_first, keys_output, values_output, wrapped_pred, wrapped_op);
  }
}

template <typename ExecutionPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2>
::cuda::std::pair<OutputIterator1, OutputIterator2> reduce_by_key(
  execution_policy<ExecutionPolicy>& exec,
  InputIterator1 keys_first,
  InputIterator1 keys_last,
  InputIterator2 values_first,
  OutputIterator1 keys_output,
  OutputIterator2 values_output)
{
  using key_type = thrust::detail::it_value_t<InputIterator1>;
  return reduce_by_key(
    exec, keys_first, keys_last, values_first, keys_output, values_output, ::cuda::std::equal_to<key_type>());
}

template <typename ExecutionPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2,
          typename BinaryPredicate>
::cuda::std::pair<OutputIterator1, OutputIterator2> reduce_by_key(
  execution_policy<ExecutionPolicy>& exec,
  InputIterator1 keys_first,
  InputIterator1 keys_last,
  InputIterator2 values_first,
  OutputIterator1 keys_output,
  OutputIterator2 values_output,
  BinaryPredicate binary_pred)
{
  using value_type = thrust::detail::it_value_t<InputIterator2>;
  return reduce_by_key(
    exec, keys_first, keys_last, values_first, keys_output, values_output, binary_pred, ::cuda::std::plus<value_type>());
}
} // namespace system::hpx::detail
THRUST_NAMESPACE_END
