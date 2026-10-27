// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*! \file generate.h
 *   \brief HPX implementation of generate algorithms.
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

#include <thrust/system/detail/generic/generate.h>
#include <thrust/system/hpx/detail/contiguous_iterator.h>
#include <thrust/system/hpx/detail/execution_policy.h>
#include <thrust/system/hpx/detail/function.h>

#include <hpx/parallel/algorithms/generate.hpp>

THRUST_NAMESPACE_BEGIN
namespace system::hpx::detail
{
template <typename ExecutionPolicy, typename ForwardIterator, typename Generator>
void generate(
  execution_policy<ExecutionPolicy>& exec [[maybe_unused]], ForwardIterator first, ForwardIterator last, Generator gen)
{
  hpx_wrapped_function<Generator> wrapped_gen{gen};

  if constexpr (::hpx::traits::is_forward_iterator_v<ForwardIterator>)
  {
    ::hpx::generate(hpx::detail::to_hpx_execution_policy(exec),
                    ::thrust::try_unwrap_contiguous_iterator(first),
                    ::thrust::try_unwrap_contiguous_iterator(last),
                    wrapped_gen);
  }
  else
  {
    ::hpx::generate(first, last, wrapped_gen);
  }
}

template <typename ExecutionPolicy, typename OutputIterator, typename Size, typename Generator>
OutputIterator
generate_n(execution_policy<ExecutionPolicy>& exec [[maybe_unused]], OutputIterator first, Size n, Generator gen)
{
  hpx_wrapped_function<Generator> wrapped_gen{gen};

  if constexpr (::hpx::traits::is_forward_iterator_v<OutputIterator>)
  {
    auto res = ::hpx::generate_n(
      hpx::detail::to_hpx_execution_policy(exec), ::thrust::try_unwrap_contiguous_iterator(first), n, wrapped_gen);
    return detail::rewrap_contiguous_iterator(res, first);
  }
  else
  {
    return ::hpx::generate_n(first, n, wrapped_gen);
  }
}
} // end namespace system::hpx::detail
THRUST_NAMESPACE_END
