//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/optional>

// constexpr const T& optional<T>::value() const &;

#include <cuda/std/cassert>
#include <cuda/std/optional>
#include <cuda/std/type_traits>

#include "test_macros.h"

using cuda::std::in_place;
using cuda::std::in_place_t;
using cuda::std::optional;
#ifndef TEST_HAS_NO_EXCEPTIONS
using cuda::std::bad_optional_access;
#endif

struct X
{
  X()         = default;
  X(const X&) = delete;
  __host__ __device__ constexpr int test() const&
  {
    return 3;
  }
  __host__ __device__ int test() &
  {
    return 4;
  }
  __host__ __device__ constexpr int test() const&&
  {
    return 5;
  }
  __host__ __device__ int test() &&
  {
    return 6;
  }
};

#ifndef TEST_HAS_NO_EXCEPTIONS
void test_exceptions()
{
  const optional<X> opt;
  try
  {
    (void) opt.value();
    assert(false);
  }
  catch (const bad_optional_access&)
  {}
}
#endif // !TEST_HAS_NO_EXCEPTIONS

int main(int, char**)
{
  {
    const optional<X> opt;
    unused(opt);
    ASSERT_NOT_NOEXCEPT(opt.value());
    ASSERT_SAME_TYPE(decltype(opt.value()), X const&);
  }
#if !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))
  {
    constexpr optional<X> opt(in_place);
    static_assert(opt.value().test() == 3, "");
  }
#endif // !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))
  {
    const optional<X> opt(in_place);
    assert(opt.value().test() == 3);
  }
#ifndef TEST_HAS_NO_EXCEPTIONS
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // !TEST_HAS_NO_EXCEPTIONS

  return 0;
}
