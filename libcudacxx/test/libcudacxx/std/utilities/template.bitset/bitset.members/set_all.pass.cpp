//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// bitset<N>& set(); // constexpr since C++23

#include <cuda/std/bitset>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>

#include "test_macros.h"

_CCCL_NV_DIAG_SUPPRESS(186)

template <cuda::std::size_t N>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test_set_all()
{
  cuda::std::bitset<N> v;
  v.set();
  for (cuda::std::size_t i = 0; i < v.size(); ++i)
  {
    {
      assert(v[i]);
    }
  }
}

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
{
  test_set_all<0>();
  test_set_all<1>();
  test_set_all<31>();
  test_set_all<32>();
  test_set_all<33>();
  test_set_all<63>();
  test_set_all<64>();
  test_set_all<65>();
  test_set_all<1000>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
