//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test numeric_limits

// infinity()

#include <cuda/std/cassert>
#include <cuda/std/cfloat>
#include <cuda/std/limits>

#if _CCCL_COMPILER(MSVC)
#  include <cmath>
#endif // _CCCL_COMPILER(MSVC)

#include "common.h"
#include "test_macros.h"

template <class T>
__host__ __device__ void test(T expected)
{
  assert(float_eq(cuda::std::numeric_limits<T>::infinity(), expected));
  assert(float_eq(cuda::std::numeric_limits<const T>::infinity(), expected));
  assert(float_eq(cuda::std::numeric_limits<volatile T>::infinity(), expected));
  assert(float_eq(cuda::std::numeric_limits<const volatile T>::infinity(), expected));
}

int main(int, char**)
{
  // MSVC has problems producing infinity from 1.0 / 0.0
#if _CCCL_COMPILER(MSVC)
  const double inf = INFINITY;
#else // ^^^ _CCCL_COMPILER(MSVC) ^^^ / vvv !_CCCL_COMPILER(MSVC) vvv
  const double inf = 1.0 / 0.0;
#endif // ^^^ !_CCCL_COMPILER(MSVC) ^^^

  test<bool>(false);
  test<char>(0);
  test<signed char>(0);
  test<unsigned char>(0);
  test<wchar_t>(0);
#if TEST_STD_VER > 2017 && defined(__cpp_char8_t)
  test<char8_t>(0);
#endif
#ifndef _LIBCUDACXX_HAS_NO_UNICODE_CHARS
  test<char16_t>(0);
  test<char32_t>(0);
#endif // _LIBCUDACXX_HAS_NO_UNICODE_CHARS
  test<short>(0);
  test<unsigned short>(0);
  test<int>(0);
  test<unsigned int>(0);
  test<long>(0);
  test<unsigned long>(0);
  test<long long>(0);
  test<unsigned long long>(0);
#if _CCCL_HAS_INT128()
  test<__int128_t>(0);
  test<__uint128_t>(0);
#endif // _CCCL_HAS_INT128()
  test<float>(inf);
  test<double>(inf);
#ifndef _LIBCUDACXX_HAS_NO_LONG_DOUBLE
  test<long double>(inf);
#endif
#if _CCCL_HAS_NVFP16()
  test<__half>(__double2half(inf));
#endif // _CCCL_HAS_NVFP16
#if _CCCL_HAS_NVBF16()
  test<__nv_bfloat16>(__double2bfloat16(inf));
#endif // _CCCL_HAS_NVBF16
#if _CCCL_HAS_NVFP8()
  test<__nv_fp8_e4m3>(__nv_fp8_e4m3{});
  test<__nv_fp8_e5m2>(make_fp8_e5m2(inf));
#  if _CCCL_CUDACC_AT_LEAST(12, 8)
  test<__nv_fp8_e8m0>(__nv_fp8_e8m0{});
#  endif // _CCCL_CUDACC_AT_LEAST(12, 8)
#endif // _CCCL_HAS_NVFP8()
#if _CCCL_HAS_NVFP6()
  test<__nv_fp6_e2m3>(__nv_fp6_e2m3{});
  test<__nv_fp6_e3m2>(__nv_fp6_e3m2{});
#endif // _CCCL_HAS_NVFP6
#if _CCCL_HAS_NVFP4()
  test<__nv_fp4_e2m1>(__nv_fp4_e2m1{});
#endif // _CCCL_HAS_NVFP4

  return 0;
}
