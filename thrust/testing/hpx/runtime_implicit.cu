#include <thrust/reduce.h>

#include <unittest/unittest.h>

void TestImplicitRuntime()
{
  using Vector = thrust::device_vector<int>;

  Vector v{1, -2, 3};
  ASSERT_EQUAL(thrust::reduce(v.begin(), v.end()), 2);
}
DECLARE_UNITTEST(TestImplicitRuntime);

// can't have any more test cases because runtime initialization is global
