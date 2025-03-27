#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/reduce.h>

#include <limits>

#include <unittest/unittest.h>


template <class Vector>
void TestReduceSimple()
{
  using T = typename Vector::value_type;

  Vector v{1, -2, 3};

  // no initializer
  ASSERT_EQUAL(thrust::reduce( v.begin(), v.end()), 2);

  // with initializer
  ASSERT_EQUAL(thrust::reduce(thrust::hpx::par, v.begin(), v.end(), (T) 10, thrust::plus<T>()), 12);
}
DECLARE_VECTOR_UNITTEST(TestReduceSimple);
