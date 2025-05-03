#include <thrust/reduce.h>

#include <hpx/include/run_as.hpp>
#include <hpx/manage_runtime.hpp>
#include <unittest/unittest.h>

void TestExplicitRuntime()
{
  using Vector = thrust::device_vector<int>;

  Vector v{1, -2, 3};

  ::hpx::manage_runtime runtime;
  ASSERT_EQUAL(runtime.start(0, nullptr), true);

  ASSERT_EQUAL(thrust::reduce(v.begin(), v.end()), 2);

  ::hpx::run_as_hpx_thread([&] {
    // call from an HPX thread
    ASSERT_EQUAL(thrust::reduce(v.begin(), v.end()), 2);
  });

  (void) runtime.stop();
}
DECLARE_UNITTEST(TestExplicitRuntime);

// can't have any more test cases because runtime initialization is global
