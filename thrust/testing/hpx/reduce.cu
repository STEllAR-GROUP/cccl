#include <thrust/reduce.h>

#include <unittest/unittest.h>

template <typename ExecutionPolicy>
void TestReduce(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;
  using T      = typename Vector::value_type;

  Vector v{1, -2, 3};

  // no initializer
  ASSERT_EQUAL(thrust::reduce(exec, v.begin(), v.end()), 2);

  // with initializer
  ASSERT_EQUAL(thrust::reduce(exec, v.begin(), v.end(), (T) 10), 12);
}

void TestReducePar()
{
  TestReduce(thrust::hpx::par);
}
DECLARE_UNITTEST(TestReducePar);

void TestReduceParOnParallelExecutor()
{
  hpx::execution::parallel_executor exec{};

  TestReduce(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestReduceParOnParallelExecutor);

void TestReduceParOnForkJoinExecutor()
{
  hpx::execution::experimental::fork_join_executor exec{};

  TestReduce(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestReduceParOnForkJoinExecutor);

void TestReduceParWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestReduce(thrust::hpx::par.with(acs));
}
DECLARE_UNITTEST(TestReduceParWithAutoChunkSize);
