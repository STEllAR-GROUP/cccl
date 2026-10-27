#include <thrust/inner_product.h>

#include "test_executor.h"
#include <unittest/unittest.h>

template <typename ExecutionPolicy>
void TestInnerProduct(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;
  using T      = typename Vector::value_type;

  const Vector v1{1, 2, 3};
  const Vector v2{4, 5, 6};

  // Test default operations (plus, multiplies)
  // (1*4 + 2*5 + 3*6) + 0 = 32
  {
    T result = thrust::inner_product(exec, v1.begin(), v1.end(), v2.begin(), (T) 0);
    ASSERT_EQUAL(result, 32);
  }

  // Test with initial value
  // (1*4 + 2*5 + 3*6) + 10 = 42
  {
    T result = thrust::inner_product(exec, v1.begin(), v1.end(), v2.begin(), (T) 10);
    ASSERT_EQUAL(result, 42);
  }

  // Test custom binary operations (plus, plus)
  // (1+4) + (2+5) + (3+6) + 10 = 5 + 7 + 9 + 10 = 31
  {
    T result =
      thrust::inner_product(exec, v1.begin(), v1.end(), v2.begin(), (T) 10, thrust::plus<T>(), thrust::plus<T>());
    ASSERT_EQUAL(result, 31);
  }

  // Test empty range
  {
    const Vector empty1;
    const Vector empty2;
    T result = thrust::inner_product(exec, empty1.begin(), empty1.end(), empty2.begin(), (T) 10);
    ASSERT_EQUAL(result, 10);
  }
}

void TestInnerProductPar()
{
  TestInnerProduct(thrust::hpx::par);
}
DECLARE_UNITTEST(TestInnerProductPar);

void TestInnerProductParOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestInnerProduct(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestInnerProductParOnSequencedExecutor);

void TestInnerProductParOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestInnerProduct(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestInnerProductParOnTestSyncExecutor);

void TestInnerProductParWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestInnerProduct(thrust::hpx::par.with(acs));
}
DECLARE_UNITTEST(TestInnerProductParWithAutoChunkSize);

void TestInnerProductParUnseq()
{
  TestInnerProduct(thrust::hpx::par_unseq);
}
DECLARE_UNITTEST(TestInnerProductParUnseq);

void TestInnerProductParUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestInnerProduct(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestInnerProductParUnseqOnSequencedExecutor);

void TestInnerProductParUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestInnerProduct(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestInnerProductParUnseqOnTestSyncExecutor);

void TestInnerProductParUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestInnerProduct(thrust::hpx::par_unseq.with(acs));
}
DECLARE_UNITTEST(TestInnerProductParUnseqWithAutoChunkSize);

void TestInnerProductSeq()
{
  TestInnerProduct(thrust::hpx::seq);
}
DECLARE_UNITTEST(TestInnerProductSeq);

void TestInnerProductSeqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestInnerProduct(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestInnerProductSeqOnSequencedExecutor);

void TestInnerProductSeqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestInnerProduct(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestInnerProductSeqOnTestSyncExecutor);

void TestInnerProductSeqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestInnerProduct(thrust::hpx::seq.with(acs));
}
DECLARE_UNITTEST(TestInnerProductSeqWithAutoChunkSize);

void TestInnerProductUnseq()
{
  TestInnerProduct(thrust::hpx::unseq);
}
DECLARE_UNITTEST(TestInnerProductUnseq);

void TestInnerProductUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestInnerProduct(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestInnerProductUnseqOnSequencedExecutor);

void TestInnerProductUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestInnerProduct(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestInnerProductUnseqOnTestSyncExecutor);

void TestInnerProductUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestInnerProduct(thrust::hpx::unseq.with(acs));
}
DECLARE_UNITTEST(TestInnerProductUnseqWithAutoChunkSize);
