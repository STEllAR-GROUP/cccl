#include <thrust/uninitialized_fill.h>

#include "test_executor.h"
#include <unittest/unittest.h>

template <typename ExecutionPolicy>
void TestUninitializedFill(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;

  // Test 1: uninitialized_fill
  {
    Vector v(5, 0);
    thrust::uninitialized_fill(exec, v.begin(), v.end(), 7);
    const Vector ref(5, 7);
    ASSERT_EQUAL(v, ref);
  }

  // Test 2: uninitialized_fill_n
  {
    Vector v(5, 0);
    auto end = thrust::uninitialized_fill_n(exec, v.begin(), 3, 9);
    ASSERT_EQUAL(end - v.begin(), 3);
    ASSERT_EQUAL(v[0], 9);
    ASSERT_EQUAL(v[1], 9);
    ASSERT_EQUAL(v[2], 9);
    ASSERT_EQUAL(v[3], 0);
    ASSERT_EQUAL(v[4], 0);
  }

  // Test 3: empty
  {
    Vector v;
    thrust::uninitialized_fill(exec, v.begin(), v.end(), 42);
    ASSERT_EQUAL(v.size(), 0);
  }
}

void TestUninitializedFillPar()
{
  TestUninitializedFill(thrust::hpx::par);
}
DECLARE_UNITTEST(TestUninitializedFillPar);

void TestUninitializedFillParOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestUninitializedFill(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestUninitializedFillParOnSequencedExecutor);

void TestUninitializedFillParOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestUninitializedFill(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestUninitializedFillParOnTestSyncExecutor);

void TestUninitializedFillParWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestUninitializedFill(thrust::hpx::par.with(acs));
}
DECLARE_UNITTEST(TestUninitializedFillParWithAutoChunkSize);

void TestUninitializedFillParUnseq()
{
  TestUninitializedFill(thrust::hpx::par_unseq);
}
DECLARE_UNITTEST(TestUninitializedFillParUnseq);

void TestUninitializedFillParUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestUninitializedFill(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestUninitializedFillParUnseqOnSequencedExecutor);

void TestUninitializedFillParUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestUninitializedFill(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestUninitializedFillParUnseqOnTestSyncExecutor);

void TestUninitializedFillParUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestUninitializedFill(thrust::hpx::par_unseq.with(acs));
}
DECLARE_UNITTEST(TestUninitializedFillParUnseqWithAutoChunkSize);

void TestUninitializedFillSeq()
{
  TestUninitializedFill(thrust::hpx::seq);
}
DECLARE_UNITTEST(TestUninitializedFillSeq);

void TestUninitializedFillSeqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestUninitializedFill(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestUninitializedFillSeqOnSequencedExecutor);

void TestUninitializedFillSeqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestUninitializedFill(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestUninitializedFillSeqOnTestSyncExecutor);

void TestUninitializedFillSeqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestUninitializedFill(thrust::hpx::seq.with(acs));
}
DECLARE_UNITTEST(TestUninitializedFillSeqWithAutoChunkSize);

void TestUninitializedFillUnseq()
{
  TestUninitializedFill(thrust::hpx::unseq);
}
DECLARE_UNITTEST(TestUninitializedFillUnseq);

void TestUninitializedFillUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestUninitializedFill(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestUninitializedFillUnseqOnSequencedExecutor);

void TestUninitializedFillUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestUninitializedFill(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestUninitializedFillUnseqOnTestSyncExecutor);

void TestUninitializedFillUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestUninitializedFill(thrust::hpx::unseq.with(acs));
}
DECLARE_UNITTEST(TestUninitializedFillUnseqWithAutoChunkSize);
