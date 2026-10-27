#include <thrust/mismatch.h>

#include "test_executor.h"
#include <unittest/unittest.h>

template <typename ExecutionPolicy>
void TestMismatch(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;

  // Test 1: mismatch exists
  {
    const Vector a{1, 2, 3, 4, 5};
    const Vector b{1, 2, 9, 4, 5};

    auto result = thrust::mismatch(exec, a.begin(), a.end(), b.begin());
    ASSERT_EQUAL(result.first - a.begin(), 2);
    ASSERT_EQUAL(result.second - b.begin(), 2);
    ASSERT_EQUAL(*result.first, 3);
    ASSERT_EQUAL(*result.second, 9);
  }

  // Test 2: no mismatch
  {
    const Vector a{1, 2, 3, 4, 5};
    const Vector b{1, 2, 3, 4, 5};

    auto result = thrust::mismatch(exec, a.begin(), a.end(), b.begin());
    ASSERT_EQUAL(result.first - a.begin(), 5);
    ASSERT_EQUAL(result.second - b.begin(), 5);
  }

  // Test 3: empty range
  {
    const Vector a;
    const Vector b;

    auto result = thrust::mismatch(exec, a.begin(), a.end(), b.begin());
    ASSERT_EQUAL(result.first - a.begin(), 0);
    ASSERT_EQUAL(result.second - b.begin(), 0);
  }
}

void TestMismatchPar()
{
  TestMismatch(thrust::hpx::par);
}
DECLARE_UNITTEST(TestMismatchPar);

void TestMismatchParOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestMismatch(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestMismatchParOnSequencedExecutor);

void TestMismatchParOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestMismatch(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestMismatchParOnTestSyncExecutor);

void TestMismatchParWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestMismatch(thrust::hpx::par.with(acs));
}
DECLARE_UNITTEST(TestMismatchParWithAutoChunkSize);

void TestMismatchParUnseq()
{
  TestMismatch(thrust::hpx::par_unseq);
}
DECLARE_UNITTEST(TestMismatchParUnseq);

void TestMismatchParUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestMismatch(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestMismatchParUnseqOnSequencedExecutor);

void TestMismatchParUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestMismatch(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestMismatchParUnseqOnTestSyncExecutor);

void TestMismatchParUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestMismatch(thrust::hpx::par_unseq.with(acs));
}
DECLARE_UNITTEST(TestMismatchParUnseqWithAutoChunkSize);

void TestMismatchSeq()
{
  TestMismatch(thrust::hpx::seq);
}
DECLARE_UNITTEST(TestMismatchSeq);

void TestMismatchSeqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestMismatch(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestMismatchSeqOnSequencedExecutor);

void TestMismatchSeqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestMismatch(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestMismatchSeqOnTestSyncExecutor);

void TestMismatchSeqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestMismatch(thrust::hpx::seq.with(acs));
}
DECLARE_UNITTEST(TestMismatchSeqWithAutoChunkSize);

void TestMismatchUnseq()
{
  TestMismatch(thrust::hpx::unseq);
}
DECLARE_UNITTEST(TestMismatchUnseq);

void TestMismatchUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestMismatch(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestMismatchUnseqOnSequencedExecutor);

void TestMismatchUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestMismatch(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestMismatchUnseqOnTestSyncExecutor);

void TestMismatchUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestMismatch(thrust::hpx::unseq.with(acs));
}
DECLARE_UNITTEST(TestMismatchUnseqWithAutoChunkSize);
