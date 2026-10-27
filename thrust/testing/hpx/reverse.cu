#include <thrust/reverse.h>

#include "test_executor.h"
#include <unittest/unittest.h>

template <typename ExecutionPolicy>
void TestReverse(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;

  // Test 1: reverse in-place
  {
    Vector data{1, 2, 3, 4, 5};
    thrust::reverse(exec, data.begin(), data.end());
    ASSERT_EQUAL(data[0], 5);
    ASSERT_EQUAL(data[1], 4);
    ASSERT_EQUAL(data[2], 3);
    ASSERT_EQUAL(data[3], 2);
    ASSERT_EQUAL(data[4], 1);
  }

  // Test 2: reverse_copy
  {
    const Vector data{1, 2, 3, 4, 5};
    Vector out(5, 0);
    auto out_end = thrust::reverse_copy(exec, data.begin(), data.end(), out.begin());
    ASSERT_EQUAL(out_end - out.begin(), 5);
    ASSERT_EQUAL(out[0], 5);
    ASSERT_EQUAL(out[1], 4);
    ASSERT_EQUAL(out[2], 3);
    ASSERT_EQUAL(out[3], 2);
    ASSERT_EQUAL(out[4], 1);
  }

  // Test 3: empty range
  {
    Vector empty;
    thrust::reverse(exec, empty.begin(), empty.end());
    ASSERT_EQUAL(empty.size(), 0);

    Vector out;
    auto out_end = thrust::reverse_copy(exec, empty.begin(), empty.end(), out.begin());
    ASSERT_EQUAL(out_end - out.begin(), 0);
  }
}

void TestReversePar()
{
  TestReverse(thrust::hpx::par);
}
DECLARE_UNITTEST(TestReversePar);

void TestReverseParOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestReverse(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestReverseParOnSequencedExecutor);

void TestReverseParOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestReverse(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestReverseParOnTestSyncExecutor);

void TestReverseParWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestReverse(thrust::hpx::par.with(acs));
}
DECLARE_UNITTEST(TestReverseParWithAutoChunkSize);

void TestReverseParUnseq()
{
  TestReverse(thrust::hpx::par_unseq);
}
DECLARE_UNITTEST(TestReverseParUnseq);

void TestReverseParUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestReverse(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestReverseParUnseqOnSequencedExecutor);

void TestReverseParUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestReverse(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestReverseParUnseqOnTestSyncExecutor);

void TestReverseParUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestReverse(thrust::hpx::par_unseq.with(acs));
}
DECLARE_UNITTEST(TestReverseParUnseqWithAutoChunkSize);

void TestReverseSeq()
{
  TestReverse(thrust::hpx::seq);
}
DECLARE_UNITTEST(TestReverseSeq);

void TestReverseSeqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestReverse(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestReverseSeqOnSequencedExecutor);

void TestReverseSeqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestReverse(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestReverseSeqOnTestSyncExecutor);

void TestReverseSeqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestReverse(thrust::hpx::seq.with(acs));
}
DECLARE_UNITTEST(TestReverseSeqWithAutoChunkSize);

void TestReverseUnseq()
{
  TestReverse(thrust::hpx::unseq);
}
DECLARE_UNITTEST(TestReverseUnseq);

void TestReverseUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestReverse(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestReverseUnseqOnSequencedExecutor);

void TestReverseUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestReverse(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestReverseUnseqOnTestSyncExecutor);

void TestReverseUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestReverse(thrust::hpx::unseq.with(acs));
}
DECLARE_UNITTEST(TestReverseUnseqWithAutoChunkSize);
