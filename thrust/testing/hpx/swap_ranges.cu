#include <thrust/swap.h>

#include "test_executor.h"
#include <unittest/unittest.h>

template <typename ExecutionPolicy>
void TestSwapRanges(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;

  // Test 1: non-empty swap_ranges
  {
    Vector v1{1, 2, 3};
    Vector v2{4, 5, 6};

    auto res = thrust::swap_ranges(exec, v1.begin(), v1.end(), v2.begin());
    ASSERT_EQUAL(res - v2.begin(), 3);

    ASSERT_EQUAL(v1[0], 4);
    ASSERT_EQUAL(v1[1], 5);
    ASSERT_EQUAL(v1[2], 6);

    ASSERT_EQUAL(v2[0], 1);
    ASSERT_EQUAL(v2[1], 2);
    ASSERT_EQUAL(v2[2], 3);
  }

  // Test 2: empty swap_ranges
  {
    Vector empty1;
    Vector empty2;

    auto res = thrust::swap_ranges(exec, empty1.begin(), empty1.end(), empty2.begin());
    ASSERT_EQUAL(res - empty2.begin(), 0);
  }
}

void TestSwapRangesPar()
{
  TestSwapRanges(thrust::hpx::par);
}
DECLARE_UNITTEST(TestSwapRangesPar);

void TestSwapRangesParOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestSwapRanges(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestSwapRangesParOnSequencedExecutor);

void TestSwapRangesParOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestSwapRanges(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestSwapRangesParOnTestSyncExecutor);

void TestSwapRangesParWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestSwapRanges(thrust::hpx::par.with(acs));
}
DECLARE_UNITTEST(TestSwapRangesParWithAutoChunkSize);

void TestSwapRangesParUnseq()
{
  TestSwapRanges(thrust::hpx::par_unseq);
}
DECLARE_UNITTEST(TestSwapRangesParUnseq);

void TestSwapRangesParUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestSwapRanges(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestSwapRangesParUnseqOnSequencedExecutor);

void TestSwapRangesParUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestSwapRanges(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestSwapRangesParUnseqOnTestSyncExecutor);

void TestSwapRangesParUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestSwapRanges(thrust::hpx::par_unseq.with(acs));
}
DECLARE_UNITTEST(TestSwapRangesParUnseqWithAutoChunkSize);

void TestSwapRangesSeq()
{
  TestSwapRanges(thrust::hpx::seq);
}
DECLARE_UNITTEST(TestSwapRangesSeq);

void TestSwapRangesSeqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestSwapRanges(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestSwapRangesSeqOnSequencedExecutor);

void TestSwapRangesSeqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestSwapRanges(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestSwapRangesSeqOnTestSyncExecutor);

void TestSwapRangesSeqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestSwapRanges(thrust::hpx::seq.with(acs));
}
DECLARE_UNITTEST(TestSwapRangesSeqWithAutoChunkSize);

void TestSwapRangesUnseq()
{
  TestSwapRanges(thrust::hpx::unseq);
}
DECLARE_UNITTEST(TestSwapRangesUnseq);

void TestSwapRangesUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestSwapRanges(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestSwapRangesUnseqOnSequencedExecutor);

void TestSwapRangesUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestSwapRanges(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestSwapRangesUnseqOnTestSyncExecutor);

void TestSwapRangesUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestSwapRanges(thrust::hpx::unseq.with(acs));
}
DECLARE_UNITTEST(TestSwapRangesUnseqWithAutoChunkSize);
