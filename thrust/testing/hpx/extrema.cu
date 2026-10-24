#include <thrust/extrema.h>

#include "test_executor.h"
#include <unittest/unittest.h>

template <typename ExecutionPolicy>
void TestExtrema(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;

  // Test min_element & max_element
  {
    const Vector data{5, 2, 8, 1, 9, 3};

    auto min_it = thrust::min_element(exec, data.begin(), data.end());
    ASSERT_EQUAL(min_it - data.begin(), 3);
    ASSERT_EQUAL(*min_it, 1);

    auto max_it = thrust::max_element(exec, data.begin(), data.end());
    ASSERT_EQUAL(max_it - data.begin(), 4);
    ASSERT_EQUAL(*max_it, 9);
  }

  // Test minmax_element
  {
    const Vector data{5, 2, 8, 1, 9, 3};

    auto result = thrust::minmax_element(exec, data.begin(), data.end());
    ASSERT_EQUAL(result.first - data.begin(), 3);
    ASSERT_EQUAL(*result.first, 1);
    ASSERT_EQUAL(result.second - data.begin(), 4);
    ASSERT_EQUAL(*result.second, 9);
  }

  // Test empty range
  {
    const Vector empty;
    auto min_it = thrust::min_element(exec, empty.begin(), empty.end());
    ASSERT_EQUAL(min_it - empty.begin(), 0);

    auto max_it = thrust::max_element(exec, empty.begin(), empty.end());
    ASSERT_EQUAL(max_it - empty.begin(), 0);

    auto result = thrust::minmax_element(exec, empty.begin(), empty.end());
    ASSERT_EQUAL(result.first - empty.begin(), 0);
    ASSERT_EQUAL(result.second - empty.begin(), 0);
  }
}

void TestExtremaPar()
{
  TestExtrema(thrust::hpx::par);
}
DECLARE_UNITTEST(TestExtremaPar);

void TestExtremaParOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestExtrema(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestExtremaParOnSequencedExecutor);

void TestExtremaParOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestExtrema(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestExtremaParOnTestSyncExecutor);

void TestExtremaParWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestExtrema(thrust::hpx::par.with(acs));
}
DECLARE_UNITTEST(TestExtremaParWithAutoChunkSize);

void TestExtremaParUnseq()
{
  TestExtrema(thrust::hpx::par_unseq);
}
DECLARE_UNITTEST(TestExtremaParUnseq);

void TestExtremaParUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestExtrema(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestExtremaParUnseqOnSequencedExecutor);

void TestExtremaParUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestExtrema(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestExtremaParUnseqOnTestSyncExecutor);

void TestExtremaParUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestExtrema(thrust::hpx::par_unseq.with(acs));
}
DECLARE_UNITTEST(TestExtremaParUnseqWithAutoChunkSize);

void TestExtremaSeq()
{
  TestExtrema(thrust::hpx::seq);
}
DECLARE_UNITTEST(TestExtremaSeq);

void TestExtremaSeqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestExtrema(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestExtremaSeqOnSequencedExecutor);

void TestExtremaSeqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestExtrema(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestExtremaSeqOnTestSyncExecutor);

void TestExtremaSeqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestExtrema(thrust::hpx::seq.with(acs));
}
DECLARE_UNITTEST(TestExtremaSeqWithAutoChunkSize);

void TestExtremaUnseq()
{
  TestExtrema(thrust::hpx::unseq);
}
DECLARE_UNITTEST(TestExtremaUnseq);

void TestExtremaUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestExtrema(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestExtremaUnseqOnSequencedExecutor);

void TestExtremaUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestExtrema(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestExtremaUnseqOnTestSyncExecutor);

void TestExtremaUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestExtrema(thrust::hpx::unseq.with(acs));
}
DECLARE_UNITTEST(TestExtremaUnseqWithAutoChunkSize);
