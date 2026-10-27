#include <thrust/unique.h>

#include "test_executor.h"
#include <unittest/unittest.h>

template <typename ExecutionPolicy>
void TestUnique(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;

  // Test 1: unique with default predicate
  {
    Vector data{11, 11, 12, 20, 21, 21, 31, 31, 37};
    auto new_last = thrust::unique(exec, data.begin(), data.end());
    ASSERT_EQUAL(new_last - data.begin(), 6);
    ASSERT_EQUAL(data[0], 11);
    ASSERT_EQUAL(data[1], 12);
    ASSERT_EQUAL(data[2], 20);
    ASSERT_EQUAL(data[3], 21);
    ASSERT_EQUAL(data[4], 31);
    ASSERT_EQUAL(data[5], 37);
  }

  // Test 2: unique_copy
  {
    const Vector data{11, 11, 12, 20, 21, 21, 31, 31, 37};
    Vector out(6, 0);
    auto out_end = thrust::unique_copy(exec, data.begin(), data.end(), out.begin());
    ASSERT_EQUAL(out_end - out.begin(), 6);
    ASSERT_EQUAL(out[0], 11);
    ASSERT_EQUAL(out[1], 12);
    ASSERT_EQUAL(out[2], 20);
    ASSERT_EQUAL(out[3], 21);
    ASSERT_EQUAL(out[4], 31);
    ASSERT_EQUAL(out[5], 37);
  }

  // Test 3: unique_count
  {
    const Vector data{11, 11, 12, 20, 21, 21, 31, 31, 37};
    ASSERT_EQUAL(thrust::unique_count(exec, data.begin(), data.end()), 6);
  }

  // Test 4: empty range
  {
    Vector empty;
    auto new_last = thrust::unique(exec, empty.begin(), empty.end());
    ASSERT_EQUAL(new_last - empty.begin(), 0);
  }
}

void TestUniquePar()
{
  TestUnique(thrust::hpx::par);
}
DECLARE_UNITTEST(TestUniquePar);

void TestUniqueParOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestUnique(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestUniqueParOnSequencedExecutor);

void TestUniqueParOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestUnique(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestUniqueParOnTestSyncExecutor);

void TestUniqueParWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestUnique(thrust::hpx::par.with(acs));
}
DECLARE_UNITTEST(TestUniqueParWithAutoChunkSize);

void TestUniqueParUnseq()
{
  TestUnique(thrust::hpx::par_unseq);
}
DECLARE_UNITTEST(TestUniqueParUnseq);

void TestUniqueParUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestUnique(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestUniqueParUnseqOnSequencedExecutor);

void TestUniqueParUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestUnique(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestUniqueParUnseqOnTestSyncExecutor);

void TestUniqueParUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestUnique(thrust::hpx::par_unseq.with(acs));
}
DECLARE_UNITTEST(TestUniqueParUnseqWithAutoChunkSize);

void TestUniqueSeq()
{
  TestUnique(thrust::hpx::seq);
}
DECLARE_UNITTEST(TestUniqueSeq);

void TestUniqueSeqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestUnique(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestUniqueSeqOnSequencedExecutor);

void TestUniqueSeqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestUnique(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestUniqueSeqOnTestSyncExecutor);

void TestUniqueSeqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestUnique(thrust::hpx::seq.with(acs));
}
DECLARE_UNITTEST(TestUniqueSeqWithAutoChunkSize);

void TestUniqueUnseq()
{
  TestUnique(thrust::hpx::unseq);
}
DECLARE_UNITTEST(TestUniqueUnseq);

void TestUniqueUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestUnique(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestUniqueUnseqOnSequencedExecutor);

void TestUniqueUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestUnique(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestUniqueUnseqOnTestSyncExecutor);

void TestUniqueUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestUnique(thrust::hpx::unseq.with(acs));
}
DECLARE_UNITTEST(TestUniqueUnseqWithAutoChunkSize);
