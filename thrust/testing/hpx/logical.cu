#include <thrust/logical.h>

#include "test_executor.h"
#include <unittest/unittest.h>

struct greater_than_zero
{
  __host__ __device__ bool operator()(int x) const
  {
    return x > 0;
  }
};

template <typename ExecutionPolicy>
void TestLogical(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;

  // Test all_of
  {
    const Vector all_pos{1, 2, 3, 4};
    const Vector some_pos{1, -2, 3, -4};

    ASSERT_EQUAL(thrust::all_of(exec, all_pos.begin(), all_pos.end(), greater_than_zero{}), true);
    ASSERT_EQUAL(thrust::all_of(exec, some_pos.begin(), some_pos.end(), greater_than_zero{}), false);
  }

  // Test any_of
  {
    const Vector some_pos{1, -2, 3, -4};
    const Vector none_pos{-1, -2, -3, -4};

    ASSERT_EQUAL(thrust::any_of(exec, some_pos.begin(), some_pos.end(), greater_than_zero{}), true);
    ASSERT_EQUAL(thrust::any_of(exec, none_pos.begin(), none_pos.end(), greater_than_zero{}), false);
  }

  // Test none_of
  {
    const Vector none_pos{-1, -2, -3, -4};
    const Vector some_pos{1, -2, 3, -4};

    ASSERT_EQUAL(thrust::none_of(exec, none_pos.begin(), none_pos.end(), greater_than_zero{}), true);
    ASSERT_EQUAL(thrust::none_of(exec, some_pos.begin(), some_pos.end(), greater_than_zero{}), false);
  }

  // Test empty range
  {
    const Vector empty;
    ASSERT_EQUAL(thrust::all_of(exec, empty.begin(), empty.end(), greater_than_zero{}), true);
    ASSERT_EQUAL(thrust::any_of(exec, empty.begin(), empty.end(), greater_than_zero{}), false);
    ASSERT_EQUAL(thrust::none_of(exec, empty.begin(), empty.end(), greater_than_zero{}), true);
  }
}

void TestLogicalPar()
{
  TestLogical(thrust::hpx::par);
}
DECLARE_UNITTEST(TestLogicalPar);

void TestLogicalParOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestLogical(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestLogicalParOnSequencedExecutor);

void TestLogicalParOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestLogical(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestLogicalParOnTestSyncExecutor);

void TestLogicalParWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestLogical(thrust::hpx::par.with(acs));
}
DECLARE_UNITTEST(TestLogicalParWithAutoChunkSize);

void TestLogicalParUnseq()
{
  TestLogical(thrust::hpx::par_unseq);
}
DECLARE_UNITTEST(TestLogicalParUnseq);

void TestLogicalParUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestLogical(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestLogicalParUnseqOnSequencedExecutor);

void TestLogicalParUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestLogical(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestLogicalParUnseqOnTestSyncExecutor);

void TestLogicalParUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestLogical(thrust::hpx::par_unseq.with(acs));
}
DECLARE_UNITTEST(TestLogicalParUnseqWithAutoChunkSize);

void TestLogicalSeq()
{
  TestLogical(thrust::hpx::seq);
}
DECLARE_UNITTEST(TestLogicalSeq);

void TestLogicalSeqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestLogical(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestLogicalSeqOnSequencedExecutor);

void TestLogicalSeqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestLogical(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestLogicalSeqOnTestSyncExecutor);

void TestLogicalSeqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestLogical(thrust::hpx::seq.with(acs));
}
DECLARE_UNITTEST(TestLogicalSeqWithAutoChunkSize);

void TestLogicalUnseq()
{
  TestLogical(thrust::hpx::unseq);
}
DECLARE_UNITTEST(TestLogicalUnseq);

void TestLogicalUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestLogical(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestLogicalUnseqOnSequencedExecutor);

void TestLogicalUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestLogical(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestLogicalUnseqOnTestSyncExecutor);

void TestLogicalUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestLogical(thrust::hpx::unseq.with(acs));
}
DECLARE_UNITTEST(TestLogicalUnseqWithAutoChunkSize);
