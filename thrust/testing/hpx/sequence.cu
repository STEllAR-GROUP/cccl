#include <thrust/sequence.h>

#include "test_executor.h"
#include <unittest/unittest.h>

template <typename ExecutionPolicy>
void TestSequence(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;

  // Test 1: sequence default
  {
    Vector data(5, 0);
    thrust::sequence(exec, data.begin(), data.end());
    for (int i = 0; i < 5; ++i)
    {
      ASSERT_EQUAL(data[i], i);
    }
  }

  // Test 2: sequence with init and step
  {
    Vector data(5, 0);
    thrust::sequence(exec, data.begin(), data.end(), 10, 2);
    ASSERT_EQUAL(data[0], 10);
    ASSERT_EQUAL(data[1], 12);
    ASSERT_EQUAL(data[2], 14);
    ASSERT_EQUAL(data[3], 16);
    ASSERT_EQUAL(data[4], 18);
  }

  // Test 3: empty
  {
    Vector empty;
    thrust::sequence(exec, empty.begin(), empty.end());
    ASSERT_EQUAL(empty.size(), 0);
  }
}

void TestSequencePar()
{
  TestSequence(thrust::hpx::par);
}
DECLARE_UNITTEST(TestSequencePar);

void TestSequenceParOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestSequence(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestSequenceParOnSequencedExecutor);

void TestSequenceParOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestSequence(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestSequenceParOnTestSyncExecutor);

void TestSequenceParWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestSequence(thrust::hpx::par.with(acs));
}
DECLARE_UNITTEST(TestSequenceParWithAutoChunkSize);

void TestSequenceParUnseq()
{
  TestSequence(thrust::hpx::par_unseq);
}
DECLARE_UNITTEST(TestSequenceParUnseq);

void TestSequenceParUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestSequence(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestSequenceParUnseqOnSequencedExecutor);

void TestSequenceParUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestSequence(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestSequenceParUnseqOnTestSyncExecutor);

void TestSequenceParUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestSequence(thrust::hpx::par_unseq.with(acs));
}
DECLARE_UNITTEST(TestSequenceParUnseqWithAutoChunkSize);

void TestSequenceSeq()
{
  TestSequence(thrust::hpx::seq);
}
DECLARE_UNITTEST(TestSequenceSeq);

void TestSequenceSeqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestSequence(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestSequenceSeqOnSequencedExecutor);

void TestSequenceSeqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestSequence(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestSequenceSeqOnTestSyncExecutor);

void TestSequenceSeqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestSequence(thrust::hpx::seq.with(acs));
}
DECLARE_UNITTEST(TestSequenceSeqWithAutoChunkSize);

void TestSequenceUnseq()
{
  TestSequence(thrust::hpx::unseq);
}
DECLARE_UNITTEST(TestSequenceUnseq);

void TestSequenceUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestSequence(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestSequenceUnseqOnSequencedExecutor);

void TestSequenceUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestSequence(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestSequenceUnseqOnTestSyncExecutor);

void TestSequenceUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestSequence(thrust::hpx::unseq.with(acs));
}
DECLARE_UNITTEST(TestSequenceUnseqWithAutoChunkSize);
