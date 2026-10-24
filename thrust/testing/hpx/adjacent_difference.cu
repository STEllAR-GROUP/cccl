#include <thrust/adjacent_difference.h>

#include "test_executor.h"
#include <unittest/unittest.h>

template <typename ExecutionPolicy>
void TestAdjacentDifference(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;
  using T      = typename Vector::value_type;

  {
    Vector input{1, 4, 6, 13, 20};
    Vector output(5);
    Vector ref{1, 3, 2, 7, 7};

    auto result = thrust::adjacent_difference(exec, input.begin(), input.end(), output.begin());

    ASSERT_EQUAL(output, ref);
    ASSERT_EQUAL(result, output.end());
  }
  {
    Vector input{1, 4, 6, 13, 20};
    Vector output(5);
    Vector ref{1, 5, 10, 19, 33};

    auto result = thrust::adjacent_difference(exec, input.begin(), input.end(), output.begin(), thrust::plus<T>());

    ASSERT_EQUAL(output, ref);
    ASSERT_EQUAL(result, output.end());
  }
}

void TestAdjacentDifferencePar()
{
  TestAdjacentDifference(thrust::hpx::par);
}
DECLARE_UNITTEST(TestAdjacentDifferencePar);

void TestAdjacentDifferenceParOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestAdjacentDifference(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestAdjacentDifferenceParOnSequencedExecutor);

void TestAdjacentDifferenceParOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestAdjacentDifference(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestAdjacentDifferenceParOnTestSyncExecutor);

void TestAdjacentDifferenceParWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestAdjacentDifference(thrust::hpx::par.with(acs));
}
DECLARE_UNITTEST(TestAdjacentDifferenceParWithAutoChunkSize);

void TestAdjacentDifferenceParUnseq()
{
  TestAdjacentDifference(thrust::hpx::par_unseq);
}
DECLARE_UNITTEST(TestAdjacentDifferenceParUnseq);

void TestAdjacentDifferenceParUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestAdjacentDifference(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestAdjacentDifferenceParUnseqOnSequencedExecutor);

void TestAdjacentDifferenceParUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestAdjacentDifference(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestAdjacentDifferenceParUnseqOnTestSyncExecutor);

void TestAdjacentDifferenceParUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestAdjacentDifference(thrust::hpx::par_unseq.with(acs));
}
DECLARE_UNITTEST(TestAdjacentDifferenceParUnseqWithAutoChunkSize);

void TestAdjacentDifferenceSeq()
{
  TestAdjacentDifference(thrust::hpx::seq);
}
DECLARE_UNITTEST(TestAdjacentDifferenceSeq);

void TestAdjacentDifferenceSeqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestAdjacentDifference(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestAdjacentDifferenceSeqOnSequencedExecutor);

void TestAdjacentDifferenceSeqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestAdjacentDifference(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestAdjacentDifferenceSeqOnTestSyncExecutor);

void TestAdjacentDifferenceSeqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestAdjacentDifference(thrust::hpx::seq.with(acs));
}
DECLARE_UNITTEST(TestAdjacentDifferenceSeqWithAutoChunkSize);

void TestAdjacentDifferenceUnseq()
{
  TestAdjacentDifference(thrust::hpx::unseq);
}
DECLARE_UNITTEST(TestAdjacentDifferenceUnseq);

void TestAdjacentDifferenceUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestAdjacentDifference(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestAdjacentDifferenceUnseqOnSequencedExecutor);

void TestAdjacentDifferenceUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestAdjacentDifference(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestAdjacentDifferenceUnseqOnTestSyncExecutor);

void TestAdjacentDifferenceUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestAdjacentDifference(thrust::hpx::unseq.with(acs));
}
DECLARE_UNITTEST(TestAdjacentDifferenceUnseqWithAutoChunkSize);
