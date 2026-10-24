#include <thrust/sort.h>

#include "test_executor.h"
#include <unittest/unittest.h>

template <typename ExecutionPolicy>
void TestSort(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;
  using T      = typename Vector::value_type;

  {
    Vector data{5, 2, 8, 1, 9, 3};
    Vector ref{1, 2, 3, 5, 8, 9};

    thrust::sort(exec, data.begin(), data.end());

    ASSERT_EQUAL(data, ref);
  }
  {
    Vector data{5, 2, 8, 1, 9, 3};
    Vector ref{9, 8, 5, 3, 2, 1};

    thrust::sort(exec, data.begin(), data.end(), thrust::greater<T>());

    ASSERT_EQUAL(data, ref);
  }
}

void TestSortPar()
{
  TestSort(thrust::hpx::par);
}
DECLARE_UNITTEST(TestSortPar);

void TestSortParOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestSort(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestSortParOnSequencedExecutor);

void TestSortParOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestSort(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestSortParOnTestSyncExecutor);

void TestSortParWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestSort(thrust::hpx::par.with(acs));
}
DECLARE_UNITTEST(TestSortParWithAutoChunkSize);

void TestSortParUnseq()
{
  TestSort(thrust::hpx::par_unseq);
}
DECLARE_UNITTEST(TestSortParUnseq);

void TestSortParUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestSort(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestSortParUnseqOnSequencedExecutor);

void TestSortParUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestSort(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestSortParUnseqOnTestSyncExecutor);

void TestSortParUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestSort(thrust::hpx::par_unseq.with(acs));
}
DECLARE_UNITTEST(TestSortParUnseqWithAutoChunkSize);

void TestSortSeq()
{
  TestSort(thrust::hpx::seq);
}
DECLARE_UNITTEST(TestSortSeq);

void TestSortSeqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestSort(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestSortSeqOnSequencedExecutor);

void TestSortSeqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestSort(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestSortSeqOnTestSyncExecutor);

void TestSortSeqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestSort(thrust::hpx::seq.with(acs));
}
DECLARE_UNITTEST(TestSortSeqWithAutoChunkSize);

void TestSortUnseq()
{
  TestSort(thrust::hpx::unseq);
}
DECLARE_UNITTEST(TestSortUnseq);

void TestSortUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestSort(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestSortUnseqOnSequencedExecutor);

void TestSortUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestSort(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestSortUnseqOnTestSyncExecutor);

void TestSortUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestSort(thrust::hpx::unseq.with(acs));
}
DECLARE_UNITTEST(TestSortUnseqWithAutoChunkSize);
