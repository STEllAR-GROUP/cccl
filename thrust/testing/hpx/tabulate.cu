#include <thrust/tabulate.h>

#include "test_executor.h"
#include <unittest/unittest.h>

struct square
{
  __host__ __device__ int operator()(int i) const
  {
    return i * i;
  }
};

template <typename ExecutionPolicy>
void TestTabulate(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;

  // Test 1: tabulate
  {
    Vector data(5, 0);
    thrust::tabulate(exec, data.begin(), data.end(), square{});
    ASSERT_EQUAL(data[0], 0);
    ASSERT_EQUAL(data[1], 1);
    ASSERT_EQUAL(data[2], 4);
    ASSERT_EQUAL(data[3], 9);
    ASSERT_EQUAL(data[4], 16);
  }

  // Test 2: empty
  {
    Vector empty;
    thrust::tabulate(exec, empty.begin(), empty.end(), square{});
    ASSERT_EQUAL(empty.size(), 0);
  }
}

void TestTabulatePar()
{
  TestTabulate(thrust::hpx::par);
}
DECLARE_UNITTEST(TestTabulatePar);

void TestTabulateParOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestTabulate(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestTabulateParOnSequencedExecutor);

void TestTabulateParOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestTabulate(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestTabulateParOnTestSyncExecutor);

void TestTabulateParWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestTabulate(thrust::hpx::par.with(acs));
}
DECLARE_UNITTEST(TestTabulateParWithAutoChunkSize);

void TestTabulateParUnseq()
{
  TestTabulate(thrust::hpx::par_unseq);
}
DECLARE_UNITTEST(TestTabulateParUnseq);

void TestTabulateParUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestTabulate(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestTabulateParUnseqOnSequencedExecutor);

void TestTabulateParUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestTabulate(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestTabulateParUnseqOnTestSyncExecutor);

void TestTabulateParUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestTabulate(thrust::hpx::par_unseq.with(acs));
}
DECLARE_UNITTEST(TestTabulateParUnseqWithAutoChunkSize);

void TestTabulateSeq()
{
  TestTabulate(thrust::hpx::seq);
}
DECLARE_UNITTEST(TestTabulateSeq);

void TestTabulateSeqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestTabulate(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestTabulateSeqOnSequencedExecutor);

void TestTabulateSeqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestTabulate(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestTabulateSeqOnTestSyncExecutor);

void TestTabulateSeqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestTabulate(thrust::hpx::seq.with(acs));
}
DECLARE_UNITTEST(TestTabulateSeqWithAutoChunkSize);

void TestTabulateUnseq()
{
  TestTabulate(thrust::hpx::unseq);
}
DECLARE_UNITTEST(TestTabulateUnseq);

void TestTabulateUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestTabulate(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestTabulateUnseqOnSequencedExecutor);

void TestTabulateUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestTabulate(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestTabulateUnseqOnTestSyncExecutor);

void TestTabulateUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestTabulate(thrust::hpx::unseq.with(acs));
}
DECLARE_UNITTEST(TestTabulateUnseqWithAutoChunkSize);
