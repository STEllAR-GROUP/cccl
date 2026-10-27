#include <thrust/functional.h>
#include <thrust/transform_scan.h>

#include "test_executor.h"
#include <unittest/unittest.h>

struct negate
{
  __host__ __device__ int operator()(int x) const
  {
    return -x;
  }
};

template <typename ExecutionPolicy>
void TestTransformScan(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;

  // Test 1: transform_inclusive_scan without init
  {
    const Vector input{1, 2, 3, 4};
    Vector output(4, 0);

    auto end =
      thrust::transform_inclusive_scan(exec, input.begin(), input.end(), output.begin(), negate{}, thrust::plus<int>{});
    ASSERT_EQUAL(end - output.begin(), 4);
    ASSERT_EQUAL(output[0], -1);
    ASSERT_EQUAL(output[1], -3);
    ASSERT_EQUAL(output[2], -6);
    ASSERT_EQUAL(output[3], -10);
  }

  // Test 2: transform_inclusive_scan with init
  {
    const Vector input{1, 2, 3, 4};
    Vector output(4, 0);

    auto end = thrust::transform_inclusive_scan(
      exec, input.begin(), input.end(), output.begin(), negate{}, 10, thrust::plus<int>{});
    ASSERT_EQUAL(end - output.begin(), 4);
    ASSERT_EQUAL(output[0], 9);
    ASSERT_EQUAL(output[1], 7);
    ASSERT_EQUAL(output[2], 4);
    ASSERT_EQUAL(output[3], 0);
  }

  // Test 3: transform_exclusive_scan
  {
    const Vector input{1, 2, 3, 4};
    Vector output(4, 0);

    auto end = thrust::transform_exclusive_scan(
      exec, input.begin(), input.end(), output.begin(), negate{}, 0, thrust::plus<int>{});
    ASSERT_EQUAL(end - output.begin(), 4);
    ASSERT_EQUAL(output[0], 0);
    ASSERT_EQUAL(output[1], -1);
    ASSERT_EQUAL(output[2], -3);
    ASSERT_EQUAL(output[3], -6);
  }

  // Test 4: empty range
  {
    const Vector empty;
    Vector out;

    auto end =
      thrust::transform_inclusive_scan(exec, empty.begin(), empty.end(), out.begin(), negate{}, thrust::plus<int>{});
    ASSERT_EQUAL(end - out.begin(), 0);
  }
}

void TestTransformScanPar()
{
  TestTransformScan(thrust::hpx::par);
}
DECLARE_UNITTEST(TestTransformScanPar);

void TestTransformScanParOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestTransformScan(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestTransformScanParOnSequencedExecutor);

void TestTransformScanParOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestTransformScan(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestTransformScanParOnTestSyncExecutor);

void TestTransformScanParWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestTransformScan(thrust::hpx::par.with(acs));
}
DECLARE_UNITTEST(TestTransformScanParWithAutoChunkSize);

void TestTransformScanParUnseq()
{
  TestTransformScan(thrust::hpx::par_unseq);
}
DECLARE_UNITTEST(TestTransformScanParUnseq);

void TestTransformScanParUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestTransformScan(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestTransformScanParUnseqOnSequencedExecutor);

void TestTransformScanParUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestTransformScan(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestTransformScanParUnseqOnTestSyncExecutor);

void TestTransformScanParUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestTransformScan(thrust::hpx::par_unseq.with(acs));
}
DECLARE_UNITTEST(TestTransformScanParUnseqWithAutoChunkSize);

void TestTransformScanSeq()
{
  TestTransformScan(thrust::hpx::seq);
}
DECLARE_UNITTEST(TestTransformScanSeq);

void TestTransformScanSeqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestTransformScan(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestTransformScanSeqOnSequencedExecutor);

void TestTransformScanSeqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestTransformScan(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestTransformScanSeqOnTestSyncExecutor);

void TestTransformScanSeqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestTransformScan(thrust::hpx::seq.with(acs));
}
DECLARE_UNITTEST(TestTransformScanSeqWithAutoChunkSize);

void TestTransformScanUnseq()
{
  TestTransformScan(thrust::hpx::unseq);
}
DECLARE_UNITTEST(TestTransformScanUnseq);

void TestTransformScanUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestTransformScan(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestTransformScanUnseqOnSequencedExecutor);

void TestTransformScanUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestTransformScan(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestTransformScanUnseqOnTestSyncExecutor);

void TestTransformScanUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestTransformScan(thrust::hpx::unseq.with(acs));
}
DECLARE_UNITTEST(TestTransformScanUnseqWithAutoChunkSize);
