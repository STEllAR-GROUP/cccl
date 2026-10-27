#include <thrust/scan.h>

#include "test_executor.h"
#include <unittest/unittest.h>

template <typename ExecutionPolicy>
void TestScan(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;
  using T      = typename Vector::value_type;

  const Vector input{1, 3, 5, 7, 9};

  // Test inclusive_scan (without init)
  {
    Vector output(input.size());
    const Vector ref{1, 4, 9, 16, 25};

    auto iter = thrust::inclusive_scan(exec, input.begin(), input.end(), output.begin());

    ASSERT_EQUAL(output, ref);
    ASSERT_EQUAL(iter, output.end());
  }

  // Test inclusive_scan with custom binary op
  {
    Vector output(input.size());
    const Vector ref{1, 3, 15, 105, 945};

    auto iter = thrust::inclusive_scan(exec, input.begin(), input.end(), output.begin(), thrust::multiplies<T>());

    ASSERT_EQUAL(output, ref);
    ASSERT_EQUAL(iter, output.end());
  }

  // Test inclusive_scan with init
  {
    Vector output(input.size());
    const Vector ref{11, 14, 19, 26, 35};

    auto iter = thrust::inclusive_scan(exec, input.begin(), input.end(), output.begin(), (T) 10, thrust::plus<T>());

    ASSERT_EQUAL(output, ref);
    ASSERT_EQUAL(iter, output.end());
  }

  // Test exclusive_scan (default init 0)
  {
    Vector output(input.size());
    const Vector ref{0, 1, 4, 9, 16};

    auto iter = thrust::exclusive_scan(exec, input.begin(), input.end(), output.begin());

    ASSERT_EQUAL(output, ref);
    ASSERT_EQUAL(iter, output.end());
  }

  // Test exclusive_scan with custom init
  {
    Vector output(input.size());
    const Vector ref{10, 11, 14, 19, 26};

    auto iter = thrust::exclusive_scan(exec, input.begin(), input.end(), output.begin(), (T) 10);

    ASSERT_EQUAL(output, ref);
    ASSERT_EQUAL(iter, output.end());
  }

  // Test exclusive_scan with custom init and binary op
  {
    Vector output(input.size());
    const Vector ref{2, 2, 6, 30, 210};

    auto iter =
      thrust::exclusive_scan(exec, input.begin(), input.end(), output.begin(), (T) 2, thrust::multiplies<T>());

    ASSERT_EQUAL(output, ref);
    ASSERT_EQUAL(iter, output.end());
  }

  // Test empty ranges
  {
    Vector empty_in;
    Vector empty_out;

    auto iter_inc = thrust::inclusive_scan(exec, empty_in.begin(), empty_in.end(), empty_out.begin());
    ASSERT_EQUAL(iter_inc, empty_out.end());

    auto iter_exc = thrust::exclusive_scan(exec, empty_in.begin(), empty_in.end(), empty_out.begin(), (T) 0);
    ASSERT_EQUAL(iter_exc, empty_out.end());
  }
}

void TestScanPar()
{
  TestScan(thrust::hpx::par);
}
DECLARE_UNITTEST(TestScanPar);

void TestScanParOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestScan(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestScanParOnSequencedExecutor);

void TestScanParOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestScan(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestScanParOnTestSyncExecutor);

void TestScanParWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestScan(thrust::hpx::par.with(acs));
}
DECLARE_UNITTEST(TestScanParWithAutoChunkSize);

void TestScanParUnseq()
{
  TestScan(thrust::hpx::par_unseq);
}
DECLARE_UNITTEST(TestScanParUnseq);

void TestScanParUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestScan(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestScanParUnseqOnSequencedExecutor);

void TestScanParUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestScan(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestScanParUnseqOnTestSyncExecutor);

void TestScanParUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestScan(thrust::hpx::par_unseq.with(acs));
}
DECLARE_UNITTEST(TestScanParUnseqWithAutoChunkSize);

void TestScanSeq()
{
  TestScan(thrust::hpx::seq);
}
DECLARE_UNITTEST(TestScanSeq);

void TestScanSeqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestScan(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestScanSeqOnSequencedExecutor);

void TestScanSeqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestScan(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestScanSeqOnTestSyncExecutor);

void TestScanSeqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestScan(thrust::hpx::seq.with(acs));
}
DECLARE_UNITTEST(TestScanSeqWithAutoChunkSize);

void TestScanUnseq()
{
  TestScan(thrust::hpx::unseq);
}
DECLARE_UNITTEST(TestScanUnseq);

void TestScanUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestScan(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestScanUnseqOnSequencedExecutor);

void TestScanUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestScan(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestScanUnseqOnTestSyncExecutor);

void TestScanUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestScan(thrust::hpx::unseq.with(acs));
}
DECLARE_UNITTEST(TestScanUnseqWithAutoChunkSize);
