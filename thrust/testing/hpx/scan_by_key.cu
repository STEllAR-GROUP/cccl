#include <thrust/functional.h>
#include <thrust/scan.h>

#include "test_executor.h"
#include <unittest/unittest.h>

template <typename ExecutionPolicy>
void TestScanByKey(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;

  const Vector keys{0, 1, 1, 1, 0, 1, 1};
  const Vector vals{1, 2, 3, 4, 5, 6, 7};

  // Test 1: inclusive_scan_by_key
  {
    Vector output(7, 0);
    auto end = thrust::inclusive_scan_by_key(exec, keys.begin(), keys.end(), vals.begin(), output.begin());
    ASSERT_EQUAL(end - output.begin(), 7);
    const Vector ref{1, 2, 5, 9, 5, 6, 13};
    ASSERT_EQUAL(output, ref);
  }

  // Test 2: inclusive_scan_by_key with custom binary op
  {
    Vector output(7, 0);
    auto end = thrust::inclusive_scan_by_key(
      exec, keys.begin(), keys.end(), vals.begin(), output.begin(), thrust::equal_to<int>(), thrust::plus<int>());
    ASSERT_EQUAL(end - output.begin(), 7);
    const Vector ref{1, 2, 5, 9, 5, 6, 13};
    ASSERT_EQUAL(output, ref);
  }

  // Test 3: exclusive_scan_by_key with init
  {
    Vector output(7, 0);
    auto end = thrust::exclusive_scan_by_key(exec, keys.begin(), keys.end(), vals.begin(), output.begin(), 10);
    ASSERT_EQUAL(end - output.begin(), 7);
    const Vector ref{10, 10, 12, 15, 10, 10, 16};
    ASSERT_EQUAL(output, ref);
  }

  // Test 4: exclusive_scan_by_key default init
  {
    Vector output(7, 0);
    auto end = thrust::exclusive_scan_by_key(exec, keys.begin(), keys.end(), vals.begin(), output.begin());
    ASSERT_EQUAL(end - output.begin(), 7);
    const Vector ref{0, 0, 2, 5, 0, 0, 6};
    ASSERT_EQUAL(output, ref);
  }

  // Test 5: empty range
  {
    const Vector empty_keys;
    const Vector empty_vals;
    Vector empty_out;
    auto end =
      thrust::inclusive_scan_by_key(exec, empty_keys.begin(), empty_keys.end(), empty_vals.begin(), empty_out.begin());
    ASSERT_EQUAL(end - empty_out.begin(), 0);
  }
}

void TestScanByKeyPar()
{
  TestScanByKey(thrust::hpx::par);
}
DECLARE_UNITTEST(TestScanByKeyPar);

void TestScanByKeyParOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestScanByKey(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestScanByKeyParOnSequencedExecutor);

void TestScanByKeyParOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestScanByKey(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestScanByKeyParOnTestSyncExecutor);

void TestScanByKeyParWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestScanByKey(thrust::hpx::par.with(acs));
}
DECLARE_UNITTEST(TestScanByKeyParWithAutoChunkSize);

void TestScanByKeyParUnseq()
{
  TestScanByKey(thrust::hpx::par_unseq);
}
DECLARE_UNITTEST(TestScanByKeyParUnseq);

void TestScanByKeyParUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestScanByKey(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestScanByKeyParUnseqOnSequencedExecutor);

void TestScanByKeyParUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestScanByKey(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestScanByKeyParUnseqOnTestSyncExecutor);

void TestScanByKeyParUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestScanByKey(thrust::hpx::par_unseq.with(acs));
}
DECLARE_UNITTEST(TestScanByKeyParUnseqWithAutoChunkSize);

void TestScanByKeySeq()
{
  TestScanByKey(thrust::hpx::seq);
}
DECLARE_UNITTEST(TestScanByKeySeq);

void TestScanByKeySeqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestScanByKey(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestScanByKeySeqOnSequencedExecutor);

void TestScanByKeySeqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestScanByKey(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestScanByKeySeqOnTestSyncExecutor);

void TestScanByKeySeqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestScanByKey(thrust::hpx::seq.with(acs));
}
DECLARE_UNITTEST(TestScanByKeySeqWithAutoChunkSize);

void TestScanByKeyUnseq()
{
  TestScanByKey(thrust::hpx::unseq);
}
DECLARE_UNITTEST(TestScanByKeyUnseq);

void TestScanByKeyUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestScanByKey(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestScanByKeyUnseqOnSequencedExecutor);

void TestScanByKeyUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestScanByKey(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestScanByKeyUnseqOnTestSyncExecutor);

void TestScanByKeyUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestScanByKey(thrust::hpx::unseq.with(acs));
}
DECLARE_UNITTEST(TestScanByKeyUnseqWithAutoChunkSize);
