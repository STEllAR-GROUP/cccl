#include <thrust/functional.h>
#include <thrust/reduce.h>

#include "test_executor.h"
#include <unittest/unittest.h>

template <typename ExecutionPolicy>
void TestReduceByKey(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;

  // Test 1: Basic reduce_by_key
  {
    const Vector keys{11, 11, 21, 20, 21, 21, 21, 37, 37};
    const Vector values{0, 1, 2, 3, 4, 5, 6, 7, 8};

    Vector out_keys(9, 0);
    Vector out_values(9, 0);

    auto new_last =
      thrust::reduce_by_key(exec, keys.begin(), keys.end(), values.begin(), out_keys.begin(), out_values.begin());

    ASSERT_EQUAL(new_last.first - out_keys.begin(), 5);
    ASSERT_EQUAL(new_last.second - out_values.begin(), 5);

    out_keys.resize(5);
    out_values.resize(5);

    const Vector ref_keys{11, 21, 20, 21, 37};
    const Vector ref_values{1, 2, 3, 15, 15};

    ASSERT_EQUAL(out_keys, ref_keys);
    ASSERT_EQUAL(out_values, ref_values);
  }

  // Test 2: Custom binary op (multiplies)
  {
    const Vector keys{1, 1, 2, 2, 2};
    const Vector values{2, 3, 2, 3, 4};

    Vector out_keys(5, 0);
    Vector out_values(5, 0);

    auto new_last = thrust::reduce_by_key(
      exec,
      keys.begin(),
      keys.end(),
      values.begin(),
      out_keys.begin(),
      out_values.begin(),
      thrust::equal_to<int>(),
      thrust::multiplies<int>());

    ASSERT_EQUAL(new_last.first - out_keys.begin(), 2);
    ASSERT_EQUAL(new_last.second - out_values.begin(), 2);

    out_keys.resize(2);
    out_values.resize(2);

    const Vector ref_keys{1, 2};
    const Vector ref_values{6, 24};

    ASSERT_EQUAL(out_keys, ref_keys);
    ASSERT_EQUAL(out_values, ref_values);
  }

  // Test 3: Empty range
  {
    const Vector empty_keys;
    const Vector empty_values;

    Vector out_keys;
    Vector out_values;

    auto new_last = thrust::reduce_by_key(
      exec, empty_keys.begin(), empty_keys.end(), empty_values.begin(), out_keys.begin(), out_values.begin());

    ASSERT_EQUAL(new_last.first - out_keys.begin(), 0);
    ASSERT_EQUAL(new_last.second - out_values.begin(), 0);
  }
}

void TestReduceByKeyPar()
{
  TestReduceByKey(thrust::hpx::par);
}
DECLARE_UNITTEST(TestReduceByKeyPar);

void TestReduceByKeyParOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestReduceByKey(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestReduceByKeyParOnSequencedExecutor);

void TestReduceByKeyParOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestReduceByKey(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestReduceByKeyParOnTestSyncExecutor);

void TestReduceByKeyParWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestReduceByKey(thrust::hpx::par.with(acs));
}
DECLARE_UNITTEST(TestReduceByKeyParWithAutoChunkSize);

void TestReduceByKeyParUnseq()
{
  TestReduceByKey(thrust::hpx::par_unseq);
}
DECLARE_UNITTEST(TestReduceByKeyParUnseq);

void TestReduceByKeyParUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestReduceByKey(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestReduceByKeyParUnseqOnSequencedExecutor);

void TestReduceByKeyParUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestReduceByKey(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestReduceByKeyParUnseqOnTestSyncExecutor);

void TestReduceByKeyParUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestReduceByKey(thrust::hpx::par_unseq.with(acs));
}
DECLARE_UNITTEST(TestReduceByKeyParUnseqWithAutoChunkSize);

void TestReduceByKeySeq()
{
  TestReduceByKey(thrust::hpx::seq);
}
DECLARE_UNITTEST(TestReduceByKeySeq);

void TestReduceByKeySeqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestReduceByKey(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestReduceByKeySeqOnSequencedExecutor);

void TestReduceByKeySeqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestReduceByKey(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestReduceByKeySeqOnTestSyncExecutor);

void TestReduceByKeySeqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestReduceByKey(thrust::hpx::seq.with(acs));
}
DECLARE_UNITTEST(TestReduceByKeySeqWithAutoChunkSize);

void TestReduceByKeyUnseq()
{
  TestReduceByKey(thrust::hpx::unseq);
}
DECLARE_UNITTEST(TestReduceByKeyUnseq);

void TestReduceByKeyUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestReduceByKey(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestReduceByKeyUnseqOnSequencedExecutor);

void TestReduceByKeyUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestReduceByKey(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestReduceByKeyUnseqOnTestSyncExecutor);

void TestReduceByKeyUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestReduceByKey(thrust::hpx::unseq.with(acs));
}
DECLARE_UNITTEST(TestReduceByKeyUnseqWithAutoChunkSize);
