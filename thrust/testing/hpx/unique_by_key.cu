#include <thrust/functional.h>
#include <thrust/unique.h>

#include "test_executor.h"
#include <unittest/unittest.h>

template <typename ExecutionPolicy>
void TestUniqueByKey(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;

  // Test 1: unique_by_key in-place
  {
    Vector keys{11, 11, 21, 20, 21, 21, 21, 37, 37};
    Vector values{0, 1, 2, 3, 4, 5, 6, 7, 8};

    auto new_last = thrust::unique_by_key(exec, keys.begin(), keys.end(), values.begin());

    ASSERT_EQUAL(new_last.first - keys.begin(), 5);
    ASSERT_EQUAL(new_last.second - values.begin(), 5);
    keys.resize(5);
    values.resize(5);

    const Vector keys_ref{11, 21, 20, 21, 37};
    const Vector values_ref{0, 2, 3, 4, 7};
    ASSERT_EQUAL(keys, keys_ref);
    ASSERT_EQUAL(values, values_ref);
  }

  // Test 2: unique_by_key_copy
  {
    const Vector keys{11, 11, 21, 20, 21, 21, 21, 37, 37};
    const Vector values{0, 1, 2, 3, 4, 5, 6, 7, 8};

    Vector out_keys(9, 0);
    Vector out_values(9, 0);

    auto new_last =
      thrust::unique_by_key_copy(exec, keys.begin(), keys.end(), values.begin(), out_keys.begin(), out_values.begin());

    ASSERT_EQUAL(new_last.first - out_keys.begin(), 5);
    ASSERT_EQUAL(new_last.second - out_values.begin(), 5);
    out_keys.resize(5);
    out_values.resize(5);

    const Vector keys_ref{11, 21, 20, 21, 37};
    const Vector values_ref{0, 2, 3, 4, 7};
    ASSERT_EQUAL(out_keys, keys_ref);
    ASSERT_EQUAL(out_values, values_ref);
  }

  // Test 3: empty range
  {
    Vector empty_keys;
    Vector empty_values;

    auto new_last = thrust::unique_by_key(exec, empty_keys.begin(), empty_keys.end(), empty_values.begin());
    ASSERT_EQUAL(new_last.first - empty_keys.begin(), 0);
    ASSERT_EQUAL(new_last.second - empty_values.begin(), 0);
  }
}

void TestUniqueByKeyPar()
{
  TestUniqueByKey(thrust::hpx::par);
}
DECLARE_UNITTEST(TestUniqueByKeyPar);

void TestUniqueByKeyParOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestUniqueByKey(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestUniqueByKeyParOnSequencedExecutor);

void TestUniqueByKeyParOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestUniqueByKey(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestUniqueByKeyParOnTestSyncExecutor);

void TestUniqueByKeyParWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestUniqueByKey(thrust::hpx::par.with(acs));
}
DECLARE_UNITTEST(TestUniqueByKeyParWithAutoChunkSize);

void TestUniqueByKeyParUnseq()
{
  TestUniqueByKey(thrust::hpx::par_unseq);
}
DECLARE_UNITTEST(TestUniqueByKeyParUnseq);

void TestUniqueByKeyParUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestUniqueByKey(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestUniqueByKeyParUnseqOnSequencedExecutor);

void TestUniqueByKeyParUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestUniqueByKey(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestUniqueByKeyParUnseqOnTestSyncExecutor);

void TestUniqueByKeyParUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestUniqueByKey(thrust::hpx::par_unseq.with(acs));
}
DECLARE_UNITTEST(TestUniqueByKeyParUnseqWithAutoChunkSize);

void TestUniqueByKeySeq()
{
  TestUniqueByKey(thrust::hpx::seq);
}
DECLARE_UNITTEST(TestUniqueByKeySeq);

void TestUniqueByKeySeqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestUniqueByKey(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestUniqueByKeySeqOnSequencedExecutor);

void TestUniqueByKeySeqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestUniqueByKey(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestUniqueByKeySeqOnTestSyncExecutor);

void TestUniqueByKeySeqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestUniqueByKey(thrust::hpx::seq.with(acs));
}
DECLARE_UNITTEST(TestUniqueByKeySeqWithAutoChunkSize);

void TestUniqueByKeyUnseq()
{
  TestUniqueByKey(thrust::hpx::unseq);
}
DECLARE_UNITTEST(TestUniqueByKeyUnseq);

void TestUniqueByKeyUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestUniqueByKey(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestUniqueByKeyUnseqOnSequencedExecutor);

void TestUniqueByKeyUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestUniqueByKey(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestUniqueByKeyUnseqOnTestSyncExecutor);

void TestUniqueByKeyUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestUniqueByKey(thrust::hpx::unseq.with(acs));
}
DECLARE_UNITTEST(TestUniqueByKeyUnseqWithAutoChunkSize);
