#include <thrust/replace.h>

#include "test_executor.h"
#include <unittest/unittest.h>

struct is_even
{
  __host__ __device__ bool operator()(int x) const
  {
    return (x % 2) == 0;
  }
};

template <typename ExecutionPolicy>
void TestReplace(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;

  // Test 1: replace
  {
    Vector data{1, 2, 3, 2, 5, 2};
    thrust::replace(exec, data.begin(), data.end(), 2, 99);
    ASSERT_EQUAL(data[0], 1);
    ASSERT_EQUAL(data[1], 99);
    ASSERT_EQUAL(data[2], 3);
    ASSERT_EQUAL(data[3], 99);
    ASSERT_EQUAL(data[4], 5);
    ASSERT_EQUAL(data[5], 99);
  }

  // Test 2: replace_if
  {
    Vector data{1, 2, 3, 4, 5, 6};
    thrust::replace_if(exec, data.begin(), data.end(), is_even{}, 0);
    ASSERT_EQUAL(data[0], 1);
    ASSERT_EQUAL(data[1], 0);
    ASSERT_EQUAL(data[2], 3);
    ASSERT_EQUAL(data[3], 0);
    ASSERT_EQUAL(data[4], 5);
    ASSERT_EQUAL(data[5], 0);
  }

  // Test 3: replace_copy
  {
    const Vector data{1, 2, 3, 2, 5, 2};
    Vector out(6, 0);
    auto out_end = thrust::replace_copy(exec, data.begin(), data.end(), out.begin(), 2, 99);
    ASSERT_EQUAL(out_end - out.begin(), 6);
    ASSERT_EQUAL(out[0], 1);
    ASSERT_EQUAL(out[1], 99);
    ASSERT_EQUAL(out[2], 3);
    ASSERT_EQUAL(out[3], 99);
    ASSERT_EQUAL(out[4], 5);
    ASSERT_EQUAL(out[5], 99);
  }

  // Test 4: replace_copy_if
  {
    const Vector data{1, 2, 3, 4, 5, 6};
    Vector out(6, 0);
    auto out_end = thrust::replace_copy_if(exec, data.begin(), data.end(), out.begin(), is_even{}, 0);
    ASSERT_EQUAL(out_end - out.begin(), 6);
    ASSERT_EQUAL(out[0], 1);
    ASSERT_EQUAL(out[1], 0);
    ASSERT_EQUAL(out[2], 3);
    ASSERT_EQUAL(out[3], 0);
    ASSERT_EQUAL(out[4], 5);
    ASSERT_EQUAL(out[5], 0);
  }

  // Test 5: empty range
  {
    Vector empty;
    thrust::replace(exec, empty.begin(), empty.end(), 1, 2);
    ASSERT_EQUAL(empty.size(), 0);
  }
}

void TestReplacePar()
{
  TestReplace(thrust::hpx::par);
}
DECLARE_UNITTEST(TestReplacePar);

void TestReplaceParOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestReplace(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestReplaceParOnSequencedExecutor);

void TestReplaceParOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestReplace(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestReplaceParOnTestSyncExecutor);

void TestReplaceParWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestReplace(thrust::hpx::par.with(acs));
}
DECLARE_UNITTEST(TestReplaceParWithAutoChunkSize);

void TestReplaceParUnseq()
{
  TestReplace(thrust::hpx::par_unseq);
}
DECLARE_UNITTEST(TestReplaceParUnseq);

void TestReplaceParUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestReplace(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestReplaceParUnseqOnSequencedExecutor);

void TestReplaceParUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestReplace(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestReplaceParUnseqOnTestSyncExecutor);

void TestReplaceParUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestReplace(thrust::hpx::par_unseq.with(acs));
}
DECLARE_UNITTEST(TestReplaceParUnseqWithAutoChunkSize);

void TestReplaceSeq()
{
  TestReplace(thrust::hpx::seq);
}
DECLARE_UNITTEST(TestReplaceSeq);

void TestReplaceSeqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestReplace(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestReplaceSeqOnSequencedExecutor);

void TestReplaceSeqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestReplace(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestReplaceSeqOnTestSyncExecutor);

void TestReplaceSeqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestReplace(thrust::hpx::seq.with(acs));
}
DECLARE_UNITTEST(TestReplaceSeqWithAutoChunkSize);

void TestReplaceUnseq()
{
  TestReplace(thrust::hpx::unseq);
}
DECLARE_UNITTEST(TestReplaceUnseq);

void TestReplaceUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestReplace(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestReplaceUnseqOnSequencedExecutor);

void TestReplaceUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestReplace(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestReplaceUnseqOnTestSyncExecutor);

void TestReplaceUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestReplace(thrust::hpx::unseq.with(acs));
}
DECLARE_UNITTEST(TestReplaceUnseqWithAutoChunkSize);
