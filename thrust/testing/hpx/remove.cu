#include <thrust/remove.h>

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
void TestRemove(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;

  // Test 1: remove
  {
    Vector data{1, 2, 3, 2, 5, 2};
    auto mid = thrust::remove(exec, data.begin(), data.end(), 2);
    ASSERT_EQUAL(mid - data.begin(), 3);
    ASSERT_EQUAL(data[0], 1);
    ASSERT_EQUAL(data[1], 3);
    ASSERT_EQUAL(data[2], 5);
  }

  // Test 2: remove_if
  {
    Vector data{1, 2, 3, 4, 5, 6};
    auto mid = thrust::remove_if(exec, data.begin(), data.end(), is_even{});
    ASSERT_EQUAL(mid - data.begin(), 3);
    ASSERT_EQUAL(data[0], 1);
    ASSERT_EQUAL(data[1], 3);
    ASSERT_EQUAL(data[2], 5);
  }

  // Test 3: remove_copy
  {
    const Vector data{1, 2, 3, 2, 5, 2};
    Vector out(3, 0);
    auto mid = thrust::remove_copy(exec, data.begin(), data.end(), out.begin(), 2);
    ASSERT_EQUAL(mid - out.begin(), 3);
    ASSERT_EQUAL(out[0], 1);
    ASSERT_EQUAL(out[1], 3);
    ASSERT_EQUAL(out[2], 5);
  }

  // Test 4: remove_copy_if
  {
    const Vector data{1, 2, 3, 4, 5, 6};
    Vector out(3, 0);
    auto mid = thrust::remove_copy_if(exec, data.begin(), data.end(), out.begin(), is_even{});
    ASSERT_EQUAL(mid - out.begin(), 3);
    ASSERT_EQUAL(out[0], 1);
    ASSERT_EQUAL(out[1], 3);
    ASSERT_EQUAL(out[2], 5);
  }

  // Test 5: empty range
  {
    Vector empty;
    auto mid = thrust::remove(exec, empty.begin(), empty.end(), 2);
    ASSERT_EQUAL(mid - empty.begin(), 0);
  }
}

void TestRemovePar()
{
  TestRemove(thrust::hpx::par);
}
DECLARE_UNITTEST(TestRemovePar);

void TestRemoveParOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestRemove(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestRemoveParOnSequencedExecutor);

void TestRemoveParOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestRemove(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestRemoveParOnTestSyncExecutor);

void TestRemoveParWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestRemove(thrust::hpx::par.with(acs));
}
DECLARE_UNITTEST(TestRemoveParWithAutoChunkSize);

void TestRemoveParUnseq()
{
  TestRemove(thrust::hpx::par_unseq);
}
DECLARE_UNITTEST(TestRemoveParUnseq);

void TestRemoveParUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestRemove(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestRemoveParUnseqOnSequencedExecutor);

void TestRemoveParUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestRemove(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestRemoveParUnseqOnTestSyncExecutor);

void TestRemoveParUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestRemove(thrust::hpx::par_unseq.with(acs));
}
DECLARE_UNITTEST(TestRemoveParUnseqWithAutoChunkSize);

void TestRemoveSeq()
{
  TestRemove(thrust::hpx::seq);
}
DECLARE_UNITTEST(TestRemoveSeq);

void TestRemoveSeqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestRemove(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestRemoveSeqOnSequencedExecutor);

void TestRemoveSeqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestRemove(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestRemoveSeqOnTestSyncExecutor);

void TestRemoveSeqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestRemove(thrust::hpx::seq.with(acs));
}
DECLARE_UNITTEST(TestRemoveSeqWithAutoChunkSize);

void TestRemoveUnseq()
{
  TestRemove(thrust::hpx::unseq);
}
DECLARE_UNITTEST(TestRemoveUnseq);

void TestRemoveUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestRemove(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestRemoveUnseqOnSequencedExecutor);

void TestRemoveUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestRemove(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestRemoveUnseqOnTestSyncExecutor);

void TestRemoveUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestRemove(thrust::hpx::unseq.with(acs));
}
DECLARE_UNITTEST(TestRemoveUnseqWithAutoChunkSize);
