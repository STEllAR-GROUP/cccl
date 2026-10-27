#include <thrust/functional.h>
#include <thrust/gather.h>

#include "test_executor.h"
#include <unittest/unittest.h>

struct is_even_gather
{
  _CCCL_HOST_DEVICE bool operator()(int x) const
  {
    return (x % 2) == 0;
  }
};

template <typename ExecutionPolicy>
void TestGather(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;

  const Vector map{6, 2, 1, 7, 2};
  const Vector src{0, 1, 2, 3, 4, 5, 6, 7};

  // Test 1: gather
  {
    Vector dst(5, 0);
    auto end = thrust::gather(exec, map.begin(), map.end(), src.begin(), dst.begin());
    ASSERT_EQUAL(end - dst.begin(), 5);
    const Vector ref{6, 2, 1, 7, 2};
    ASSERT_EQUAL(dst, ref);
  }

  // Test 2: gather_if
  {
    const Vector stencil{1, 0, 1, 0, 1};
    Vector dst(5, 0);
    auto end = thrust::gather_if(exec, map.begin(), map.end(), stencil.begin(), src.begin(), dst.begin());
    ASSERT_EQUAL(end - dst.begin(), 5);
    const Vector ref{6, 0, 1, 0, 2};
    ASSERT_EQUAL(dst, ref);
  }

  // Test 3: gather_if with predicate
  {
    const Vector stencil{2, 3, 4, 5, 6};
    Vector dst(5, 0);
    auto end =
      thrust::gather_if(exec, map.begin(), map.end(), stencil.begin(), src.begin(), dst.begin(), is_even_gather{});
    ASSERT_EQUAL(end - dst.begin(), 5);
    const Vector ref{6, 0, 1, 0, 2};
    ASSERT_EQUAL(dst, ref);
  }

  // Test 4: empty
  {
    const Vector empty_map;
    Vector dst;
    auto end = thrust::gather(exec, empty_map.begin(), empty_map.end(), src.begin(), dst.begin());
    ASSERT_EQUAL(end - dst.begin(), 0);
  }
}

void TestGatherPar()
{
  TestGather(thrust::hpx::par);
}
DECLARE_UNITTEST(TestGatherPar);

void TestGatherParOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestGather(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestGatherParOnSequencedExecutor);

void TestGatherParOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestGather(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestGatherParOnTestSyncExecutor);

void TestGatherParWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestGather(thrust::hpx::par.with(acs));
}
DECLARE_UNITTEST(TestGatherParWithAutoChunkSize);

void TestGatherParUnseq()
{
  TestGather(thrust::hpx::par_unseq);
}
DECLARE_UNITTEST(TestGatherParUnseq);

void TestGatherParUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestGather(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestGatherParUnseqOnSequencedExecutor);

void TestGatherParUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestGather(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestGatherParUnseqOnTestSyncExecutor);

void TestGatherParUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestGather(thrust::hpx::par_unseq.with(acs));
}
DECLARE_UNITTEST(TestGatherParUnseqWithAutoChunkSize);

void TestGatherSeq()
{
  TestGather(thrust::hpx::seq);
}
DECLARE_UNITTEST(TestGatherSeq);

void TestGatherSeqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestGather(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestGatherSeqOnSequencedExecutor);

void TestGatherSeqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestGather(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestGatherSeqOnTestSyncExecutor);

void TestGatherSeqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestGather(thrust::hpx::seq.with(acs));
}
DECLARE_UNITTEST(TestGatherSeqWithAutoChunkSize);

void TestGatherUnseq()
{
  TestGather(thrust::hpx::unseq);
}
DECLARE_UNITTEST(TestGatherUnseq);

void TestGatherUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestGather(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestGatherUnseqOnSequencedExecutor);

void TestGatherUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestGather(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestGatherUnseqOnTestSyncExecutor);

void TestGatherUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestGather(thrust::hpx::unseq.with(acs));
}
DECLARE_UNITTEST(TestGatherUnseqWithAutoChunkSize);
