#include <thrust/functional.h>
#include <thrust/scatter.h>

#include "test_executor.h"
#include <unittest/unittest.h>

struct is_even_scatter
{
  _CCCL_HOST_DEVICE bool operator()(int x) const
  {
    return (x % 2) == 0;
  }
};

template <typename ExecutionPolicy>
void TestScatter(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;

  const Vector src{0, 1, 2, 3, 4};
  const Vector map{6, 3, 1, 7, 2};

  // Test 1: scatter
  {
    Vector dst(8, 0);
    thrust::scatter(exec, src.begin(), src.end(), map.begin(), dst.begin());
    const Vector ref{0, 2, 4, 1, 0, 0, 0, 3};
    ASSERT_EQUAL(dst, ref);
  }

  // Test 2: scatter_if
  {
    const Vector stencil{1, 0, 1, 0, 1};
    Vector dst(8, 0);
    thrust::scatter_if(exec, src.begin(), src.end(), map.begin(), stencil.begin(), dst.begin());
    const Vector ref{0, 0, 4, 0, 0, 0, 0, 0};
    ASSERT_EQUAL(dst, ref);
  }

  // Test 3: scatter_if with predicate
  {
    const Vector stencil{2, 3, 4, 5, 6};
    Vector dst(8, 0);
    thrust::scatter_if(exec, src.begin(), src.end(), map.begin(), stencil.begin(), dst.begin(), is_even_scatter{});
    const Vector ref{0, 0, 4, 0, 0, 0, 0, 0};
    ASSERT_EQUAL(dst, ref);
  }

  // Test 4: empty
  {
    const Vector empty_src;
    const Vector empty_map;
    Vector dst(8, 0);
    thrust::scatter(exec, empty_src.begin(), empty_src.end(), empty_map.begin(), dst.begin());
    const Vector ref(8, 0);
    ASSERT_EQUAL(dst, ref);
  }
}

void TestScatterPar()
{
  TestScatter(thrust::hpx::par);
}
DECLARE_UNITTEST(TestScatterPar);

void TestScatterParOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestScatter(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestScatterParOnSequencedExecutor);

void TestScatterParOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestScatter(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestScatterParOnTestSyncExecutor);

void TestScatterParWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestScatter(thrust::hpx::par.with(acs));
}
DECLARE_UNITTEST(TestScatterParWithAutoChunkSize);

void TestScatterParUnseq()
{
  TestScatter(thrust::hpx::par_unseq);
}
DECLARE_UNITTEST(TestScatterParUnseq);

void TestScatterParUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestScatter(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestScatterParUnseqOnSequencedExecutor);

void TestScatterParUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestScatter(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestScatterParUnseqOnTestSyncExecutor);

void TestScatterParUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestScatter(thrust::hpx::par_unseq.with(acs));
}
DECLARE_UNITTEST(TestScatterParUnseqWithAutoChunkSize);

void TestScatterSeq()
{
  TestScatter(thrust::hpx::seq);
}
DECLARE_UNITTEST(TestScatterSeq);

void TestScatterSeqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestScatter(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestScatterSeqOnSequencedExecutor);

void TestScatterSeqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestScatter(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestScatterSeqOnTestSyncExecutor);

void TestScatterSeqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestScatter(thrust::hpx::seq.with(acs));
}
DECLARE_UNITTEST(TestScatterSeqWithAutoChunkSize);

void TestScatterUnseq()
{
  TestScatter(thrust::hpx::unseq);
}
DECLARE_UNITTEST(TestScatterUnseq);

void TestScatterUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestScatter(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestScatterUnseqOnSequencedExecutor);

void TestScatterUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestScatter(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestScatterUnseqOnTestSyncExecutor);

void TestScatterUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestScatter(thrust::hpx::unseq.with(acs));
}
DECLARE_UNITTEST(TestScatterUnseqWithAutoChunkSize);
