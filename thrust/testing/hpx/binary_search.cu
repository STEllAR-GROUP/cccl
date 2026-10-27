#include <thrust/binary_search.h>

#include "test_executor.h"
#include <unittest/unittest.h>

template <typename ExecutionPolicy>
void TestBinarySearch(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;

  const Vector data{10, 20, 20, 30, 40};

  // Test 1: scalar lower_bound, upper_bound, binary_search, equal_range
  {
    auto lb = thrust::lower_bound(exec, data.begin(), data.end(), 20);
    ASSERT_EQUAL(lb - data.begin(), 1);

    auto ub = thrust::upper_bound(exec, data.begin(), data.end(), 20);
    ASSERT_EQUAL(ub - data.begin(), 3);

    bool found = thrust::binary_search(exec, data.begin(), data.end(), 20);
    ASSERT_EQUAL(found, true);

    bool not_found = thrust::binary_search(exec, data.begin(), data.end(), 25);
    ASSERT_EQUAL(not_found, false);

    auto range = thrust::equal_range(exec, data.begin(), data.end(), 20);
    ASSERT_EQUAL(range.first - data.begin(), 1);
    ASSERT_EQUAL(range.second - data.begin(), 3);
  }

  // Test 2: batched lower_bound, upper_bound, binary_search
  {
    const Vector queries{5, 20, 25, 45};
    Vector lb_out(4, 0);
    thrust::lower_bound(exec, data.begin(), data.end(), queries.begin(), queries.end(), lb_out.begin());
    ASSERT_EQUAL(lb_out[0], 0);
    ASSERT_EQUAL(lb_out[1], 1);
    ASSERT_EQUAL(lb_out[2], 3);
    ASSERT_EQUAL(lb_out[3], 5);

    Vector ub_out(4, 0);
    thrust::upper_bound(exec, data.begin(), data.end(), queries.begin(), queries.end(), ub_out.begin());
    ASSERT_EQUAL(ub_out[0], 0);
    ASSERT_EQUAL(ub_out[1], 3);
    ASSERT_EQUAL(ub_out[2], 3);
    ASSERT_EQUAL(ub_out[3], 5);

    Vector bs_out(4, 0);
    thrust::binary_search(exec, data.begin(), data.end(), queries.begin(), queries.end(), bs_out.begin());
    ASSERT_EQUAL(bs_out[0], 0);
    ASSERT_EQUAL(bs_out[1], 1);
    ASSERT_EQUAL(bs_out[2], 0);
    ASSERT_EQUAL(bs_out[3], 0);
  }

  // Test 3: empty range
  {
    const Vector empty;
    auto lb = thrust::lower_bound(exec, empty.begin(), empty.end(), 10);
    ASSERT_EQUAL(lb - empty.begin(), 0);

    bool found = thrust::binary_search(exec, empty.begin(), empty.end(), 10);
    ASSERT_EQUAL(found, false);
  }
}

void TestBinarySearchPar()
{
  TestBinarySearch(thrust::hpx::par);
}
DECLARE_UNITTEST(TestBinarySearchPar);

void TestBinarySearchParOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestBinarySearch(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestBinarySearchParOnSequencedExecutor);

void TestBinarySearchParOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestBinarySearch(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestBinarySearchParOnTestSyncExecutor);

void TestBinarySearchParWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestBinarySearch(thrust::hpx::par.with(acs));
}
DECLARE_UNITTEST(TestBinarySearchParWithAutoChunkSize);

void TestBinarySearchParUnseq()
{
  TestBinarySearch(thrust::hpx::par_unseq);
}
DECLARE_UNITTEST(TestBinarySearchParUnseq);

void TestBinarySearchParUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestBinarySearch(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestBinarySearchParUnseqOnSequencedExecutor);

void TestBinarySearchParUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestBinarySearch(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestBinarySearchParUnseqOnTestSyncExecutor);

void TestBinarySearchParUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestBinarySearch(thrust::hpx::par_unseq.with(acs));
}
DECLARE_UNITTEST(TestBinarySearchParUnseqWithAutoChunkSize);

void TestBinarySearchSeq()
{
  TestBinarySearch(thrust::hpx::seq);
}
DECLARE_UNITTEST(TestBinarySearchSeq);

void TestBinarySearchSeqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestBinarySearch(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestBinarySearchSeqOnSequencedExecutor);

void TestBinarySearchSeqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestBinarySearch(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestBinarySearchSeqOnTestSyncExecutor);

void TestBinarySearchSeqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestBinarySearch(thrust::hpx::seq.with(acs));
}
DECLARE_UNITTEST(TestBinarySearchSeqWithAutoChunkSize);

void TestBinarySearchUnseq()
{
  TestBinarySearch(thrust::hpx::unseq);
}
DECLARE_UNITTEST(TestBinarySearchUnseq);

void TestBinarySearchUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestBinarySearch(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestBinarySearchUnseqOnSequencedExecutor);

void TestBinarySearchUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestBinarySearch(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestBinarySearchUnseqOnTestSyncExecutor);

void TestBinarySearchUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestBinarySearch(thrust::hpx::unseq.with(acs));
}
DECLARE_UNITTEST(TestBinarySearchUnseqWithAutoChunkSize);
