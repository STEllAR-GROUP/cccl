#include <thrust/functional.h>
#include <thrust/set_operations.h>

#include "test_executor.h"
#include <unittest/unittest.h>

template <typename ExecutionPolicy>
void TestSetOperations(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;

  // Inputs: a = {0, 2, 4}, b = {0, 3, 3, 4}
  const Vector a{0, 2, 4};
  const Vector b{0, 3, 3, 4};

  // Test 1: set_union
  {
    Vector result(5, 0);
    auto end = thrust::set_union(exec, a.begin(), a.end(), b.begin(), b.end(), result.begin());
    ASSERT_EQUAL(end - result.begin(), 5);
    const Vector ref{0, 2, 3, 3, 4};
    ASSERT_EQUAL(result, ref);
  }

  // Test 2: set_intersection
  {
    Vector result(2, 0);
    auto end = thrust::set_intersection(exec, a.begin(), a.end(), b.begin(), b.end(), result.begin());
    ASSERT_EQUAL(end - result.begin(), 2);
    ASSERT_EQUAL(result[0], 0);
    ASSERT_EQUAL(result[1], 4);
  }

  // Test 3: set_difference (a - b)
  {
    Vector result(1, 0);
    auto end = thrust::set_difference(exec, a.begin(), a.end(), b.begin(), b.end(), result.begin());
    ASSERT_EQUAL(end - result.begin(), 1);
    ASSERT_EQUAL(result[0], 2);
  }

  // Test 4: set_symmetric_difference
  {
    Vector result(3, 0);
    auto end = thrust::set_symmetric_difference(exec, a.begin(), a.end(), b.begin(), b.end(), result.begin());
    ASSERT_EQUAL(end - result.begin(), 3);
    const Vector ref{2, 3, 3};
    ASSERT_EQUAL(result, ref);
  }

  // Test 5: custom comparator (descending)
  {
    const Vector a_desc{4, 2, 0};
    const Vector b_desc{4, 3, 3, 0};
    Vector result(5, 0);
    auto end = thrust::set_union(
      exec, a_desc.begin(), a_desc.end(), b_desc.begin(), b_desc.end(), result.begin(), thrust::greater<int>());
    ASSERT_EQUAL(end - result.begin(), 5);
    const Vector ref{4, 3, 3, 2, 0};
    ASSERT_EQUAL(result, ref);
  }

  // Test 6: empty ranges
  {
    const Vector empty;
    Vector result(3, 0);
    auto end = thrust::set_union(exec, empty.begin(), empty.end(), a.begin(), a.end(), result.begin());
    ASSERT_EQUAL(end - result.begin(), 3);
    ASSERT_EQUAL(result, a);
  }
}

void TestSetOperationsPar()
{
  TestSetOperations(thrust::hpx::par);
}
DECLARE_UNITTEST(TestSetOperationsPar);

void TestSetOperationsParOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestSetOperations(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestSetOperationsParOnSequencedExecutor);

void TestSetOperationsParOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestSetOperations(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestSetOperationsParOnTestSyncExecutor);

void TestSetOperationsParWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestSetOperations(thrust::hpx::par.with(acs));
}
DECLARE_UNITTEST(TestSetOperationsParWithAutoChunkSize);

void TestSetOperationsParUnseq()
{
  TestSetOperations(thrust::hpx::par_unseq);
}
DECLARE_UNITTEST(TestSetOperationsParUnseq);

void TestSetOperationsParUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestSetOperations(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestSetOperationsParUnseqOnSequencedExecutor);

void TestSetOperationsParUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestSetOperations(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestSetOperationsParUnseqOnTestSyncExecutor);

void TestSetOperationsParUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestSetOperations(thrust::hpx::par_unseq.with(acs));
}
DECLARE_UNITTEST(TestSetOperationsParUnseqWithAutoChunkSize);

void TestSetOperationsSeq()
{
  TestSetOperations(thrust::hpx::seq);
}
DECLARE_UNITTEST(TestSetOperationsSeq);

void TestSetOperationsSeqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestSetOperations(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestSetOperationsSeqOnSequencedExecutor);

void TestSetOperationsSeqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestSetOperations(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestSetOperationsSeqOnTestSyncExecutor);

void TestSetOperationsSeqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestSetOperations(thrust::hpx::seq.with(acs));
}
DECLARE_UNITTEST(TestSetOperationsSeqWithAutoChunkSize);

void TestSetOperationsUnseq()
{
  TestSetOperations(thrust::hpx::unseq);
}
DECLARE_UNITTEST(TestSetOperationsUnseq);

void TestSetOperationsUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestSetOperations(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestSetOperationsUnseqOnSequencedExecutor);

void TestSetOperationsUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestSetOperations(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestSetOperationsUnseqOnTestSyncExecutor);

void TestSetOperationsUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestSetOperations(thrust::hpx::unseq.with(acs));
}
DECLARE_UNITTEST(TestSetOperationsUnseqWithAutoChunkSize);
