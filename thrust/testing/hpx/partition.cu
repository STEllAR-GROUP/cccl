#include <thrust/partition.h>

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
void TestPartition(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;

  // Test 1: partition
  {
    Vector data{1, 2, 3, 4, 5, 6};
    auto mid = thrust::partition(exec, data.begin(), data.end(), is_even{});
    ASSERT_EQUAL(mid - data.begin(), 3);
    for (auto it = data.begin(); it != mid; ++it)
    {
      ASSERT_EQUAL(is_even{}(*it), true);
    }
    for (auto it = mid; it != data.end(); ++it)
    {
      ASSERT_EQUAL(is_even{}(*it), false);
    }
    ASSERT_EQUAL(thrust::is_partitioned(exec, data.begin(), data.end(), is_even{}), true);
    ASSERT_EQUAL(thrust::partition_point(exec, data.begin(), data.end(), is_even{}), mid);
  }

  // Test 2: stable_partition
  {
    Vector data{1, 2, 3, 4, 5, 6};
    auto mid = thrust::stable_partition(exec, data.begin(), data.end(), is_even{});
    ASSERT_EQUAL(mid - data.begin(), 3);
    ASSERT_EQUAL(data[0], 2);
    ASSERT_EQUAL(data[1], 4);
    ASSERT_EQUAL(data[2], 6);
    ASSERT_EQUAL(data[3], 1);
    ASSERT_EQUAL(data[4], 3);
    ASSERT_EQUAL(data[5], 5);
  }

  // Test 3: partition_copy
  {
    const Vector data{1, 2, 3, 4, 5, 6};
    Vector true_out(3);
    Vector false_out(3);

    auto result =
      thrust::partition_copy(exec, data.begin(), data.end(), true_out.begin(), false_out.begin(), is_even{});
    ASSERT_EQUAL(result.first - true_out.begin(), 3);
    ASSERT_EQUAL(result.second - false_out.begin(), 3);
    ASSERT_EQUAL(true_out[0], 2);
    ASSERT_EQUAL(true_out[1], 4);
    ASSERT_EQUAL(true_out[2], 6);
    ASSERT_EQUAL(false_out[0], 1);
    ASSERT_EQUAL(false_out[1], 3);
    ASSERT_EQUAL(false_out[2], 5);
  }

  // Test 4: empty range
  {
    Vector empty;
    auto mid = thrust::partition(exec, empty.begin(), empty.end(), is_even{});
    ASSERT_EQUAL(mid - empty.begin(), 0);
  }
}

void TestPartitionPar()
{
  TestPartition(thrust::hpx::par);
}
DECLARE_UNITTEST(TestPartitionPar);

void TestPartitionParOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestPartition(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestPartitionParOnSequencedExecutor);

void TestPartitionParOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestPartition(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestPartitionParOnTestSyncExecutor);

void TestPartitionParWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestPartition(thrust::hpx::par.with(acs));
}
DECLARE_UNITTEST(TestPartitionParWithAutoChunkSize);

void TestPartitionParUnseq()
{
  TestPartition(thrust::hpx::par_unseq);
}
DECLARE_UNITTEST(TestPartitionParUnseq);

void TestPartitionParUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestPartition(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestPartitionParUnseqOnSequencedExecutor);

void TestPartitionParUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestPartition(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestPartitionParUnseqOnTestSyncExecutor);

void TestPartitionParUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestPartition(thrust::hpx::par_unseq.with(acs));
}
DECLARE_UNITTEST(TestPartitionParUnseqWithAutoChunkSize);

void TestPartitionSeq()
{
  TestPartition(thrust::hpx::seq);
}
DECLARE_UNITTEST(TestPartitionSeq);

void TestPartitionSeqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestPartition(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestPartitionSeqOnSequencedExecutor);

void TestPartitionSeqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestPartition(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestPartitionSeqOnTestSyncExecutor);

void TestPartitionSeqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestPartition(thrust::hpx::seq.with(acs));
}
DECLARE_UNITTEST(TestPartitionSeqWithAutoChunkSize);

void TestPartitionUnseq()
{
  TestPartition(thrust::hpx::unseq);
}
DECLARE_UNITTEST(TestPartitionUnseq);

void TestPartitionUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestPartition(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestPartitionUnseqOnSequencedExecutor);

void TestPartitionUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestPartition(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestPartitionUnseqOnTestSyncExecutor);

void TestPartitionUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestPartition(thrust::hpx::unseq.with(acs));
}
DECLARE_UNITTEST(TestPartitionUnseqWithAutoChunkSize);
