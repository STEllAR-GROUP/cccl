#include <thrust/uninitialized_copy.h>

#include "test_executor.h"
#include <unittest/unittest.h>

template <typename ExecutionPolicy>
void TestUninitializedCopy(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;

  const Vector src{0, 1, 2, 3, 4};

  // Test 1: uninitialized_copy
  {
    Vector dst(5, 0);
    auto end = thrust::uninitialized_copy(exec, src.begin(), src.end(), dst.begin());
    ASSERT_EQUAL(end - dst.begin(), 5);
    ASSERT_EQUAL(dst, src);
  }

  // Test 2: uninitialized_copy_n
  {
    Vector dst(5, 0);
    auto end = thrust::uninitialized_copy_n(exec, src.begin(), 3, dst.begin());
    ASSERT_EQUAL(end - dst.begin(), 3);
    ASSERT_EQUAL(dst[0], 0);
    ASSERT_EQUAL(dst[1], 1);
    ASSERT_EQUAL(dst[2], 2);
  }

  // Test 3: empty
  {
    const Vector empty;
    Vector dst;
    auto end = thrust::uninitialized_copy(exec, empty.begin(), empty.end(), dst.begin());
    ASSERT_EQUAL(end - dst.begin(), 0);
  }
}

void TestUninitializedCopyPar()
{
  TestUninitializedCopy(thrust::hpx::par);
}
DECLARE_UNITTEST(TestUninitializedCopyPar);

void TestUninitializedCopyParOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestUninitializedCopy(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestUninitializedCopyParOnSequencedExecutor);

void TestUninitializedCopyParOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestUninitializedCopy(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestUninitializedCopyParOnTestSyncExecutor);

void TestUninitializedCopyParWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestUninitializedCopy(thrust::hpx::par.with(acs));
}
DECLARE_UNITTEST(TestUninitializedCopyParWithAutoChunkSize);

void TestUninitializedCopyParUnseq()
{
  TestUninitializedCopy(thrust::hpx::par_unseq);
}
DECLARE_UNITTEST(TestUninitializedCopyParUnseq);

void TestUninitializedCopyParUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestUninitializedCopy(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestUninitializedCopyParUnseqOnSequencedExecutor);

void TestUninitializedCopyParUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestUninitializedCopy(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestUninitializedCopyParUnseqOnTestSyncExecutor);

void TestUninitializedCopyParUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestUninitializedCopy(thrust::hpx::par_unseq.with(acs));
}
DECLARE_UNITTEST(TestUninitializedCopyParUnseqWithAutoChunkSize);

void TestUninitializedCopySeq()
{
  TestUninitializedCopy(thrust::hpx::seq);
}
DECLARE_UNITTEST(TestUninitializedCopySeq);

void TestUninitializedCopySeqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestUninitializedCopy(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestUninitializedCopySeqOnSequencedExecutor);

void TestUninitializedCopySeqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestUninitializedCopy(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestUninitializedCopySeqOnTestSyncExecutor);

void TestUninitializedCopySeqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestUninitializedCopy(thrust::hpx::seq.with(acs));
}
DECLARE_UNITTEST(TestUninitializedCopySeqWithAutoChunkSize);

void TestUninitializedCopyUnseq()
{
  TestUninitializedCopy(thrust::hpx::unseq);
}
DECLARE_UNITTEST(TestUninitializedCopyUnseq);

void TestUninitializedCopyUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestUninitializedCopy(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestUninitializedCopyUnseqOnSequencedExecutor);

void TestUninitializedCopyUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestUninitializedCopy(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestUninitializedCopyUnseqOnTestSyncExecutor);

void TestUninitializedCopyUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestUninitializedCopy(thrust::hpx::unseq.with(acs));
}
DECLARE_UNITTEST(TestUninitializedCopyUnseqWithAutoChunkSize);
