#include <thrust/copy.h>

#include "test_executor.h"
#include <unittest/unittest.h>

struct is_even
{
  __host__ __device__ bool operator()(int x) const
  {
    return (x % 2) == 0;
  }
};

struct is_true
{
  __host__ __device__ bool operator()(int x) const
  {
    return x != 0;
  }
};

template <typename ExecutionPolicy>
void TestCopyIf(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;

  // Test 1: copy_if without stencil
  {
    const Vector input{1, 2, 3, 4, 5, 6};
    Vector output(6, 0);

    auto out_end = thrust::copy_if(exec, input.begin(), input.end(), output.begin(), is_even{});
    ASSERT_EQUAL(out_end - output.begin(), 3);
    ASSERT_EQUAL(output[0], 2);
    ASSERT_EQUAL(output[1], 4);
    ASSERT_EQUAL(output[2], 6);
  }

  // Test 2: copy_if with stencil
  {
    const Vector input{10, 20, 30, 40, 50, 60};
    const Vector stencil{1, 0, 1, 0, 1, 0};
    Vector output(6, 0);

    auto out_end = thrust::copy_if(exec, input.begin(), input.end(), stencil.begin(), output.begin(), is_true{});
    ASSERT_EQUAL(out_end - output.begin(), 3);
    ASSERT_EQUAL(output[0], 10);
    ASSERT_EQUAL(output[1], 30);
    ASSERT_EQUAL(output[2], 50);
  }

  // Test 3: empty range
  {
    const Vector empty_in;
    Vector empty_out;

    auto out_end = thrust::copy_if(exec, empty_in.begin(), empty_in.end(), empty_out.begin(), is_even{});
    ASSERT_EQUAL(out_end - empty_out.begin(), 0);
  }
}

void TestCopyIfPar()
{
  TestCopyIf(thrust::hpx::par);
}
DECLARE_UNITTEST(TestCopyIfPar);

void TestCopyIfParOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestCopyIf(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestCopyIfParOnSequencedExecutor);

void TestCopyIfParOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestCopyIf(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestCopyIfParOnTestSyncExecutor);

void TestCopyIfParWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestCopyIf(thrust::hpx::par.with(acs));
}
DECLARE_UNITTEST(TestCopyIfParWithAutoChunkSize);

void TestCopyIfParUnseq()
{
  TestCopyIf(thrust::hpx::par_unseq);
}
DECLARE_UNITTEST(TestCopyIfParUnseq);

void TestCopyIfParUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestCopyIf(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestCopyIfParUnseqOnSequencedExecutor);

void TestCopyIfParUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestCopyIf(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestCopyIfParUnseqOnTestSyncExecutor);

void TestCopyIfParUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestCopyIf(thrust::hpx::par_unseq.with(acs));
}
DECLARE_UNITTEST(TestCopyIfParUnseqWithAutoChunkSize);

void TestCopyIfSeq()
{
  TestCopyIf(thrust::hpx::seq);
}
DECLARE_UNITTEST(TestCopyIfSeq);

void TestCopyIfSeqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestCopyIf(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestCopyIfSeqOnSequencedExecutor);

void TestCopyIfSeqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestCopyIf(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestCopyIfSeqOnTestSyncExecutor);

void TestCopyIfSeqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestCopyIf(thrust::hpx::seq.with(acs));
}
DECLARE_UNITTEST(TestCopyIfSeqWithAutoChunkSize);

void TestCopyIfUnseq()
{
  TestCopyIf(thrust::hpx::unseq);
}
DECLARE_UNITTEST(TestCopyIfUnseq);

void TestCopyIfUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestCopyIf(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestCopyIfUnseqOnSequencedExecutor);

void TestCopyIfUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestCopyIf(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestCopyIfUnseqOnTestSyncExecutor);

void TestCopyIfUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestCopyIf(thrust::hpx::unseq.with(acs));
}
DECLARE_UNITTEST(TestCopyIfUnseqWithAutoChunkSize);
