#include <thrust/generate.h>

#include "test_executor.h"
#include <unittest/unittest.h>

struct value_generator
{
  __host__ __device__ int operator()() const
  {
    return 42;
  }
};

template <typename ExecutionPolicy>
void TestGenerate(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;

  // Test 1: generate
  {
    Vector data(5, 0);
    thrust::generate(exec, data.begin(), data.end(), value_generator{});
    for (int i = 0; i < 5; ++i)
    {
      ASSERT_EQUAL(data[i], 42);
    }
  }

  // Test 2: generate_n
  {
    Vector data(5, 0);
    auto end = thrust::generate_n(exec, data.begin(), 3, value_generator{});
    ASSERT_EQUAL(end - data.begin(), 3);
    ASSERT_EQUAL(data[0], 42);
    ASSERT_EQUAL(data[1], 42);
    ASSERT_EQUAL(data[2], 42);
    ASSERT_EQUAL(data[3], 0);
    ASSERT_EQUAL(data[4], 0);
  }

  // Test 3: empty range
  {
    Vector empty;
    thrust::generate(exec, empty.begin(), empty.end(), value_generator{});
    ASSERT_EQUAL(empty.size(), 0);

    auto end = thrust::generate_n(exec, empty.begin(), 0, value_generator{});
    ASSERT_EQUAL(end - empty.begin(), 0);
  }
}

void TestGeneratePar()
{
  TestGenerate(thrust::hpx::par);
}
DECLARE_UNITTEST(TestGeneratePar);

void TestGenerateParOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestGenerate(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestGenerateParOnSequencedExecutor);

void TestGenerateParOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestGenerate(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestGenerateParOnTestSyncExecutor);

void TestGenerateParWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestGenerate(thrust::hpx::par.with(acs));
}
DECLARE_UNITTEST(TestGenerateParWithAutoChunkSize);

void TestGenerateParUnseq()
{
  TestGenerate(thrust::hpx::par_unseq);
}
DECLARE_UNITTEST(TestGenerateParUnseq);

void TestGenerateParUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestGenerate(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestGenerateParUnseqOnSequencedExecutor);

void TestGenerateParUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestGenerate(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestGenerateParUnseqOnTestSyncExecutor);

void TestGenerateParUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestGenerate(thrust::hpx::par_unseq.with(acs));
}
DECLARE_UNITTEST(TestGenerateParUnseqWithAutoChunkSize);

void TestGenerateSeq()
{
  TestGenerate(thrust::hpx::seq);
}
DECLARE_UNITTEST(TestGenerateSeq);

void TestGenerateSeqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestGenerate(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestGenerateSeqOnSequencedExecutor);

void TestGenerateSeqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestGenerate(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestGenerateSeqOnTestSyncExecutor);

void TestGenerateSeqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestGenerate(thrust::hpx::seq.with(acs));
}
DECLARE_UNITTEST(TestGenerateSeqWithAutoChunkSize);

void TestGenerateUnseq()
{
  TestGenerate(thrust::hpx::unseq);
}
DECLARE_UNITTEST(TestGenerateUnseq);

void TestGenerateUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestGenerate(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestGenerateUnseqOnSequencedExecutor);

void TestGenerateUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestGenerate(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestGenerateUnseqOnTestSyncExecutor);

void TestGenerateUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestGenerate(thrust::hpx::unseq.with(acs));
}
DECLARE_UNITTEST(TestGenerateUnseqWithAutoChunkSize);
