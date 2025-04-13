#include <thrust/reduce.h>

#include <unittest/unittest.h>

template <typename ExecutionPolicy>
void TestReduce(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;
  using T      = typename Vector::value_type;

  Vector v{1, -2, 3};

  // no initializer
  ASSERT_EQUAL(thrust::reduce(exec, v.begin(), v.end()), 2);

  // with initializer
  ASSERT_EQUAL(thrust::reduce(exec, v.begin(), v.end(), (T) 10), 12);
}

void TestReducePar()
{
  TestReduce(thrust::hpx::par);
}
DECLARE_UNITTEST(TestReducePar);

void TestReduceParOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestReduce(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestReduceParOnSequencedExecutor);

void TestReduceParOnForkJoinExecutor()
{
  hpx::execution::experimental::fork_join_executor exec{};

  TestReduce(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestReduceParOnForkJoinExecutor);

void TestReduceParWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestReduce(thrust::hpx::par.with(acs));
}
DECLARE_UNITTEST(TestReduceParWithAutoChunkSize);

void TestReduceParUnseq()
{
  TestReduce(thrust::hpx::par_unseq);
}
DECLARE_UNITTEST(TestReduceParUnseq);

void TestReduceParUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestReduce(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestReduceParUnseqOnSequencedExecutor);

void TestReduceParUnseqOnForkJoinExecutor()
{
  hpx::execution::experimental::fork_join_executor exec{};

  TestReduce(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestReduceParUnseqOnForkJoinExecutor);

void TestReduceParUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestReduce(thrust::hpx::par_unseq.with(acs));
}
DECLARE_UNITTEST(TestReduceParUnseqWithAutoChunkSize);

void TestReduceSeq()
{
  TestReduce(thrust::hpx::seq);
}
DECLARE_UNITTEST(TestReduceSeq);

void TestReduceSeqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestReduce(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestReduceSeqOnSequencedExecutor);

//void TestReduceSeqOnForkJoinExecutor()
//{
//  hpx::execution::experimental::fork_join_executor exec{};
//
//  TestReduce(thrust::hpx::seq.on(exec));
//}
//DECLARE_UNITTEST(TestReduceSeqOnForkJoinExecutor);

void TestReduceSeqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestReduce(thrust::hpx::seq.with(acs));
}
DECLARE_UNITTEST(TestReduceSeqWithAutoChunkSize);

void TestReduceUnseq()
{
  TestReduce(thrust::hpx::unseq);
}
DECLARE_UNITTEST(TestReduceUnseq);

void TestReduceUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestReduce(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestReduceUnseqOnSequencedExecutor);

//void TestReduceUnseqOnForkJoinExecutor()
//{
//  hpx::execution::experimental::fork_join_executor exec{};
//
//  TestReduce(thrust::hpx::unseq.on(exec));
//}
//DECLARE_UNITTEST(TestReduceUnseqOnForkJoinExecutor);

void TestReduceUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestReduce(thrust::hpx::unseq.with(acs));
}
DECLARE_UNITTEST(TestReduceUnseqWithAutoChunkSize);
