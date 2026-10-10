#include <chrono>
#include <exception>
#include <future>
#include <memory>
#include <set>
#include <thread>

#include <c10/cuda/CUDAException.h>
#include <gtest/gtest.h>

#include <xpool/devkit/timeline/recorder.hpp>

namespace {
XPOOL_KERNEL_FN void produce(xpool::devkit::timeline::PoolView source, std::uint64_t tag) {
  auto value = xpool::devkit::timeline::Record{};
  value.kind = tag;
  source.commit(source.reserve(), value);
}

XPOOL_KERNEL_FN void produce_with_abort(xpool::devkit::timeline::PoolView source) {
  auto value = xpool::devkit::timeline::Record{};
  value.kind = 23;
  source.commit(source.reserve(), value);
  source.commit(source.reserve(), value, true);
}

TEST(TimelineDevice, SnapshotPreservesAbortCountersAndReadyChunkState) {
  C10_CUDA_CHECK(cudaSetDevice(0));
  const auto options = xpool::devkit::timeline::Options{true, 262144, 262144, 4096};
  auto owner = std::make_shared<xpool::devkit::timeline::Recorder>(options, 0);
  produce_with_abort<<<1, 1, 0, nullptr>>>(owner->view());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  owner->stop();
  std::size_t received = 0;
  std::size_t aborted = 0;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!owner->drained() && std::chrono::steady_clock::now() < deadline) {
    if (auto chunk = owner->collect()) {
      EXPECT_EQ(chunk->epoch(), 1);
      for (const auto &record : chunk->records()) {
        if (record.kind == 0) {
          EXPECT_EQ(record.sequence, 2);
          ++aborted;
        } else {
          EXPECT_EQ(record.kind, 23);
          EXPECT_EQ(record.sequence, 1);
          ++received;
        }
      }
      chunk->release();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  ASSERT_TRUE(owner->drained());
  EXPECT_EQ(received, 1);
  EXPECT_EQ(aborted, 1);
  const auto counts = owner->counters();
  EXPECT_EQ(counts.attempted, 2);
  EXPECT_EQ(counts.committed, 1);
  EXPECT_EQ(counts.dropped, 1);
  EXPECT_EQ(counts.contention, 0);
  EXPECT_EQ(counts.first_gap, 2);
  EXPECT_EQ(counts.last_gap, 2);
  EXPECT_EQ(counts.stopped, 1);
  EXPECT_EQ(counts.exhausted, 0);
  owner->close();
}

TEST(TimelineDevice, CrossStreamReceiptAndGraphReplayUseFreshSequences) {
  C10_CUDA_CHECK(cudaSetDevice(0));
  const auto options = xpool::devkit::timeline::Options{true, 262144, 262144, 4096};
  auto owner = std::make_shared<xpool::devkit::timeline::Recorder>(options, 0);
  cudaStream_t first{}, second{};
  C10_CUDA_CHECK(cudaStreamCreateWithFlags(&first, cudaStreamNonBlocking));
  C10_CUDA_CHECK(cudaStreamCreateWithFlags(&second, cudaStreamNonBlocking));
  cudaGraph_t graph{};
  cudaGraphExec_t executable{};
  C10_CUDA_CHECK(cudaStreamBeginCapture(first, cudaStreamCaptureModeGlobal));
  produce<<<1, 32, 0, first>>>(owner->view(), 17);
  C10_CUDA_CHECK(cudaStreamEndCapture(first, &graph));
  C10_CUDA_CHECK(cudaGraphInstantiate(&executable, graph, 0));
  for (auto replay = 0; replay < 3; ++replay)
    C10_CUDA_CHECK(cudaGraphLaunch(executable, first));
  produce<<<1, 32, 0, second>>>(owner->view(), 18);
  C10_CUDA_CHECK(cudaStreamSynchronize(first));
  C10_CUDA_CHECK(cudaStreamSynchronize(second));
  owner->stop();
  std::size_t received = 0;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!owner->drained() && std::chrono::steady_clock::now() < deadline) {
    if (auto chunk = owner->collect()) {
      for (const auto &record : chunk->records()) {
        EXPECT_TRUE(record.kind == 17 || record.kind == 18);
        EXPECT_GT(record.sequence, 0);
        ++received;
      }
      chunk->release();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  ASSERT_TRUE(owner->drained());
  EXPECT_EQ(received, 128);
  EXPECT_EQ(owner->counters().attempted, 128);
  EXPECT_EQ(owner->counters().dropped, 0);
  C10_CUDA_CHECK(cudaGraphExecDestroy(executable));
  C10_CUDA_CHECK(cudaGraphDestroy(graph));
  C10_CUDA_CHECK(cudaStreamDestroy(first));
  C10_CUDA_CHECK(cudaStreamDestroy(second));
  owner->close();
}

cudaStreamCaptureMode capture_mode() {
  auto mode = cudaStreamCaptureModeRelaxed;
  C10_CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  const auto current = mode;
  C10_CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  return current;
}

class TimelineCapture : public ::testing::TestWithParam<cudaStreamCaptureMode> {};

TEST_P(TimelineCapture, CollectsAndDrainsDuringGlobalCaptureWithoutChangingCallerMode) {
  C10_CUDA_CHECK(cudaSetDevice(0));
  const auto options = xpool::devkit::timeline::Options{true, 262144, 262144, 4096};
  auto owner = std::make_shared<xpool::devkit::timeline::Recorder>(options, 0);
  auto stopping = std::make_shared<xpool::devkit::timeline::Recorder>(options, 0);
  cudaStream_t capture{}, producer{};
  C10_CUDA_CHECK(cudaStreamCreateWithFlags(&capture, cudaStreamNonBlocking));
  C10_CUDA_CHECK(cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking));
  // Load producer and collector kernels before capture; then keep the actual
  // capture open until the background receipt/reclaim transaction completes.
  produce<<<1, 32, 0, producer>>>(owner->view(), 18);
  produce<<<1, 1, 0, producer>>>(stopping->view(), 19);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  C10_CUDA_CHECK(cudaStreamSynchronize(producer));
  stopping->stop();
  EXPECT_FALSE(owner->collect());
  EXPECT_FALSE(stopping->collect());
  auto sequences = std::set<std::uint64_t>{};
  std::promise<void> captured;
  auto capture_started = captured.get_future();
  const auto caller_mode = GetParam();
  auto collector = std::async(std::launch::async, [&] {
    C10_CUDA_CHECK(cudaSetDevice(0));
    auto previous = caller_mode;
    C10_CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&previous));
    capture_started.wait();
    auto stopped_records = std::size_t{0};
    auto empty_polls = std::size_t{0};
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline) {
      if (auto chunk = owner->collect()) {
        for (const auto &record : chunk->records()) {
          EXPECT_EQ(record.kind, 18);
          EXPECT_TRUE(sequences.insert(record.sequence).second);
        }
        chunk->release();
      } else {
        ++empty_polls;
      }
      EXPECT_EQ(capture_mode(), caller_mode);
      if (auto chunk = stopping->collect()) {
        for (const auto &record : chunk->records()) {
          EXPECT_EQ(record.kind, 19);
          ++stopped_records;
        }
        chunk->release();
      }
      EXPECT_EQ(capture_mode(), caller_mode);
      const auto drained = stopping->drained();
      EXPECT_EQ(capture_mode(), caller_mode);
      if (sequences.size() == 32 && drained) {
        EXPECT_EQ(stopped_records, 1);
        EXPECT_GT(empty_polls, 0);
        C10_CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&previous));
        return true;
      }
      std::this_thread::yield();
    }
    C10_CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&previous));
    return false;
  });
  C10_CUDA_CHECK(cudaStreamBeginCapture(capture, cudaStreamCaptureModeGlobal));
  produce<<<1, 32, 0, capture>>>(owner->view(), 17);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  captured.set_value();
  auto collected = false;
  std::exception_ptr error;
  try {
    collected = collector.get();
  } catch (...) {
    error = std::current_exception();
  }
  cudaGraph_t graph{};
  C10_CUDA_CHECK(cudaStreamEndCapture(capture, &graph));
  if (error)
    std::rethrow_exception(error);
  ASSERT_TRUE(collected);
  auto node_count = std::size_t{0};
  C10_CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &node_count));
  ASSERT_EQ(node_count, 1);
  cudaGraphNode_t node{};
  C10_CUDA_CHECK(cudaGraphGetNodes(graph, &node, &node_count));
  cudaGraphNodeType node_type{};
  C10_CUDA_CHECK(cudaGraphNodeGetType(node, &node_type));
  EXPECT_EQ(node_type, cudaGraphNodeTypeKernel);
  cudaGraphExec_t executable{};
  C10_CUDA_CHECK(cudaGraphInstantiate(&executable, graph, 0));
  for (auto replay = 0; replay < 3; ++replay)
    C10_CUDA_CHECK(cudaGraphLaunch(executable, capture));
  C10_CUDA_CHECK(cudaStreamSynchronize(capture));
  owner->stop();
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!owner->drained() && std::chrono::steady_clock::now() < deadline) {
    if (auto chunk = owner->collect()) {
      for (const auto &record : chunk->records()) {
        EXPECT_EQ(record.kind, 17);
        EXPECT_TRUE(sequences.insert(record.sequence).second);
      }
      chunk->release();
    }
    std::this_thread::yield();
  }
  ASSERT_TRUE(owner->drained());
  EXPECT_EQ(sequences.size(), 128);
  EXPECT_EQ(owner->counters().attempted, 128);
  EXPECT_EQ(owner->counters().committed, 128);
  EXPECT_EQ(owner->counters().dropped, 0);
  stopping->close();
  owner->close();
  C10_CUDA_CHECK(cudaGraphExecDestroy(executable));
  C10_CUDA_CHECK(cudaGraphDestroy(graph));
  C10_CUDA_CHECK(cudaStreamDestroy(producer));
  C10_CUDA_CHECK(cudaStreamDestroy(capture));
}

INSTANTIATE_TEST_SUITE_P(CallerModes, TimelineCapture,
                         ::testing::Values(cudaStreamCaptureModeGlobal, cudaStreamCaptureModeThreadLocal,
                                           cudaStreamCaptureModeRelaxed));

TEST(TimelineDevice, SaturatedPoolDropsNewRecordsAndPreservesCommittedPrefix) {
  C10_CUDA_CHECK(cudaSetDevice(0));
  const auto options = xpool::devkit::timeline::Options{true, 262144, 262144, 4096};
  auto owner = std::make_shared<xpool::devkit::timeline::Recorder>(options, 0);
  produce<<<64, 128, 0, nullptr>>>(owner->view(), 19);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  owner->stop();
  auto sequences = std::set<std::uint64_t>{};
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!owner->drained() && std::chrono::steady_clock::now() < deadline) {
    if (auto chunk = owner->collect()) {
      for (const auto &record : chunk->records()) {
        EXPECT_EQ(record.kind, 19);
        EXPECT_TRUE(sequences.insert(record.sequence).second);
      }
      chunk->release();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  ASSERT_TRUE(owner->drained());
  const auto counts = owner->counters();
  EXPECT_EQ(counts.attempted, 8192);
  EXPECT_GT(counts.dropped, 0);
  EXPECT_EQ(counts.attempted, counts.committed + counts.dropped);
  EXPECT_EQ(counts.committed, sequences.size());
  EXPECT_LE(counts.committed, owner->view().chunks * owner->view().capacity);
  owner->close();
}
} // namespace
