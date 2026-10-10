#include <chrono>
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
