#include <chrono>
#include <cstdint>
#include <thread>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <xpool/debug/options.hpp>
#include <xpool/devkit/adapters.cuh>
#include <xpool/devkit/capacity_counter.hpp>
#include <xpool/macros.hpp>

namespace {
XPOOL_KERNEL_FN void emit() {
  auto context = xpool::hooks::TransportInstanceProtocolEvent::Context{
      .arena = {}, .kind = xpool::hooks::TransportProtocolEventKind::RequestPublished};
  xpool::devkit::capacity_counter::DeviceAdapter::observe(context);
  auto collective = xpool::hooks::FabricFfnAgentProtocolEvent::Context{
      .arena = {}, .kind = xpool::hooks::FabricFfnAgentProtocolEvent::Kind::ComputeStarted};
  xpool::devkit::capacity_counter::DeviceAdapter::observe(collective);
}

TEST(CapacityCounterCuda, ConcurrentHooksAndWindowExhaustionRemainExplicit) {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
    GTEST_SKIP() << "Device is unavailable";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
  const auto previous = xpool::debug::options_h;
  xpool::debug::options_h.capacity_counter = {true, 1000};
  xpool::devkit::capacity_counter::initialize();
  emit<<<8, 32, 0, stream>>>();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  const auto result = xpool::devkit::capacity_counter::finish();
  std::uint64_t transport = 0, collective = 0, graph = 0;
  for (const auto &bucket : result.buckets) {
    transport += bucket[0];
    collective += bucket[3];
    graph += bucket[7];
  }
  EXPECT_EQ(transport, 256);
  EXPECT_EQ(collective, 8);
  EXPECT_EQ(graph, 8);
  EXPECT_EQ(result.outside_window, 0);
  EXPECT_EQ(result.arithmetic_overflow, 0);

  xpool::debug::options_h.capacity_counter = {true, 1};
  xpool::devkit::capacity_counter::initialize();
  emit<<<1, 1, 0, stream>>>();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  emit<<<1, 1, 0, stream>>>();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
  const auto exhausted = xpool::devkit::capacity_counter::finish();
  EXPECT_GE(exhausted.outside_window, 3);
  EXPECT_EQ(exhausted.arithmetic_overflow, 0);
  xpool::debug::options_h = previous;
  EXPECT_THROW(xpool::devkit::capacity_counter::finish(), c10::Error);
}
} // namespace
