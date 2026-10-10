#include <chrono>
#include <map>
#include <set>
#include <thread>

#include <c10/cuda/CUDAException.h>
#include <gtest/gtest.h>

#include <xpool/debug/options.hpp>
#include <xpool/devkit/adapters.cuh>
#include <xpool/devkit/timeline/recorder.hpp>
#include <xpool/transport/arena.cuh>
#include <xpool/transport/protocol.cuh>

namespace {
XPOOL_KERNEL_FN void open_endpoint(xpool::transport::ArenaView arena) { arena.mailbox().open(); }

XPOOL_KERNEL_FN void round_trip(xpool::transport::ArenaView arena) {
  using Kind = xpool::hooks::TransportProtocolEventKind;
  xpool::abort_if(!arena.mailbox().try_begin_staging());
  auto request = xpool::transport::RequestMetadata{};
  request.layer_ordinal = 7;
  request.forward_mode = xpool::ffn::ForwardMode::Decode;
  request.output_requirement = xpool::ffn::OutputRequirement::PerRankComplete;
  request.dp_row_layout = xpool::ffn::DpRowLayout::None;
  auto instance = xpool::hooks::TransportInstanceProtocolEvent::Context{
      .arena = arena, .kind = Kind::RequestStagingStarted, .payload_rows = 2, .request = &request};
  auto agent = xpool::hooks::TransportAtnAgentProtocolEvent::Context{.arena = arena, .kind = Kind::RequestObserved};
  xpool::devkit::timeline::DeviceAdapter::observe(instance);
  arena.mailbox().request = request;
  arena.mailbox().payload_rows = 2;
  arena.mailbox().publish_request();
  instance.kind = Kind::RequestPublished;
  xpool::devkit::timeline::DeviceAdapter::observe(instance);
  xpool::devkit::timeline::DeviceAdapter::observe(agent);
  agent.kind = Kind::ExecutionStarted;
  xpool::devkit::timeline::DeviceAdapter::observe(agent);
  agent.kind = Kind::ExecutionCompleted;
  xpool::devkit::timeline::DeviceAdapter::observe(agent);
  arena.mailbox().publish_result(xpool::ffn::ResultCode::Ok);
  agent.kind = Kind::ResultPublished;
  agent.result_code = xpool::ffn::ResultCode::Ok;
  xpool::devkit::timeline::DeviceAdapter::observe(agent);
  instance.kind = Kind::ResultObserved;
  instance.result_code = xpool::ffn::ResultCode::Ok;
  xpool::devkit::timeline::DeviceAdapter::observe(instance);
  instance.kind = Kind::OutputCopied;
  xpool::devkit::timeline::DeviceAdapter::observe(instance);
  arena.mailbox().acknowledge();
}

TEST(TimelineHooks, SharedEndpointIdentityAndOperationsSurviveGraphReplay) {
  C10_CUDA_CHECK(cudaSetDevice(0));
  auto debug = xpool::debug::Options{};
  debug.timeline = {true, 262144, 262144, 4096};
  xpool::debug::configure(debug, 0);
  xpool::devkit::timeline::configure(1001, 1002, 0);
  xpool::devkit::timeline::set_generation(11, 12, 0);
  xpool::devkit::timeline::install_device();
  const auto layout = xpool::transport::ArenaLayout::create(0, 0, 0, 1, 0, 1, 4, 2, c10::ScalarType::Half);
  ASSERT_NE(layout.observation_identity_offset, 0);
  auto arena = xpool::transport::Arena::create(0, layout);
  const auto identity = xpool::transport::ObservationIdentity{1002, 1, 0, 1};
  C10_CUDA_CHECK(cudaMemcpy(arena.view().observation_identity(layout.observation_identity_offset), &identity,
                            sizeof(identity), cudaMemcpyHostToDevice));
  cudaStream_t stream{};
  cudaGraph_t graph{};
  cudaGraphExec_t executable{};
  C10_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  open_endpoint<<<1, 1, 0, stream>>>(arena.view());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  C10_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  round_trip<<<1, 1, 0, stream>>>(arena.view());
  C10_CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  C10_CUDA_CHECK(cudaGraphInstantiate(&executable, graph, 0));
  for (auto replay = 0; replay < 3; ++replay)
    C10_CUDA_CHECK(cudaGraphLaunch(executable, stream));
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  auto owner = xpool::devkit::timeline::device_recorder();
  owner->stop();
  auto operations = std::map<std::uint64_t, std::set<std::uint64_t>>{};
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!owner->drained() && std::chrono::steady_clock::now() < deadline) {
    if (auto chunk = owner->collect()) {
      for (const auto &record : chunk->records()) {
        EXPECT_EQ(record.generation_high, 11);
        EXPECT_EQ(record.generation_low, 12);
        EXPECT_EQ(record.endpoint_creator, 1002);
        EXPECT_EQ(record.endpoint_index, 1);
        EXPECT_EQ(record.layer, 7);
        operations[record.operation].insert(record.kind);
      }
      chunk->release();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  ASSERT_TRUE(owner->drained());
  EXPECT_EQ(owner->counters().committed, 24);
  ASSERT_EQ(operations.size(), 3);
  for (auto operation = std::uint64_t{1}; operation <= 3; ++operation)
    EXPECT_EQ(operations[operation], (std::set<std::uint64_t>{100, 101, 102, 103, 104, 105, 106, 107}));
  owner->close();
  auto host = xpool::devkit::timeline::host_recorder();
  host->stop();
  host->close();
  C10_CUDA_CHECK(cudaGraphExecDestroy(executable));
  C10_CUDA_CHECK(cudaGraphDestroy(graph));
  C10_CUDA_CHECK(cudaStreamDestroy(stream));
  arena.destroy();
}
} // namespace
