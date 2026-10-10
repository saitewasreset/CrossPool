#include <limits>
#include <mutex>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <xpool/debug/options.cuh>
#include <xpool/devkit/adapters.cuh>
#include <xpool/devkit/adapters.hpp>
#include <xpool/devkit/timeline/device.cuh>
#include <xpool/devkit/timeline/recorder.hpp>
#include <xpool/fabric/arena.cuh>
#include <xpool/transport/arena.cuh>
#include <xpool/utils/time.cuh>

namespace xpool::devkit::timeline {

namespace {

struct GenerationContext {
  std::uint64_t high = 0;
  std::uint64_t low = 0;
  std::uint64_t pe = 0;
};

XPOOL_DEVICE_CONST GenerationContext generation{};
std::mutex endpoint_mutex;
std::uint64_t endpoint_sequence = 0;
std::size_t live_owned_endpoints = 0;
GenerationContext configured_generation{};

Record host_event(std::uint64_t kind) {
  Record result{};
  result.kind = kind;
  return result;
}

XPOOL_DEVICE_FN Record device_event(std::uint64_t kind, std::uint64_t site) {
  Record result{};
  result.timestamp = xpool::utils::time::now();
  result.kind = kind;
  result.site = site;
  result.generation_high = generation.high;
  result.generation_low = generation.low;
  return result;
}

XPOOL_DEVICE_FN void observe_transport(xpool::transport::ArenaView arena, xpool::hooks::TransportProtocolEventKind kind,
                                       std::uint64_t site, xpool::ffn::ResultCode result_code,
                                       const xpool::transport::RequestMetadata *request, std::size_t rows) {
  using Kind = xpool::hooks::TransportProtocolEventKind;
  if (!xpool::debug::options().timeline.enable)
    return;
  auto *identity = arena.observation_identity(arena.layout().observation_identity_offset);
  if (!identity || !identity->valid)
    return;
  if (kind == Kind::RequestStagingStarted) {
    if (identity->operation == std::numeric_limits<std::uint64_t>::max()) {
      identity->valid = 0;
      if (device_view.base)
        Atomic{device_view.counters().exhausted}.store(1);
      return;
    }
    ++identity->operation;
  }
  std::uint64_t event = 0;
  switch (kind) {
  case Kind::RequestStagingStarted:
    event = 100;
    break;
  case Kind::RequestPublished:
    event = 101;
    break;
  case Kind::RequestObserved:
    event = 102;
    break;
  case Kind::ExecutionStarted:
    event = 103;
    break;
  case Kind::ExecutionCompleted:
    event = 104;
    break;
  case Kind::ResultPublished:
    event = 105;
    break;
  case Kind::ResultObserved:
    event = 106;
    break;
  case Kind::OutputCopied:
    event = 107;
    break;
  case Kind::Closed:
    event = 108;
    break;
  default:
    return;
  }
  auto record = device_event(event, site);
  record.endpoint_creator = identity->creator;
  record.endpoint_index = identity->endpoint;
  record.operation = identity->operation;
  record.instance = arena.layout().instance_index;
  record.layer = request ? request->layer_ordinal : arena.mailbox().request.layer_ordinal;
  if (kind == Kind::ResultPublished || kind == Kind::ResultObserved || kind == Kind::Closed)
    record.result = static_cast<std::uint64_t>(result_code);
  record.rows = request ? rows : arena.mailbox().payload_rows;
  timeline::record(record);
}

} // namespace

void set_generation(std::uint64_t high, std::uint64_t low, std::uint64_t pe) {
  if (!xpool::debug::options().timeline.enable)
    return;
  TORCH_CHECK((configured_generation.high == 0 && configured_generation.low == 0) ||
                  (configured_generation.high == high && configured_generation.low == low),
              "xpool timeline generation identity changed");
  configured_generation.high = high;
  configured_generation.low = low;
  configured_generation.pe = pe;
  C10_CUDA_CHECK(cudaMemcpyToSymbol(generation, &configured_generation, sizeof(configured_generation)));
}

XPOOL_HOST_HOOK_FN(xpool::hooks::TransportEndpointOpenPostEvent, HostAdapter::observe, context) {
  if (!xpool::debug::options().timeline.enable)
    return;
  const auto guard = c10::cuda::CUDAGuard(context.device);
  install_device();
  if (context.site == xpool::hooks::TransportEndpointSite::AtnAgent) {
    const auto lock = std::lock_guard{endpoint_mutex};
    TORCH_CHECK(live_owned_endpoints < 256 && endpoint_sequence != std::numeric_limits<std::uint64_t>::max(),
                "xpool timeline endpoint metadata allowance exhausted");
    const xpool::transport::ObservationIdentity identity{
        .creator = device_identity(), .endpoint = ++endpoint_sequence, .operation = 0, .valid = 1};
    C10_CUDA_CHECK(cudaMemcpy(context.arena.observation_identity(context.layout.observation_identity_offset), &identity,
                              sizeof(identity), cudaMemcpyHostToDevice));
    ++live_owned_endpoints;
  }
  host_recorder()->record(host_event(3));
}
XPOOL_HOST_HOOK_FN(xpool::hooks::TransportEndpointClosePreEvent, HostAdapter::observe, context) {
  if (!xpool::debug::options().timeline.enable)
    return;
  host_recorder()->record(host_event(4));
  if (context.site == xpool::hooks::TransportEndpointSite::AtnAgent) {
    const auto lock = std::lock_guard{endpoint_mutex};
    --live_owned_endpoints;
  }
}
XPOOL_HOST_HOOK_FN(xpool::hooks::FabricJoinPostEvent, HostAdapter::observe, context) {
  if (!xpool::debug::options().timeline.enable)
    return;
  install_device();
  host_recorder()->record(host_event(5));
}
XPOOL_HOST_HOOK_FN(xpool::hooks::FabricFinalizePreEvent, HostAdapter::observe, context) {
  if (xpool::debug::options().timeline.enable)
    host_recorder()->record(host_event(6));
}
XPOOL_DEVICE_HOOK_FN(xpool::hooks::TransportInstanceProtocolEvent, DeviceAdapter::observe, context) {
  if (threadIdx.x == 0)
    observe_transport(context.arena, context.kind, 0, context.result_code, context.request, context.payload_rows);
}
XPOOL_DEVICE_HOOK_FN(xpool::hooks::TransportAtnAgentProtocolEvent, DeviceAdapter::observe, context) {
  if (threadIdx.x == 0)
    observe_transport(context.arena, context.kind, 1, context.result_code, context.request, context.payload_rows);
}
XPOOL_DEVICE_HOOK_FN(xpool::hooks::FabricFfnAgentProtocolEvent, DeviceAdapter::observe, context) {
  if (!xpool::debug::options().timeline.enable || threadIdx.x != 0)
    return;
  using Kind = xpool::hooks::FabricFfnAgentProtocolEvent::Kind;
  std::uint64_t event = 0;
  switch (context.kind) {
  case Kind::LaneExecutionObserved:
    event = 200;
    break;
  case Kind::ComputeStarted:
    event = 201;
    break;
  case Kind::ComputeCompleted:
    event = 202;
    break;
  default:
    return;
  }
  const auto &execution = *context.execution;
  auto value = device_event(event, generation.pe);
  value.operation = execution.key.invocation_sequence;
  value.instance = execution.key.instance_index;
  value.layer = execution.layer_ordinal;
  value.lane = context.executor_lane_index;
  value.lease = execution.executor_lease_sequence;
  value.rows = execution.payload_rows;
  timeline::record(value);
}
XPOOL_DEVICE_HOOK_FN(xpool::hooks::FabricCoordinatorProtocolEvent, DeviceAdapter::observe, context) {
  if (!xpool::debug::options().timeline.enable || threadIdx.x != 0)
    return;
  using Kind = xpool::hooks::FabricCoordinatorProtocolEvent::Kind;
  if (context.kind != Kind::Scheduled && context.kind != Kind::LaneReleased)
    return;
  // Context carries production-owned identity even after scheduler retirement.
  auto value = device_event(context.kind == Kind::Scheduled ? 203 : 204, generation.pe);
  value.operation = context.invocation->key.invocation_sequence;
  value.instance = context.instance_index;
  value.lane = context.executor_lane_index;
  value.lease = context.executor_lease_sequence;
  timeline::record(value);
}

} // namespace xpool::devkit::timeline
