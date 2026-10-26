#include <xpool/devkit/capacity_counter.hpp>

#include <algorithm>
#include <limits>
#include <mutex>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <cooperative_groups.h>
#include <cuda/atomic>

#include <xpool/debug/options.cuh>
#include <xpool/devkit/adapters.cuh>
#include <xpool/fabric/arena.cuh>
#include <xpool/utils/time.cuh>

namespace xpool::devkit::capacity_counter {
namespace {
struct Header {
  unsigned long long origin = 0;
  unsigned long long last = 0;
  unsigned long long outside = 0;
  unsigned long long overflow = 0;
};
struct Storage {
  Header *header = nullptr;
  Bucket *buckets = nullptr;
  std::size_t count = 0;
};
XPOOL_DEVICE_CONST Storage storage{};
Storage host_storage{};
int host_device = -1;
std::mutex mutex;

XPOOL_DEVICE_FN void add(Metric metric, std::uint64_t amount = 1) {
  if (storage.header == nullptr) {
    return;
  }
  auto *header = storage.header;
  atomicCAS(&header->origin, 0ULL, xpool::utils::time::now());
  const auto now = xpool::utils::time::now();
  atomicMax(&header->last, static_cast<unsigned long long>(now));
  const auto origin = atomicAdd(&header->origin, 0ULL);
  const auto index = bucket_index(origin, now, storage.count);
  if (index >= storage.count) {
    atomicAdd(&header->outside, 1ULL);
    return;
  }
  auto &value = storage.buckets[index][static_cast<std::size_t>(metric)];
  auto counter = cuda::atomic_ref<std::uint64_t, cuda::thread_scope_device>(value);
  const auto previous = counter.fetch_add(amount, cuda::memory_order_relaxed);
  if (previous > std::numeric_limits<std::uint64_t>::max() - amount) {
    atomicExch(&header->overflow, 1ULL);
  }
}
} // namespace

void initialize() {
  const auto lock = std::lock_guard{mutex};
  const auto &options = xpool::debug::options().capacity_counter;
  if (!options.enable) {
    return;
  }
  TORCH_CHECK(host_storage.header == nullptr, "xpool capacity counter already initialized");
  TORCH_CHECK(options.bucket_count > 0 && options.bucket_count <= 360000,
              "xpool capacity counter bucket count outside [1, 360000]");
  C10_CUDA_CHECK(cudaGetDevice(&host_device));
  const auto bytes = sizeof(Header) + options.bucket_count * sizeof(Bucket);
  void *allocation = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&allocation, bytes));
  try {
    C10_CUDA_CHECK(cudaMemset(allocation, 0, bytes));
    const auto next =
        Storage{static_cast<Header *>(allocation),
                reinterpret_cast<Bucket *>(static_cast<char *>(allocation) + sizeof(Header)), options.bucket_count};
    C10_CUDA_CHECK(cudaMemcpyToSymbol(storage, &next, sizeof(next)));
    host_storage = next;
  } catch (...) {
    C10_CUDA_CHECK(cudaFree(allocation));
    throw;
  }
}

Snapshot finish() {
  const auto lock = std::lock_guard{mutex};
  TORCH_CHECK(host_storage.header != nullptr, "xpool capacity counter is not initialized");
  const auto device_guard = c10::cuda::CUDAGuard{static_cast<c10::DeviceIndex>(host_device)};
  Header header;
  C10_CUDA_CHECK(cudaMemcpy(&header, host_storage.header, sizeof(header), cudaMemcpyDeviceToHost));
  auto result = Snapshot{header.origin,
                         header.last,
                         header.outside,
                         header.overflow,
                         sizeof(Header) + host_storage.count * sizeof(Bucket),
                         {}};
  const auto count =
      header.origin == 0
          ? std::size_t{0}
          : std::min(host_storage.count, static_cast<std::size_t>((header.last - header.origin) / kBucketNs + 1));
  result.buckets.resize(count);
  if (count != 0) {
    C10_CUDA_CHECK(
        cudaMemcpy(result.buckets.data(), host_storage.buckets, count * sizeof(Bucket), cudaMemcpyDeviceToHost));
  }
  const auto empty = Storage{};
  C10_CUDA_CHECK(cudaMemcpyToSymbol(storage, &empty, sizeof(empty)));
  C10_CUDA_CHECK(cudaFree(host_storage.header));
  host_storage = {};
  host_device = -1;
  return result;
}

XPOOL_DEVICE_HOOK_FN(xpool::hooks::TransportInstanceProtocolEvent, DeviceAdapter::observe, context) {
  add(Metric::Transport);
}
XPOOL_DEVICE_HOOK_FN(xpool::hooks::TransportAtnAgentProtocolEvent, DeviceAdapter::observe, context) {
  add(Metric::Transport);
}
XPOOL_DEVICE_HOOK_FN(xpool::hooks::FabricAtnAgentProtocolEvent, DeviceAdapter::observe, context) {
  add(Metric::AtnAgent);
  if (context.kind == xpool::hooks::FabricAtnAgentProtocolEvent::Kind::SubmissionPrepared) {
    add(Metric::Association);
  }
}
XPOOL_DEVICE_HOOK_FN(xpool::hooks::FabricCoordinatorProtocolEvent, DeviceAdapter::observe, context) {
  add(Metric::Coordinator);
}
XPOOL_DEVICE_HOOK_FN(xpool::hooks::FabricFfnAgentProtocolEvent, DeviceAdapter::observe, context) {
  if (cooperative_groups::this_thread_block().thread_rank() != 0) {
    return;
  }
  add(Metric::FfnAgent);
  using Kind = xpool::hooks::FabricFfnAgentProtocolEvent::Kind;
  if (context.kind == Kind::ComputeStarted || context.kind == Kind::ComputeCompleted) {
    add(Metric::GraphActivityEstimate);
  }
  if (context.kind == Kind::RoutingMetadataPublished) {
    const auto &execution = *context.execution;
    const auto &layer = context.arena.layer_entry(execution.key.instance_index, execution.layer_ordinal);
    add(Metric::RoutingPublished);
    add(Metric::RoutingBytes, static_cast<std::uint64_t>(execution.payload_rows) * layer.effective_topk * 8);
  }
}
} // namespace xpool::devkit::capacity_counter
